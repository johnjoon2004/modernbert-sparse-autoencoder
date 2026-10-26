import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from src.sae import SparseAutoencoder
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_activation_distance_matrix(
    sae: SparseAutoencoder, dataloader: DataLoader, live_feature_indices: list, device: torch.device, num_samples: int
) -> np.ndarray:

    sae.eval()
    sae.to(device)

    all_feature_activations = []
    samples_collected = 0

    with torch.no_grad():
        for inputs, attention_mask, _ in tqdm(
            dataloader,
            desc=f"활성화 기반 거리(activation_distance) 계산 중... (샘플 {num_samples}건 수집)",
            total=num_samples,
        ):
            if samples_collected >= num_samples:
                break

            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)

            # [CLS] ('<s>') and [SEP] ('<\\s>') 토큰 제외
            attn_mask = attention_mask.clone()
            attn_mask[:, 0] = 0  # [CLS] 토큰 마스킹
            seq_lengths = attention_mask.sum(dim=1)
            sep_indices = seq_lengths - 1
            attn_mask[torch.arange(attn_mask.size(0)), sep_indices] = 0  # [SEP] 토큰 마스킹

            # SAE 통과
            processed_inputs = sae._preprocess_input(inputs)
            pre_activations = sae.encoder(processed_inputs)
            post_activations = sae._apply_activation(pre_activations)  # (Batch, SeqLen, FeatureDim)

            # 마스크 적용
            mask = attn_mask.unsqueeze(-1)  # (Batch, SeqLen, 1)
            post_masked = post_activations * mask  # (Batch, SeqLen, FeatureDim)

            all_feature_activations.append(post_masked.cpu())
            samples_collected += inputs.size(0)

    # 모든 배치 스택
    all_batches = torch.cat(all_feature_activations, dim=0)
    print("[DEBUG] BEF: ", all_batches.shape)

    # 정확히 N개 샘플로 자르기
    all_batches = all_batches[:num_samples]  # (N_Samples, SeqLen, FeatureDim)
    print("[DEBUG] AFT: ", all_batches.shape)

    # live_feature만 필터링
    live_indices_tensor = torch.tensor(live_feature_indices, dtype=torch.long)
    live_activations = all_batches.index_select(
        dim=2, index=live_indices_tensor
    )  # (N_Samples, SeqLen, NumLiveFeatures)
    print(f"[DEBUG]  -> 수집된 활성화 텐서 Shape: {live_activations.shape}")

    # (NumLiveFeatures, N_Samples, SeqLen) -> (NumLiveFeatures, N_Samples * SeqLen)
    # 각 특징(행)이 N_Samples * SeqLen 길이의 "플랫 벡터"가 됨
    transposed_activations = live_activations.permute(2, 0, 1)
    flat_feature_vectors = transposed_activations.reshape(len(live_feature_indices), -1)
    print(f"[DEBUG]  -> 플랫 특징 벡터 행렬 Shape: {flat_feature_vectors.shape}")

    # Flat 벡터의 Jaccard 거리 계산
    # Binarization: 활성화 값(>0)이 아닌, 활성화 여부(True/False)로 변경
    feature_matrix_np = flat_feature_vectors.numpy()
    binary_flat_vectors = feature_matrix_np > 0
    binary_sparse_vectors = csr_matrix(binary_flat_vectors, dtype=np.float64)

    print(f"[DEBUG]  -> 희소 자카드 거리 계산 시작...")
    # activation_distance_matrix = pairwise_distances(binary_sparse_vectors, metric="jaccard")

    intersection = binary_sparse_vectors.dot(binary_sparse_vectors.T)
    intersection = intersection.toarray()  # (N, N) 밀집 행렬로 변환

    row_sums = binary_sparse_vectors.sum(axis=1)  # 각 행의 1의 총 개수 (각 특징의 총 활성화 토큰 수) 계산
    row_sums = np.asarray(row_sums)  # (N, 1) 배열로 변환

    union = row_sums + row_sums.T - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard_similarity = np.nan_to_num(intersection / union, nan=1.0)

    activation_distance_matrix = 1.0 - jaccard_similarity  # Jaccard 거리 계산 (1 - 유사도)
    np.fill_diagonal(activation_distance_matrix, 0)  # 자기 자신과의 거리는 0으로 보정 (부동소수점 오차 방지)
    print(f"[DEBUG]  -> 자카드(Jaccard) 활성화 거리 행렬 Shape: {activation_distance_matrix.shape}")

    return activation_distance_matrix


def cluster_with_protection(
    distance_matrix: np.ndarray, significance_scores: np.ndarray, top_n_to_protect: int, eps: float, min_samples: int
) -> np.ndarray:
    print(f"--- DBSCAN (eps={eps}, min_samples={min_samples}) 실행 ---")

    # --- 1. 표준 DBSCAN 실행 ---
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(distance_matrix)

    labels = db.labels_

    # DBSCAN 결과 요약
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(f"표준 결과: {n_clusters_}개 클러스터, {n_noise_}개 노이즈")

    # --- 2. "구출" 대상 선정 ---
    # Significance 점수가 높은 순으로 "로컬 인덱스" (0 ~ N-1)를 찾습니다.
    # np.argsort()는 오름차순이므로, [-N:]으로 뒤에서 N개를 선택합니다.
    top_n_local_indices = np.argsort(significance_scores)[-top_n_to_protect:]

    # --- 3. "구출" 작업 수행 ---
    labels_corrected = np.copy(labels)
    rescued_count = 0

    # DBSCAN의 "핵심 지점(Core Point)" 인덱스를 가져옵니다.
    # 클러스터의 "중심" 역할을 하는 지점들입니다.
    core_sample_indices = db.core_sample_indices_

    if len(core_sample_indices) > 0:
        # 3-1. (표준 방식) "핵심 지점"이 존재하는 경우
        # 모든 특징에서 -> 모든 "핵심 지점"까지의 거리만 추출 (N, K)
        core_point_distances = distance_matrix[:, core_sample_indices]
        # 각 "핵심 지점"의 클러스터 라벨
        core_point_labels = labels[core_sample_indices]

        for local_idx in top_n_local_indices:
            if labels[local_idx] == -1:  # 만약 Top-N 특징이 노이즈라면
                # 이 특징(local_idx)에서 가장 가까운 "핵심 지점"을 찾습니다.
                distances_to_cores = core_point_distances[local_idx]
                nearest_core_idx_in_core_array = np.argmin(distances_to_cores)

                # 가장 가까운 핵심 지점의 클러스터 ID로 할당(구출)합니다.
                rescue_cluster_id = core_point_labels[nearest_core_idx_in_core_array]
                labels_corrected[local_idx] = rescue_cluster_id
                rescued_count += 1

    else:
        # 3-2. (대체 방식) "핵심 지점"이 없는 경우 (eps가 너무 작아서 모두 노이즈일 때)
        print("Warning: 핵심 지점(Core Point)이 없습니다. 가장 가까운 '비-노이즈' 지점에 할당합니다.")
        non_noise_indices = np.where(labels != -1)[0]

        if len(non_noise_indices) > 0:
            non_noise_distances = distance_matrix[:, non_noise_indices]
            non_noise_labels = labels[non_noise_indices]

            for local_idx in top_n_local_indices:
                if labels[local_idx] == -1:
                    distances_to_neighbors = non_noise_distances[local_idx]
                    nearest_neighbor_idx_in_array = np.argmin(distances_to_neighbors)
                    rescue_cluster_id = non_noise_labels[nearest_neighbor_idx_in_array]
                    labels_corrected[local_idx] = rescue_cluster_id
                    rescued_count += 1
        else:
            print("Error: 모든 특징이 노이즈(-1)입니다. 구출이 불가능합니다.")

    print(f"--- 'Top-N' 특징 {rescued_count}개를 노이즈에서 구출 완료 ---")

    # 최종 결과 요약
    n_clusters_final = len(set(labels_corrected)) - (1 if -1 in labels_corrected else 0)
    n_noise_final = list(labels_corrected).count(-1)
    print(f"최종 결과: {n_clusters_final}개 클러스터, {n_noise_final}개 노이즈")

    return labels_corrected


def visualize_clusters(final_distance_matrix: np.ndarray, labels: np.ndarray, live_feature_indices: list):
    mds = MDS(
        n_components=2,  # 2D로 축소
        dissimilarity="precomputed",  # (중요) 거리 행렬을 직접 입력
        random_state=42,
        n_init=4,  # 안정적인 결과를 위해 4번 시도
        max_iter=300,
    )
    pos_2d = mds.fit_transform(final_distance_matrix)

    plt.figure(figsize=(14, 10))

    # 클러스터 ID별로 색상 매핑
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)
    colors = plt.get_cmap("Spectral", n_clusters)

    for cluster_id in unique_labels:
        # 이 클러스터에 해당하는 점들의 (로컬) 인덱스
        indices = np.where(labels == cluster_id)[0]

        # 2D 위치
        cluster_pos = pos_2d[indices]

        if cluster_id == -1:
            # 노이즈(-1)는 회색으로 작게 표시
            plt.scatter(cluster_pos[:, 0], cluster_pos[:, 1], c="lightgray", label="Noise (-1)", s=15, alpha=0.5)
        else:
            # 일반 클러스터
            plt.scatter(
                cluster_pos[:, 0],
                cluster_pos[:, 1],
                color=colors(cluster_id / n_clusters),
                label=f"Cluster {cluster_id}",
                s=50,
                alpha=0.8,
            )

    # (선택) 각 점에 피처 인덱스 태그 달기
    # (점이 너무 많으면 지저분해 보일 수 있음)
    show_labels = True
    if show_labels:
        for i in range(pos_2d.shape[0]):
            # 점 10개 중 1개만 (또는 노이즈가 아닌 것만) 표시
            if labels[i] != -1:
                plt.text(
                    pos_2d[i, 0],
                    pos_2d[i, 1],
                    f" #{live_feature_indices[i]}",  # f"#{feature_index}"
                    fontsize=7,
                    alpha=0.6,
                )

    plt.title("Feature Cluster Visualization (MDS)", fontsize=16)
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def show_heatmap(labels: np.ndarray | list, final_distance_matrix: np.ndarray):
    unique_labels = sorted([l for l in set(labels) if l != -1])
    n_clusters = len(unique_labels)

    if n_clusters == 0:
        print("오류: 유의미한 클러스터가 없습니다. (노이즈 -1 뿐)")
    else:
        # (K, K) 크기의 빈 행렬 생성
        cluster_distance_matrix = np.zeros((n_clusters, n_clusters))

        print("클러스터 간 평균 거리 계산 중...")
        for i in range(n_clusters):
            cluster_id_i = unique_labels[i]
            indices_i = np.where(labels == cluster_id_i)[0]  # 클러스터 i의 로컬 인덱스

            for j in range(i, n_clusters):
                cluster_id_j = unique_labels[j]
                indices_j = np.where(labels == cluster_id_j)[0]  # 클러스터 j의 로컬 인덱스

                # (N_i, N_j) 부분 행렬 추출
                sub_matrix_indices = np.ix_(indices_i, indices_j)
                distance_sub_matrix = final_distance_matrix[sub_matrix_indices]

                # 모든 쌍의 평균 거리 계산
                mean_dist = distance_sub_matrix.mean()

                cluster_distance_matrix[i, j] = mean_dist
                cluster_distance_matrix[j, i] = mean_dist  # 대칭

        # 3. Pandas DataFrame으로 변환 (Seaborn 시각화용)
        cluster_labels_str = [f"C{l}" for l in unique_labels]
        df_heatmap = pd.DataFrame(cluster_distance_matrix, index=cluster_labels_str, columns=cluster_labels_str)

        # 4. Seaborn 히트맵 시각화
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            df_heatmap,
            annot=True,  # 셀에 숫자 표시
            fmt=".2f",  # 소수점 2자리
            cmap="vlag_r",  # (중요) vlag_r: 낮을수록(가까움) 파란색, 높을수록(멈) 빨간색
            linewidths=0.5,
        )
        plt.title("Cluster-to-Cluster Average Distance Heatmap", fontsize=16)
        plt.xlabel("Cluster ID")
        plt.ylabel("Cluster ID")
        plt.tight_layout()
