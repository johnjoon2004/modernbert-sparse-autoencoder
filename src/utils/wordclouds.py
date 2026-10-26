import math
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import torch
from src.global_settings import settings
from src.sae import SparseAutoencoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from wordcloud import WordCloud


def collect_cluster_tokens(
    activation_loader: DataLoader,
    sae: SparseAutoencoder,
    num_samples: int = 1000,
    text_df: pd.DataFrame | None = None,
    tokenizer: AutoTokenizer | None = None,
    max_seq_len: int | None = None,
    device: torch.device | None = None,
) -> dict:
    text_df = text_df or pd.read_csv(settings.text_dataset_path)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(settings.lm_weights_path)
    max_seq_len = max_seq_len or settings.prev_seq_len
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sae.eval()
    sae.to(device)

    feature_to_tokens = defaultdict(list)
    samples_collected = 0
    with torch.no_grad():
        for inputs, attention_mask, text_indices in tqdm(
            activation_loader, desc="Processing samples...", total=num_samples
        ):
            if samples_collected >= num_samples:
                break

            current_batch_size = inputs.size(0)

            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)

            attn_mask = attention_mask.clone()
            attn_mask[:, 0] = 0  # Mask CLS
            seq_lengths = attention_mask.sum(dim=1)
            sep_indices = seq_lengths - 1
            attn_mask[torch.arange(current_batch_size), sep_indices] = 0  # Mask SEP

            processed_inputs = sae._preprocess_input(inputs)
            pre_activations = sae.encoder(processed_inputs)
            post_activations = sae._apply_activation(pre_activations)  # (Batch, SeqLen, FeatureDim)

            mask = attn_mask.unsqueeze(-1)  # (Batch, SeqLen, 1)
            post_masked = post_activations * mask
            active_mask = post_masked > 0  # (Batch, SeqLen, FeatureDim)

            batch_texts = text_df.loc[text_indices.cpu().numpy()]["text"].tolist()
            tokenized_batch = tokenizer(
                batch_texts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt"
            )
            batch_input_ids = tokenized_batch["input_ids"]
            for b_idx in range(current_batch_size):
                sample_tokens = tokenizer.convert_ids_to_tokens(batch_input_ids[b_idx])
                for s_idx in range(max_seq_len):
                    active_feature_indices_here = torch.where(active_mask[b_idx, s_idx])[
                        0
                    ]  # (FeatureDim,) boolean mask
                    if active_feature_indices_here.numel() > 0:
                        token_str = sample_tokens[s_idx]
                        if token_str == tokenizer.pad_token:
                            continue
                        for feat_idx_tensor in active_feature_indices_here:
                            feat_idx = feat_idx_tensor.item()
                            feature_to_tokens[feat_idx].append(token_str)

            samples_collected += current_batch_size

    print(f"Token collection complete. Total {len(feature_to_tokens)} features activated.")
    return feature_to_tokens


def show_wordcloud(cluster_id: int, clusters: dict[list], feature_to_tokens: dict):
    cluster_features = clusters[cluster_id]
    cluster_tokens = []
    for feat_idx in cluster_features:
        tokens = feature_to_tokens.get(feat_idx, [])
        cluster_tokens.extend(tokens)

    if not cluster_tokens:
        print(f"Cluster {cluster_id} has no activated tokens.")
        return

    text_to_plot = " ".join(cluster_tokens)
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            random_state=42,
            font_path=str(settings.font_path),
        ).generate(text_to_plot)
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for Cluster {cluster_id}", fontsize=16)
        plt.show()
    except Exception as e:
        print(f"Error generating word cloud for Cluster {cluster_id}: {e}")


def show_all_wordclouds(clusters: dict[list], feature_to_tokens: dict, skip_noise_cluster: bool = False, **kwargs):
    width = kwargs.get("width", 400)
    height = kwargs.get("height", 200)

    fqi_map: dict[int, int] = kwargs.get("fqi_map", {})

    num_clusters = len([cid for cid in clusters.keys() if cid != -1])  # 노이즈(-1) 제외

    num_rows, num_cols = math.ceil(num_clusters / 3), 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3))
    axes = axes.flatten()

    cluster_ids = sorted(clusters.keys(), key=lambda x: int(x))
    if skip_noise_cluster:
        cluster_ids = [cid for cid in cluster_ids if cid != -1]

    warnings.filterwarnings("ignore", category=UserWarning, module="wordcloud")

    for ax, cluster_id in zip(axes, cluster_ids):
        cluster_tokens = []
        for feat_idx in clusters[cluster_id]:
            tokens = feature_to_tokens.get(feat_idx, [])
            cluster_tokens.extend(tokens)

        if not cluster_tokens:
            ax.set_title(f"Cluster {cluster_id} (No tokens)", fontsize=10)
            ax.axis("off")
            continue

        text_to_plot = " ".join(cluster_tokens)

        try:
            title = f"Cluster {cluster_id}"
            if fqi_map:
                fqi = fqi_map.get(cluster_id, None)
                if fqi is not None:
                    title = title + f" (FQI: {fqi:.2f})"

            wordcloud = WordCloud(
                width=width,
                height=height,
                background_color="white",
                colormap="viridis",
                random_state=42,
                font_path=str(settings.font_path),
            ).generate(text_to_plot)
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontsize=10)

        except Exception as e:
            ax.set_title(f"Cluster {cluster_id} (Error)", fontsize=10)
            ax.axis("off")
            print(f"Error generating word cloud for Cluster {cluster_id}: {e}")

    for ax in axes[len(cluster_ids) :]:
        ax.axis("off")

    plt.tight_layout(pad=1.0)
    plt.show()
    plt.close()
