import os
import warnings
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import torch
from src.features import SampleRecord
from src.global_settings import settings
from src.sae import SparseAutoencoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "CUDA not available, check device or disable this assertion."


class SaeFeatureHandler:
    def __init__(
        self,
        model: SparseAutoencoder,
        checkpoint: str,
        layer: int,
        dataloader: DataLoader,
        text_df: pd.DataFrame | None = None,
        tokenizer: AutoTokenizer | None = None,
        device: torch.device | None = None,
        feature_indices: List[int] | None = None,
    ):
        """
        특정 checkpoint/layer에 포함된 모든 feature들의 샘플별 점수 및 (Top-N positive / Top-N negative) 샘플의
        SampleRecord를 계산, 저장 및 로드하는 기능을 제공하는 handler class.

        Args:
            model: `SparseAutoencoder` model object(suggested to use `load_sae` function from `utils.model_loader`).
            tokenizer: Original tokenizer.
            checkpoint: Specific SAE checkpoint (e.g., "f3072_k256_alpha").
            layer: Index of specific layer (e.g., 1, 6, 21, ...)
            top_n: Number of top positive/negative samples to keep for each feature. (default: 50) (total 100 samples per feature)
            feature_indices: List of feature index to process/analyze.

        Note:
            total_score_post: sum(post-activation of token_score) 방식으로 계산됨.
            avg_score_post: avg(post-activation of token_score) 방식으로 계산됨.
            total_score_pre: sum(pre-activation of token_score) 방식으로 계산됨.
            avg_score_pre: avg(pre-activation of token_score) 방식으로 계산됨.
            Top-N Positive samples: total_score_post 기준으로 선정됨.
            Top-N Negative samples: avg_score_pre 기준으로 선정됨. (Top-K 활성화 함수가 K-norm 이상의 값은 모두 0으로 만들기 때문)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            warnings.warn("Device is not CUDA. This may lead to performance issues.", UserWarning)

        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.lm_weights_path) if tokenizer is None else tokenizer
        self.checkpoint = checkpoint
        self.layer = layer
        self.dataloader = dataloader
        self.text_df = text_df if text_df is not None else pd.read_csv(settings.text_dataset_path)
        self.feature_indices = feature_indices if feature_indices else list(range(self.model.feature_dim))

        # Global sample-feature score map
        self.total_scores_pre: np.ndarray | None = None
        self.total_score_post: np.ndarray | None = None
        self.avg_scores_pre: np.ndarray | None = None
        self.avg_scores_post: np.ndarray | None = None

        # Top-N positive/negative samples for all features
        self.topn_pos_samples: Dict[int, List[SampleRecord]] = {feat_idx: [] for feat_idx in self.feature_indices}
        self.topn_neg_samples: Dict[int, List[SampleRecord]] = {feat_idx: [] for feat_idx in self.feature_indices}

    def process_all_features(self, top_n: int = 50):
        """
        Calculate all features' scores for all samples in the dataloader, and
        create `SampleRecord`s for those features.

        Save as:
            Global Scores: np.ndarray of (scores, sample_indices, feature_indices) -> data/feature_meta/{checkpoint}/{layer}/{score_type}.npy
            Top-N Records: Dict[int, SampleRecord] -> data/feature_records/{checkpoint}/{layer}/{feature_idx}.parquet
        """
        self.top_n = top_n

        num_samples = len(self.dataloader.dataset)
        num_features = self.model.feature_dim

        self.total_scores_pre = np.zeros((num_samples, num_features), dtype=np.float32)
        self.total_scores_post = np.zeros((num_samples, num_features), dtype=np.float32)
        self.avg_scores_pre = np.zeros((num_samples, num_features), dtype=np.float32)
        self.avg_scores_post = np.zeros((num_samples, num_features), dtype=np.float32)

        top_pos_scores = torch.full((num_features, top_n), -torch.inf, device=self.device)
        top_pos_indices = torch.full((num_features, top_n), -1, dtype=torch.long, device=self.device)
        top_neg_scores = torch.full((num_features, top_n), torch.inf, device=self.device)
        top_neg_indices = torch.full((num_features, top_n), -1, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for inputs, attention_mask, text_indices in tqdm(
                self.dataloader, desc=f"[{self.checkpoint}/Layer #{self.layer}] Pass 1: Processing samples/features..."
            ):
                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)
                text_indices = text_indices.to(self.device)

                # [CLS] ('<s>') and [SEP] ('<\\s>') 토큰 제외
                attn_mask = attention_mask.clone()
                attn_mask[:, 0] = 0
                attn_mask[torch.arange(attn_mask.size(0)), attention_mask.sum(dim=1) - 1] = 0

                processed_inputs = self.model._preprocess_input(inputs)
                pre_activations = self.model.encoder(processed_inputs)
                post_activations = self.model._apply_activation(pre_activations)

                mask = attn_mask.unsqueeze(-1)
                pre_maksed = pre_activations * mask
                post_masked = post_activations * mask
                num_tokens = attn_mask.sum(dim=1, keepdim=True)
                num_tokens[num_tokens <= 0] = 1

                total_score_pre_batch = pre_maksed.sum(dim=1)
                total_score_post_batch = post_masked.sum(dim=1)
                avg_score_pre_batch = total_score_pre_batch / num_tokens
                avg_score_post_batch = total_score_post_batch / num_tokens

                text_indices_cpu = text_indices.cpu().numpy()

                self.total_scores_pre[text_indices_cpu, :] = total_score_pre_batch.cpu().numpy()
                self.total_scores_post[text_indices_cpu, :] = total_score_post_batch.cpu().numpy()
                self.avg_scores_pre[text_indices_cpu, :] = avg_score_pre_batch.cpu().numpy()
                self.avg_scores_post[text_indices_cpu, :] = avg_score_post_batch.cpu().numpy()

                batch_indices_expanded = text_indices.unsqueeze(0).T.expand(-1, num_features).T

                # Positive samples (by total_score_post)
                combined_scores_pos = torch.cat([top_pos_scores, total_score_post_batch.T], dim=1)
                combined_indices_pos = torch.cat([top_pos_indices, batch_indices_expanded], dim=1)
                new_top_scores_pos, top_k_indices_pos = torch.topk(combined_scores_pos, k=top_n, dim=1, largest=True)
                top_pos_scores = new_top_scores_pos
                top_pos_indices = torch.gather(combined_indices_pos, 1, top_k_indices_pos)

                # Negative samples (by avg_score_pre)
                combined_scores_neg = torch.cat([top_neg_scores, avg_score_pre_batch.T], dim=1)
                combined_indices_neg = torch.cat([top_neg_indices, batch_indices_expanded], dim=1)
                new_top_scores_neg, top_k_indices_neg = torch.topk(combined_scores_neg, k=top_n, dim=1, largest=False)
                top_neg_scores = new_top_scores_neg
                top_neg_indices = torch.gather(combined_indices_neg, 1, top_k_indices_neg)

        # data/feature_meta/{checkpoint}/{layer}/{score_type}.npy
        meta_dir = Path(settings.feature_meta_dir) / self.checkpoint / str(self.layer)
        os.makedirs(meta_dir, exist_ok=True)
        np.save(os.path.join(meta_dir, "total_pre.npy"), self.total_scores_pre)
        np.save(os.path.join(meta_dir, "total_post.npy"), self.total_scores_post)
        np.save(os.path.join(meta_dir, "avg_pre.npy"), self.avg_scores_pre)
        np.save(os.path.join(meta_dir, "avg_post.npy"), self.avg_scores_post)
        print(f"✅ Global Score metrics saved to: {meta_dir}")

        return top_pos_indices.cpu(), top_neg_indices.cpu()

    def save_topn_records(self, top_pos_indices: torch.Tensor, top_neg_indices: torch.Tensor):
        """
        Positive samples의 Top-N 선택 기준: total_score_post
        Negative samples의 Top-N 선택 기준: avg_score_pre
        """
        print(
            f"[{self.checkpoint}/Layer #{self.layer}] Pass 2: Generating Top-{self.top_n} records for each features (total={len(self.feature_indices)})..."
        )

        all_top_indices_set = set(top_pos_indices.view(-1).tolist())
        all_top_indices_set.update(top_neg_indices.view(-1).tolist())
        if -1 in all_top_indices_set:
            all_top_indices_set.remove(-1)

        pos_samples_map = {f_idx: set(top_pos_indices[f_idx].tolist()) for f_idx in self.feature_indices}
        neg_samples_map = {f_idx: set(top_neg_indices[f_idx].tolist()) for f_idx in self.feature_indices}

        top_n_pos_records = {f_idx: [] for f_idx in self.feature_indices}
        top_n_neg_records = {f_idx: [] for f_idx in self.feature_indices}

        self.model.eval()
        with torch.no_grad():
            for inputs, attention_mask, text_indices in tqdm(
                self.dataloader, desc=f"[{self.checkpoint}/L{self.layer}] Generating records"
            ):
                batch_indices_set = set(text_indices.tolist())
                relevant_batch_indices = batch_indices_set.intersection(all_top_indices_set)

                if not relevant_batch_indices:
                    continue

                inputs = inputs.to(self.device)
                attention_mask = attention_mask.to(self.device)

                # [CLS] (or '<s>') and [SEP] (or '<\\s>') 토큰 제외
                attn_mask = attention_mask.clone()
                attn_mask[:, 0] = 0
                attn_mask[torch.arange(attn_mask.size(0)), attention_mask.sum(dim=1) - 1] = 0

                processed_inputs = self.model._preprocess_input(inputs)
                pre_activations = self.model.encoder(processed_inputs)
                post_activations = self.model._apply_activation(pre_activations)

                pre_acts_cpu = (pre_activations * attn_mask.unsqueeze(-1)).cpu()
                post_acts_cpu = (post_activations * attn_mask.unsqueeze(-1)).cpu()
                attn_mask_cpu = attn_mask.cpu()

                for i, text_idx in enumerate(text_indices.tolist()):
                    if text_idx not in relevant_batch_indices:
                        continue

                    row = self.text_df.iloc[text_idx]
                    text = row["text"]
                    tokenized_inputs = self.tokenizer(
                        text, max_length=settings.prev_seq_len, truncation=True, padding="max_length"
                    )
                    tokens = self.tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"])
                    sample_mask = attn_mask_cpu[i]

                    sample_pre_acts = pre_acts_cpu[i]
                    sample_post_acts = post_acts_cpu[i]

                    for feat_idx in self.feature_indices:
                        if text_idx in pos_samples_map[feat_idx]:
                            record_data_pos = self._create_sample_record(
                                text_idx, row, tokens, sample_mask, sample_post_acts[:, feat_idx], "positive", feat_idx
                            )
                            top_n_pos_records[feat_idx].append(record_data_pos)

                        if text_idx in neg_samples_map[feat_idx]:
                            record_data_neg = self._create_sample_record(
                                text_idx, row, tokens, sample_mask, sample_pre_acts[:, feat_idx], "negative", feat_idx
                            )
                            top_n_neg_records[feat_idx].append(record_data_neg)

        record_dir = Path(settings.feature_record_dir) / self.checkpoint / str(self.layer)
        os.makedirs(record_dir, exist_ok=True)

        for feat_idx in tqdm(self.feature_indices, desc="Saving feature records"):
            pos_samples_data = sorted(top_n_pos_records[feat_idx], key=lambda x: x["total_score_post"], reverse=True)
            neg_samples_data = sorted(top_n_neg_records[feat_idx], key=lambda x: x["avg_score_pre"])

            pos_df = pd.DataFrame(pos_samples_data)
            neg_df = pd.DataFrame(neg_samples_data)

            if pos_df.empty and neg_df.empty:
                continue

            feature_record_path = record_dir / f"{feat_idx}.parquet"  # positive samples 저장 경로

            if not pos_df.empty:
                pos_record_path = feature_record_path
                pos_df.to_parquet(pos_record_path, engine="pyarrow", index=False)

            if not neg_df.empty:
                neg_record_path = record_dir / f"{feat_idx}_neg.parquet"
                # negative samples는 '*_neg.npy' 이름으로 따로 저장
                neg_df.to_parquet(neg_record_path, engine="pyarrow", index=False)

        print(f"✅ Top-{self.top_n} feature records saved to: {record_dir}")

    def _create_sample_record(
        self,
        text_idx: int,
        row: pd.Series,
        tokens: List[str],
        sample_mask: torch.Tensor,
        sample_neurons: torch.Tensor,
        sample_type: str,
        feat_idx: int,
    ) -> dict:
        """
        Positive samples의 'active_tokens': post-activation 값이 0 이상인 토큰들만 선택.
        Negative samples의 'active_tokens': pre-activation 값이 0 이하인 토큰들만 선택.
        """

        pos_threshold_ge = 0
        neg_threshold_le = 0

        if sample_type == "positive":
            token_indices = torch.where(sample_neurons > pos_threshold_ge)[0]
            score_key, score_val = "total_score_post", self.total_scores_post[text_idx, feat_idx]
            other_score_key, other_score_val = "avg_score_pre", self.avg_scores_pre[text_idx, feat_idx]
        else:  # negative
            token_indices = torch.where(sample_neurons < neg_threshold_le)[0]
            score_key, score_val = "avg_score_pre", self.avg_scores_pre[text_idx, feat_idx]
            other_score_key, other_score_val = "total_score_post", self.total_scores_post[text_idx, feat_idx]

        token_scores = sample_neurons[token_indices]

        special_tokens = {
            r"<s>",  # CLS, 0
            r"<\\s>",  # SEP, 1
            r"<pad>",  # PAD, 49999
        }

        active_tokens = [
            {
                "token_idx": idx.item(),
                "token_str": tokens[idx.item()],
                "token_score": round(score.item(), 4),
            }
            for idx, score in zip(token_indices, token_scores)
            if sample_mask[idx.item()] == 1 and tokens[idx.item()] not in special_tokens
        ]

        record_data = {
            "text_idx": text_idx,
            "label": row["label"],
            "pred": row.get("pred", -1),
            "model": row["model"],
            "prompt": row["prompt"],
            "text": row["text"],
            score_key: score_val,
            other_score_key: other_score_val,
            "active_tokens": active_tokens,
        }
        return record_data
