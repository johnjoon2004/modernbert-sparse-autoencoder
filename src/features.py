import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.auto_explainer.explanations import FeatureExplanations, ScoredExplanation
from src.global_settings import settings

RECORD_DIR = settings.feature_record_dir
META_DIR = settings.feature_meta_dir


@dataclass
class TokenRecord:
    token_idx: int
    token_str: str
    token_score: float


@dataclass
class SampleRecord:
    text_idx: int
    label: int
    pred: int
    model: str
    prompt: str
    text: str

    total_score_post: float
    avg_score_pre: float

    active_tokens: List[TokenRecord]


@dataclass
class FeatureRecord:
    layer_index: int
    feature_index: int

    scores: np.ndarray  # all samples scores. shape: (num_samples, )

    positive_samples: List[SampleRecord] = field(default_factory=list)
    negative_samples: List[SampleRecord] = field(default_factory=list)

    explanations: FeatureExplanations | None = None


class FeatureRecordWrapper:
    """
    Provides frontend-friendly interface for FeatureRecord.
    Adds statistics and filtering methods on top of FeatureRecord.
    """

    def __init__(self, feature_record: FeatureRecord, feature_explanations: Optional[FeatureExplanations] = None):
        text_df = pd.read_csv(settings.text_dataset_path)
        text_df = text_df.reset_index(drop=True)
        text_df.index.name = "text_idx"
        text_lookup = dict(zip(text_df.index, text_df["text"]))

        self.layer_index = feature_record.layer_index
        self.feature_index = feature_record.feature_index

        self.text_df = text_df
        self._text_lookup = text_lookup or {}

        self.scores = feature_record.scores

        self.positive_samples = [self._convert_sample_to_dict(sample) for sample in feature_record.positive_samples]
        self.negative_samples = [self._convert_sample_to_dict(sample) for sample in feature_record.negative_samples]

        self.stats = self._calculate_stats()

        # self.explanations = feature_explanations or []
        # self.best_explanation = feature_explanations.get_best_explanation() if feature_explanations else None
        # self.description = (
        #     feature_explanations[1].explanation
        #     if feature_explanations and len(feature_explanations.explanations) > 1
        #     else "N/A"
        # )

    def _convert_sample_to_dict(self, sample: SampleRecord) -> Dict:
        # token = sample.active_tokens[0]
        # assert isinstance(token, Dict), "Expected active_tokens in sample to be list of dicts."

        # TODO: make a better serialization

        return {
            "text_idx": sample.text_idx,
            "label": sample.label,
            "pred": sample.pred,
            "model": sample.model,
            "prompt": sample.prompt,
            "text": self._text_lookup.get(sample.text_idx, ""),
            "total_score_post": sample.total_score_post,
            "avg_score_pre": sample.avg_score_pre,
            "active_tokens": [
                {
                    "token_idx": t.get("token_idx"),
                    "token_str": t.get("token_str"),
                    "token_score": t.get("token_score"),
                }
                for t in sample.active_tokens
            ],
        }

    def _calculate_stats(self) -> Dict:
        """Calculate activation statistics and distributions by metadata field."""

        # Filtering score > 0
        active_mask = self.scores > 0
        active_scores = self.scores[active_mask]
        active_text_df = self.text_df[active_mask]  # NOTE: Do not reset index

        all_df = pd.DataFrame(
            {"score": self.scores, "label": self.text_df["label"], "pred": self.text_df["pred"]},
            index=self.text_df.index,
        )

        if len(active_scores) == 0:
            min_score, max_score = float(self.scores.min()), float(self.scores.max())
            bins = np.histogram_bin_edges(self.scores, bins="auto")
            counts, _ = np.histogram(self.scores, bins=bins)
            avg_score = float(self.scores.mean())
            variance = float(np.var(self.scores))
        else:
            min_score, max_score = float(active_scores.min()), float(active_scores.max())
            bins = np.histogram_bin_edges(active_scores, bins="auto")
            counts, _ = np.histogram(active_scores, bins=bins)
            avg_score = float(active_scores.mean())
            variance = float(np.var(active_scores))

        # Active Rate: 얼마나 많은 샘플에서 해당 특징이 발현 되었는지
        active_rate = {
            "all": float(np.mean(self.scores > 0)),
            "human": float(np.mean(self.scores[self.text_df["label"] == 0] > 0)),
            "ai": float(np.mean(self.scores[self.text_df["label"] == 1] > 0)),
        }

        # (Confusion Matrix-like statistics)
        # Active Matrix: 상황별((label=0, pred=0), (label=0, pred=1), ...) 해당 특징이 발현(score > 0)되는 비율
        # Score Matrix: 상황별((label=0, pred=0), (label=0, pred=1), ...) 해당 특징이 갖는 점수의 평균 값 (*전체 샘플 기준)
        active_matrix = (
            all_df.groupby(["label", "pred"])["score"].apply(lambda s: (s > 0).mean()).unstack(fill_value=0)
        )
        score_matrix = all_df.groupby(["label", "pred"])["score"].mean().unstack(fill_value=0)

        # AUROC: 특징의 score가 label=1과 label=0을 얼마나 잘 구분하는지
        auroc = roc_auc_score(self.text_df["label"], self.scores)

        # Pearson Correlation: label(or pred)과 score 간의 상관관계
        pearson_corr = {
            "label": float(self.text_df["label"].corr(pd.Series(self.scores))),
            "pred": float(self.text_df["pred"].corr(pd.Series(self.scores))),
        }

        # Cohen's D: label=0 / label=1 (or pred) 두 그룹의 score가 얼마나 차이나는지
        def _cohens_d(x, y):
            if len(x) == 0 or len(y) == 0 or (x.std() == 0 and y.std() == 0):
                return None
            return float((x.mean() - y.mean()) / np.sqrt((x.std() ** 2 + y.std() ** 2) / 2))

        cohens_d = {
            "label": _cohens_d(self.scores[self.text_df["label"] == 1], self.scores[self.text_df["label"] == 0]),
            "pred": _cohens_d(self.scores[self.text_df["pred"] == 1], self.scores[self.text_df["pred"] == 0]),
        }

        # (정규성 Moments)
        # Skewness: 왜도 (분포의 비대칭성, >0: 오른쪽 꼬리, <0: 왼쪽 꼬리)
        # Kurtosis: 첨도 (분포의 뾰족함, >0: 뾰족, <0: 평평)
        # skewness = float(pd.Series(self.scores).skew())
        # kurtosis = float(pd.Series(self.scores).kurtosis())

        base_stats = {
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "variance": variance,
            "active_rate": active_rate,
            "active_matrix": active_matrix,
            "score_matrix": score_matrix,
            "auroc": auroc,
            "pearson_corr": pearson_corr,
            "cohens_d": cohens_d,
            # "skewness": skewness,
            # "kurtosis": kurtosis,
        }

        active_df = pd.DataFrame(
            {"score": active_scores, "label": active_text_df["label"], "pred": active_text_df["pred"]},
            index=active_text_df.index,
        )

        dist_stats = {}
        dist_stats["score_dist_data"] = {"bins": bins, "counts": counts}
        dist_stats["score_dist_data_attrs"] = {}

        dist_attrs = ["label", "pred", "model", "prompt"]
        for attr in dist_attrs:
            dist_stats[f"{attr}_dist"] = active_text_df[attr].value_counts().to_dict()

            active_df["group"] = active_text_df[attr]

            if attr in ["label", "pred"]:
                group_values = active_text_df[attr].unique()
            else:  # limit to count top-10 values only
                group_values = active_text_df[attr].value_counts().nlargest(10).index

            groups_data = {}

            for group_val in group_values:
                group_scores = active_df[active_df["group"] == group_val]["score"]
                if not group_scores.empty:
                    counts, _ = np.histogram(group_scores, bins=bins)
                    groups_data[str(group_val)] = counts

            dist_stats["score_dist_data_attrs"][attr] = {"bins": bins, "groups": groups_data}

        stats = {**base_stats, **dist_stats}
        return stats

    def filter_samples(
        self,
        samples: List[Dict],
        label: Optional[int] = None,
        pred: Optional[int] = None,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        _score_metric: Literal["total_score_post", "avg_score_pre"] = "total_score_post",
    ) -> List[Dict]:
        if isinstance(samples[0], SampleRecord):
            filtered = [asdict(s) for s in samples]
        else:
            filtered = samples

        if label is not None:
            filtered = [s for s in filtered if s["label"] == label]
        if pred is not None:
            filtered = [s for s in filtered if s["pred"] == pred]
        if model is not None:
            filtered = [s for s in filtered if s["model"] == model]
        if prompt is not None:
            filtered = [s for s in filtered if s["prompt"] == prompt]
        if min_score is not None:
            filtered = [s for s in filtered if s[_score_metric] >= min_score]
        if max_score is not None:
            filtered = [s for s in filtered if s[_score_metric] <= max_score]

        return filtered

    def sort_samples(
        self, samples: List[Dict], sort_by: str = "total_score_post", ascending: bool = False
    ) -> List[Dict]:
        return sorted(samples, key=lambda x: x[sort_by], reverse=not ascending)

    def get_top_pos_samples(self, n: int = 10, sort_by: str = "total_score_post") -> List[Dict]:
        sorted_samples = self.sort_samples(self.positive_samples, sort_by=sort_by, ascending=False)
        return sorted_samples[:n]

    def get_top_neg_samples(self, n: int = 10, sort_by: str = "avg_score_pre") -> List[Dict]:
        sorted_samples = self.sort_samples(self.negative_samples, sort_by=sort_by, ascending=True)
        return sorted_samples[:n]


# ======================================================================================================
# ▼▼▼ Below is data loading utilities ▼▼▼
# ======================================================================================================


# def is_feature_alive(dataset_path: str, layer_index: int, feature_no: int) -> bool:
#     """Return whether the feature's sample record file exists."""
#     path = os.path.join(dataset_path, str(layer_index), f"{feature_no}.parquet")
#     return os.path.exists(path)


def load_score_data(
    checkpoint: str,
    layer: int,
    metric_name: Literal["total_pre", "total_post", "avg_pre", "avg_post"],
    meta_dir: str | Path = META_DIR,
) -> np.ndarray:
    score_path = Path(meta_dir) / checkpoint / str(layer) / f"{metric_name}.npy"
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"Score file not found: {score_path}")

    scores = np.load(score_path)
    return scores


def load_sample_records(
    checkpoint: str,
    layer: int,
    feature_index: int,
    sample_type: Literal["positive", "negative"],
    as_frame: bool = False,
    record_dir: str | Path = RECORD_DIR,
) -> pd.DataFrame | List[SampleRecord]:

    sample_dir = Path(record_dir) / checkpoint / str(layer)

    if sample_type == "negative":
        sample_path = sample_dir / f"{feature_index}_neg.parquet"
    else:  # positive
        sample_path = sample_dir / f"{feature_index}.parquet"

    records = pd.read_parquet(sample_path)

    # drop positive samples with score <= 0
    if sample_type == "positive":
        records = records[records["total_score_post"] > 0]

    if as_frame:
        return records

    records = [SampleRecord(**row) for _, row in records.iterrows()]
    return records


def get_available_features(checkpoint: str, layer: int, strict: bool = False) -> List[int]:
    record_dir = Path(RECORD_DIR) / checkpoint / str(layer)

    if not record_dir.exists():
        return []

    scores_all = load_score_data(checkpoint, layer, metric_name="total_post")

    try:
        feature_files = list(record_dir.glob("*.parquet"))
        feature_files = [
            f for f in feature_files if re.fullmatch(r"\d+\.parquet", f.name)
        ]  # count positive samples only
        feature_indices = []
        for file in feature_files:
            try:
                feat_idx = int(file.stem)

                # score 기준으로 검사
                scores_feat = scores_all[:, feat_idx]
                if scores_feat.max() <= 0:
                    continue

                if strict:
                    # Double-check: Positive 샘플이 있는데 점수가 0인 케이스는 not available로 간주
                    positive_samples = load_sample_records(
                        checkpoint=checkpoint,
                        layer=layer,
                        feature_index=feat_idx,
                        sample_type="positive",
                        as_frame=False,
                    )
                    if not positive_samples:
                        continue

                feature_indices.append(feat_idx)

            except (ValueError, IndexError):
                continue

        return sorted(feature_indices)

    except Exception as e:
        return []
