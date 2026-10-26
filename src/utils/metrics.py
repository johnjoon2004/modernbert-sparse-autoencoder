import warnings
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity


def calculate_fqi(
    feature_indices: List[int],
    significance_map: pd.Series,
    frequency_map: pd.Series,
) -> float:
    """
    주어진 특징 그룹(feature_indices)의 FQI(Feature Quality Index)를 계산합니다.
    FQI = (Sum of (Significance * Frequency)) / (Sum of Frequency)

    Args:
        feature_indices (List[int]): FQI를 계산할 대상 특징들의 인덱스 리스트.
        significance_map (pd.Series): {feature_index: significance} 맵.
        frequency_map (pd.Series): {feature_index: frequency} 맵.

    Returns:
        float: 계산된 FQI 점수.
    """

    if not feature_indices:
        return np.nan

    common_indices = list(set(feature_indices).intersection(significance_map.index).intersection(frequency_map.index))
    if not common_indices:
        return np.nan

    group_sig = significance_map.loc[common_indices]
    group_freq = frequency_map.loc[common_indices]

    denominator = group_freq.sum()
    if denominator == 0:
        return 0.0

    numerator = (group_sig * group_freq).sum()
    fqi = numerator / denominator

    return float(fqi)


def calculate_significances(
    scores_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    name: str = "significance",
    as_frame: bool = False,
) -> pd.Series | pd.DataFrame:

    meta_with_index = meta_df.loc[scores_df.index]

    is_correct = meta_with_index["label"] == meta_with_index["pred"]
    sign = np.where(is_correct, 1, -1)

    # abs_logit = meta_with_index["logit"].abs()
    abs_logit = meta_with_index["prob"].abs()

    vec_B = pd.Series(sign * abs_logit, index=meta_with_index.index, name="signed_confidence")

    vec_A_df = scores_df.rank(pct=True)

    merged_df = vec_A_df.join(vec_B, how="inner")

    if merged_df.empty:
        return pd.Series(index=scores_df.columns, dtype=float)

    correlations = merged_df.corr(method="pearson")["signed_confidence"]

    significance_scores = correlations.drop("signed_confidence")
    significance_scores.name = name

    if as_frame:
        significance_scores = significance_scores.to_frame()

    return significance_scores


def calculate_active_frequency(
    df: pd.DataFrame,
    name: str = "freq",
    ratio: bool = True,
) -> pd.Series:
    boolean_df = df > 1e-5
    if ratio:
        frequency_series = boolean_df.mean(axis=0)
    else:
        frequency_series = boolean_df.sum(axis=0)

    frequency_series.name = name
    return frequency_series


def calculate_generality(scores_df: pd.DataFrame, feature_index: int) -> float:
    """
    Calculate Generality of a feature.

    Generality measures how frequently a feature is activated across all samples.
    Formula: Generality(f) = (1/N) * Σ 1[a_{i,f} > 0]

    Args:
        scores_df: DataFrame containing feature scores with shape (num_samples, num_features)
        meta_df: DataFrame containing metadata (not used but kept for consistency)
        feature_index: Index of the feature to calculate generality for

    Returns:
        float: Generality score between 0 and 1
    """
    if feature_index not in scores_df.columns:
        raise ValueError(f"Feature index {feature_index} not found in scores DataFrame.")

    feature_scores = scores_df[feature_index]
    num_samples = len(feature_scores)
    num_activated = (feature_scores > 0).sum()

    generality = num_activated / num_samples if num_samples > 0 else 0.0
    return float(generality)


def calculate_discrimination(
    scores_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    feature_index: int,
    w_act: float = 0.5,
    w_int: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Calculate Discrimination of a feature.

    Discrimination measures how well a feature can distinguish between labels (Ground Truth).
    It combines activation-based and intensity-based discrimination using AUROC.

    Formula:
    - D_act(f) = AUROC({1[s_{i,f} > 0]}, {y_i})
    - D_int(f) = AUROC({s_{i,f} | s_{i,f} > 0}, {y_i | s_{i,f} > 0})
    - Discrimination(f) = w_act * D_act(f) + w_int * D_int(f)

    Args:
        scores_df: DataFrame containing feature scores with shape (num_samples, num_features)
        meta_df: DataFrame containing metadata with 'label' column
        feature_index: Index of the feature to calculate discrimination for
        w_act: Weight for activation-based discrimination (default: 0.5)
        w_int: Weight for intensity-based discrimination (default: 0.5)

    Returns:
        Tuple[float, float, float]: (total_discrimination, d_act, d_int)
    """
    if feature_index not in scores_df.columns:
        raise ValueError(f"Feature index {feature_index} not found in scores DataFrame.")
    if "label" not in meta_df.columns:
        raise ValueError("'label' column not found in metadata DataFrame.")

    if len(scores_df) != len(meta_df):
        warnings.warn(
            f"Scores DataFrame length ({len(scores_df)}) does not match metadata DataFrame length ({len(meta_df)}). Using common indices."
        )
        common_index = scores_df.index.intersection(meta_df.index)
        feature_scores = scores_df.loc[common_index, feature_index]
        labels = meta_df.loc[common_index, "label"]
    else:
        feature_scores = scores_df[feature_index]
        labels = meta_df["label"]

    # Activation-based discrimination: AUROC based on whether feature is activated
    activation_indicator = (feature_scores > 0).astype(int)

    try:
        # Check if we have both classes
        if len(np.unique(labels)) < 2 or len(np.unique(activation_indicator)) < 2:
            d_act = 0.5  # Default to random classifier
        else:
            d_act = roc_auc_score(labels, activation_indicator)
    except ValueError:
        d_act = 0.5

    # Intensity-based discrimination: AUROC based on score intensity for activated samples
    activated_mask = feature_scores > 0
    if activated_mask.sum() > 0:
        activated_scores = feature_scores[activated_mask]
        activated_labels = labels[activated_mask]

        try:
            if len(np.unique(activated_labels)) < 2:
                d_int = 0.5
            else:
                d_int = roc_auc_score(activated_labels, activated_scores)
        except ValueError:
            d_int = 0.5
    else:
        d_int = 0.5

    # Combined discrimination
    discrimination = w_act * d_act + w_int * d_int

    return float(discrimination), float(d_act), float(d_int)


def calculate_determination(
    scores_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    feature_index: int,
    w_act: float = 0.5,
    w_int: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Calculate Determination of a feature.

    Determination measures how consistently a feature correlates with model confidence (logit).
    It combines activation-based and intensity-based determination using Pearson correlation.

    Formula:
    - C_act(f) = corr({1[s_{i,f} > 0]}, {logit_i})
    - C_int(f) = corr({CDF(s_{i,f}) | s_{i,f} > 0}, {logit_i | s_{i,f} > 0})
    - Determination(f) = w_act * C_act(f) + w_int * C_int(f)

    Args:
        scores_df: DataFrame containing feature scores with shape (num_samples, num_features)
        meta_df: DataFrame containing metadata with 'text' column
        feature_index: Index of the feature to calculate determination for
        model: Classification model to extract logits (required if 'logit' not in meta_df)
        tokenizer: Tokenizer for the model (required if 'logit' not in meta_df)
        w_act: Weight for activation-based determination (default: 0.5)
        w_int: Weight for intensity-based determination (default: 0.5)
        device: Device to run model on (default: cuda if available)

    Returns:
        Tuple[float, float, float]: (total_determination, c_act, c_int)
    """
    if feature_index not in scores_df.columns:
        raise ValueError(f"Feature index {feature_index} not found in scores DataFrame.")

    if len(scores_df) != len(meta_df):
        warnings.warn(
            f"Scores DataFrame length ({len(scores_df)}) does not match metadata DataFrame length ({len(meta_df)}). Using common indices."
        )
        common_index = scores_df.index.intersection(meta_df.index)
        feature_scores = scores_df.loc[common_index, feature_index]
        meta_df = meta_df.loc[common_index]
    else:
        feature_scores = scores_df[feature_index]

    # Extract or compute logits
    if "logit" in meta_df.columns:
        logits = meta_df["logit"]
    else:
        raise ValueError("'logit' column not found in metadata DataFrame. Please provide logits.")

    # Activation-based determination: Pearson correlation between activation indicator and logit
    activation_indicator = (feature_scores > 0).astype(int)

    try:
        if activation_indicator.std() == 0 or logits.std() == 0:
            c_act = 0.0
        else:
            c_act = np.corrcoef(activation_indicator, logits)[0, 1]
            if np.isnan(c_act):
                c_act = 0.0
    except Exception:
        c_act = 0.0

    # Intensity-based determination: Pearson correlation between CDF-transformed scores and logit
    activated_mask = feature_scores > 0
    if activated_mask.sum() > 1:  # Need at least 2 points for correlation
        activated_scores = feature_scores[activated_mask]
        activated_logits = logits[activated_mask]

        # CDF transformation
        try:
            cdf_scores = np.array(
                [calculate_cdf(scores_df, feature_index, score, emperical=True) for score in activated_scores]
            )

            if cdf_scores.std() == 0 or activated_logits.std() == 0:
                c_int = 0.0
            else:
                c_int = np.corrcoef(cdf_scores, activated_logits)[0, 1]
                if np.isnan(c_int):
                    c_int = 0.0
        except Exception:
            c_int = 0.0
    else:
        c_int = 0.0

    # Combined determination
    determination = w_act * c_act + w_int * c_int

    return float(determination), float(c_act), float(c_int)


def calculate_pcc(
    scores_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    feature_index: int,
    key: Literal["label", "pred"],
    flop_negative: bool = False,
) -> float:
    """
    Calculate Pearson Correlation Coefficient (PCC) between feature scores and labels/predictions.

    Args:
        scores_df (pd.DataFrame): DataFrame containing feature scores with shape (num_samples, num_features).
        meta_df (pd.DataFrame): DataFrame containing metadata with 'label' and 'pred' columns.
        feature_index (int): Index of the feature to calculate PCC for.
        key (Literal["label", "pred"]): Key to use from meta_df for correlation ('label' or 'pred').

    Returns:
        float: Pearson Correlation Coefficient between the specified feature scores and labels/predictions.
    """

    if feature_index not in scores_df.columns:
        raise ValueError(f"Feature index {feature_index} not found in scores DataFrame.")
    if key not in meta_df.columns:
        raise ValueError(f"Key '{key}' not found in metadata DataFrame.")

    if len(scores_df) != len(meta_df):
        warnings.warn(
            f"Scores DataFrame length ({len(scores_df)}) does not match metadata DataFrame length ({len(meta_df)}). Truncating to minimum length."
        )

        common_index = scores_df.index.intersection(meta_df.index)
        if len(common_index) == 0:
            raise ValueError("No common indices between scores and metadata DataFrames.")

        feature_scores = scores_df.loc[common_index, feature_index]
        labels = meta_df.loc[common_index, key]

    else:
        feature_scores = scores_df[feature_index]
        labels = meta_df[key]

    corr = feature_scores.corr(labels, method="pearson")

    if flop_negative and corr < 0:
        corr = -corr

    return corr


def calculate_cdf(
    scores_df: pd.DataFrame,
    feature_index: int,
    feature_score: float,
    emperical: bool = True,
    gaussian: bool = True,
) -> float:
    values = scores_df[feature_index].dropna().values
    values = np.sort(values)

    if emperical:
        cdf_val = np.searchsorted(values, feature_score, side="right") / len(values)
    else:
        if gaussian:
            mu, sigma = np.mean(values), np.std(values)
            cdf_val = norm.cdf(feature_score, loc=mu, scale=sigma)
        else:
            # KDE kernel density estimation
            kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(values[:, None])

            def pdf_func(x):
                return np.exp(kde.score_samples([[x]]))[0]

            cdf_val = quad(pdf_func, values.min(), feature_score)[0]
            cdf_val /= quad(pdf_func, values.min(), values.max())[0]

    return float(np.clip(cdf_val, 0, 1))


def calculate_bayesian_prob(
    feature_score: float,
    scores_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    feature_index: int,
    label: Literal["label", "pred"] = "pred",
    prior: float | None = None,
) -> float:
    """
    Compute posterior probability P(label=1 | feature_score) using Bayesian inference.

    Args:
        feature_score: The feature score to compute probability for
        scores_df: DataFrame of feature scores (rows=samples, cols=feature_indices)
        meta_df: DataFrame with label/pred columns
        feature_index: Index of the feature
        label: Which column to use from meta_df ("label" or "pred")
        prior: Prior probability for label=1 (if None, use empirical prior)

    Returns:
        Posterior probability P(label=1 | feature_score)
    """
    df = scores_df[[feature_index]].join(meta_df[[label]], how="inner")
    groups = {val: df[df[label] == val][feature_index] for val in [0, 1] if val in df[label].unique()}
    stats = {val: (grp.mean(), grp.std() + 1e-8, len(grp)) for val, grp in groups.items()}

    if prior is None:
        counts = df[label].value_counts(normalize=True)
        prior_1 = counts.get(1, 0.5)
    else:
        prior_1 = prior
    prior_0 = 1 - prior_1

    p_score_1 = norm.pdf(feature_score, stats[1][0], stats[1][1]) if 1 in stats else 1e-8
    p_score_0 = norm.pdf(feature_score, stats[0][0], stats[0][1]) if 0 in stats else 1e-8
    numerator = p_score_1 * prior_1
    denominator = numerator + p_score_0 * prior_0
    posterior_1 = numerator / denominator if denominator > 0 else np.nan

    return posterior_1
