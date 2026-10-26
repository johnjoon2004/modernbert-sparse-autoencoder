"""Utility functions for inference page."""
import os
from typing import Dict, Literal, Tuple

# Disable torch.compile to avoid device mismatch issues
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm
from src.activations import ActivationCache
from src.sae import SparseAutoencoder
from transformers import PreTrainedModel, PreTrainedTokenizer

# Disable dynamo
try:
    torch._dynamo.config.disable = True
except AttributeError:
    pass


def predict_text(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float]:
    """
    Predict label and probability for input text.

    Args:
        text: Input text to classify
        model: Classification model
        tokenizer: Tokenizer
        device: Device to run inference on

    Returns:
        Tuple of (predicted_label, probability)
    """
    model.eval()
    model.to(device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        pred_prob = probs[0, pred_label].item()

    return pred_label, pred_prob


def extract_feature_scores(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sae: SparseAutoencoder,
    layer: int,
    live_feature_indices: list[int] | None = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[int, float]:
    """
    Extract feature scores for input text using SAE.

    Args:
        text: Input text
        model: Classification model
        tokenizer: Tokenizer
        sae: Sparse autoencoder model
        layer: Layer index to extract features from
        live_feature_indices: List of valid feature indices to filter (if None, include all)
        device: Device to run inference on

    Returns:
        Dictionary mapping feature_index to feature_score (only scores > 0 and in live_feature_indices)
    """
    model.eval()
    sae.eval()
    model.to(device)
    sae.to(device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=True,
    )

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Capture activations at target layer
    cache = ActivationCache(model)
    with torch.no_grad():
        with cache.capture(inputs, layer_indices=[layer]) as activations:
            activation = activations[layer]  # shape: (batch_size, seq_len, hidden_dim)

    # Extract features using SAE
    activation = activation.to(device)
    with torch.no_grad():
        processed_input = sae._preprocess_input(activation)
        encoded = sae.encoder(processed_input)
        features = sae._apply_activation(encoded)  # shape: (batch_size, seq_len, feature_dim)

        # Sum over sequence length to get per-feature scores
        feature_scores = features.sum(dim=(0, 1)).cpu().numpy()  # shape: (feature_dim,)

    # Filter only activated features (score > 0) and live features
    feature_dict = {}
    for idx, score in enumerate(feature_scores):
        if score > 0:
            # If live_feature_indices provided, only include features in the list
            if live_feature_indices is None or idx in live_feature_indices:
                feature_dict[idx] = float(score)

    return feature_dict


def compute_bayesian_prob(
    feature_score: float,
    scores_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    feature_index: int,
    label: Literal["label", "pred"] = "label",
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


def create_distribution_plot(
    scores_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    feature_index: int,
    current_score: float,
    label: Literal["label", "pred"] = "label",
    color_map: dict | None = None,
    bins: str | int | None = "auto",
) -> dict:
    """
    Create distribution plot data for Streamlit visualization.

    Args:
        scores_df: DataFrame of feature scores
        meta_df: DataFrame with label column
        feature_index: Index of the feature
        current_score: Current feature score to mark on the plot
        label: Which column to use from meta_df
        color_map: Color mapping for different labels
        bins: Histogram bins specification

    Returns:
        Dictionary containing plot data
    """
    if color_map is None:
        color_map = {
            0: (144/255, 238/255, 144/255, 0.7),  # light green (Human)
            1: (240/255, 128/255, 128/255, 0.7),  # light coral (AI)
        }

    df = scores_df[[feature_index]].join(meta_df[[label]], how="inner")

    plot_data = {
        "feature_index": feature_index,
        "current_score": current_score,
        "distributions": [],
    }

    # Collect all data for proper x-axis range
    all_scores = []
    for val in sorted(df[label].unique()):
        subset = df[df[label] == val][feature_index]
        all_scores.extend(subset.values)

        # Compute PDF for smooth curve
        mu, sigma = subset.mean(), subset.std()
        # Extend range for better visualization
        x_min = max(0, mu - 3 * sigma)
        x_max = mu + 3 * sigma
        xs = np.linspace(x_min, x_max, 300)
        pdf_values = norm.pdf(xs, mu, sigma)

        plot_data["distributions"].append({
            "label": val,
            "label_name": "AI" if val == 1 else "Human",
            "color": color_map[val],
            "pdf_x": xs,
            "pdf_y": pdf_values,
            "mean": mu,
            "std": sigma,
        })

    # Add global min/max for axis limits
    plot_data["global_min"] = max(0, min(all_scores))
    plot_data["global_max"] = max(all_scores)

    return plot_data
