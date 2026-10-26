import base64
import os
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch

# Disable torch.compile to avoid device mismatch issues with ModernBERT
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Additional safeguard: disable dynamo
try:
    torch._dynamo.config.disable = True
except AttributeError:
    pass
from src.auto_explainer.explanations import FeatureExplanations
from src.features import load_score_data
from src.frontend.components.footer import render_footer
from src.frontend.components.header import render_header
from src.frontend.components.setting_panel import render_settings_panel
from src.frontend.components.styles import add_base_styles, add_page_styles
from src.frontend.config import APP_ICON, APP_TITLE, LAYOUT, MODULE_MAP
from src.frontend.utils.inference_utils import (
    compute_bayesian_prob,
    create_distribution_plot,
    extract_feature_scores,
    predict_text,
)
from src.frontend.utils.session_fetcher import get_ckpt as get_checkpoint
from src.global_settings import settings
from src.sae import SparseAutoencoder
from src.utils.metrics import calculate_cdf, calculate_pcc
from src.utils.model_loader import load_sae
from src.utils.token_activation_scorer import TokenActivationScorer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

META_DIR = settings.feature_meta_dir
EXPLANATION_DIR = settings.feature_explanation_dir

TOP_K_FEATURES_TO_SHOW = 20
THRESHOLD_MIN_ACTIVE_SAMPLES = 200

module = MODULE_MAP["inference"]
page_icon = module.get("icon", "✍🏻")
page_title = module.get("name", "Inference")
page_desc = module.get("description", "")

st.set_page_config(
    page_title=f"{APP_TITLE.lower()}/inference",
    page_icon=APP_ICON,
    layout=LAYOUT,
)
add_base_styles()
add_page_styles()


# ============================================================================
# Cached Model Loading Functions
# ============================================================================


@st.cache_resource(show_spinner=False)
def load_classification_model() -> (
    Tuple[AutoModelForSequenceClassification, AutoTokenizer]
):
    model_path = settings.lm_weights_path
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_sae_model(checkpoint: str, layer: int) -> SparseAutoencoder:
    sae = load_sae(checkpoint=checkpoint, layer_idx=layer, return_ckpt=False)
    sae.eval()
    return sae


@st.cache_data(show_spinner=False)
def load_feature_metadata(
    checkpoint: str, layer: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scores_df = pd.DataFrame(
        load_score_data(
            checkpoint, layer, metric_name="total_post", meta_dir=Path(META_DIR)
        )
    )
    meta_df = pd.read_csv(settings.text_dataset_path)
    return scores_df, meta_df


@st.cache_data(show_spinner=False)
def get_live_features(
    checkpoint: str, layer: int, min_active_samples: int = THRESHOLD_MIN_ACTIVE_SAMPLES
) -> List[int]:
    scores_df, _ = load_feature_metadata(checkpoint, layer)
    sample_counts = (scores_df > 0).sum(axis=0)
    sample_counts = sample_counts[sample_counts > 0]
    live_feature_indices = sample_counts[
        sample_counts > min_active_samples
    ].index.tolist()
    return live_feature_indices


def load_feature_explanation(
    checkpoint: str, layer: int, feature_index: int
) -> tuple[str, str]:
    """Load explanation and description for a feature."""
    explanation_path = (
        EXPLANATION_DIR / checkpoint / str(layer) / f"{feature_index}.jsonl"
    )
    try:
        feature_explanations = FeatureExplanations.load_from_jsonl(explanation_path)
        explanation = (
            feature_explanations.get_best_explanation().explanation
            if feature_explanations.get_best_explanation()
            else "N/A"
        )
        description = (
            feature_explanations.explanations[1].explanation
            if len(feature_explanations.explanations) > 1
            else ""
        )
    except FileNotFoundError:
        explanation = "N/A"
        description = ""

    return explanation, description


# ============================================================================
# Rendering Functions
# ============================================================================


def _highlight_tokens_html(
    text: str,
    token_scores: List[dict],
    tokenizer: AutoTokenizer,
) -> str:
    """
    Convert text and token scores to highlighted HTML.

    Args:
        text: Input text
        token_scores: List of dicts with 'token_idx', 'token_str', 'token_score'
        tokenizer: Tokenizer for tokenization

    Returns:
        HTML string with highlighted tokens
    """
    if not token_scores:
        return f"<div>{text}</div>"

    # Create token score map
    token_score_map = {
        t["token_idx"]: t["token_score"]
        for t in token_scores
        if t.get("token_score") is not None
    }

    if not token_score_map:
        return f"<div>{text}</div>"

    # Tokenize the text
    all_tokens = tokenizer.tokenize(
        text, truncation=True, max_length=settings.prev_seq_len
    )

    # Calculate min-max for normalization (only for scores > 0)
    positive_scores = [s for s in token_score_map.values() if s > 0]
    if positive_scores:
        min_score = min(positive_scores)
        max_score = max(positive_scores)
        score_range = max_score - min_score if max_score != min_score else 1.0
    else:
        # No positive scores, set defaults
        min_score = 0
        max_score = 1
        score_range = 1.0

    # Build highlighted HTML
    highlighted_tokens = []
    for idx_, token in enumerate(all_tokens):
        idx = idx_ + 1  # Skip CLS token (index 0)

        # Handle token prefix and ## for subword tokens
        if idx == 1:
            prefix = ""
            _token = token
        elif token.startswith("##"):
            prefix = ""
            _token = token[2:]
        else:
            prefix = " "
            _token = token

        # Apply highlighting if token has a score and score > 0
        if idx in token_score_map:
            score = token_score_map[idx]
            if score > 0:
                intensity = (score - min_score) / score_range
                color = f"rgba(255, {255 - int(intensity * 100)}, 0, {0.3 + intensity * 0.7})"
                token_html = (
                    f'<span style="background-color: {color}; padding: 1px 1.5px; '
                    f'border-radius: 3px; cursor: help;" '
                    f'title="Token: {token}\nScore: {score:.2f}">{_token}</span>'
                )
            else:
                # Score is 0 or negative, display without background
                token_html = f"<span>{_token}</span>"
        else:
            token_html = f"<span>{_token}</span>"

        highlighted_tokens.append(f"<span>{prefix}</span>")
        highlighted_tokens.append(token_html)

    html = (
        '<div style="line-height:1.5; font-size:1rem; font-family:inherit; '
        'overflow-wrap:break-word; word-break:break-all;">'
        f"{"".join(highlighted_tokens)}</div>"
    )
    return html


def render_text_interface():
    """Render the text input interface (left side)."""
    st.markdown("### Text Input")
    text_input = st.text_area(
        label="empty",
        label_visibility="collapsed",
        height=200,
        placeholder="Type or paste your text here...",
        key="text_input",
    )

    submit_button = st.button("Submit", type="primary", use_container_width=True)

    return text_input, submit_button


def render_prediction_interface(pred_label: int, pred_prob: float):
    """Render the prediction result interface (right side)."""
    st.markdown("### Prediction")

    # Display prediction result
    label_text = "AI" if pred_label == 1 else "Human"
    color = "#FF6B6B" if pred_label == 1 else "#51CF66"

    st.markdown(
        f"""
        <div style="text-align: center; padding: 2rem; background-color: {color}20; border-radius: 10px; margin: 1rem 0;">
            <h1 style="color: {color}; font-size: 3rem; margin: 0; font-weight: bold;">
                {label_text}
            </h1>
            <p style="color: {color}; font-size: 1.5rem; margin-top: 0.5rem;">
                ({pred_prob * 100:.2f}%)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# def render_feature_card(
#     feature_index: int,
#     explanation: str,
#     description: str,
#     plot_data: dict,
#     feature_score: float,  # first score
#     calibrated_score: float,  # second score
#     cdf: float,  # third score
#     explanation_max_len: int = 150,
#     description_max_len: int = 500,
# ):
#     """Render a single feature card with explanation and distribution."""

#     if len(explanation) > explanation_max_len:
#         explanation = explanation[:explanation_max_len] + "..."

#     description = description.strip()
#     description = description or "N/A"
#     if len(description) > description_max_len:
#         if description.endswith("**"):
#             description = description[:description_max_len] + " ...**"
#         else:
#             description = description[:description_max_len] + "..."

#     with st.container():
#         metrics_html = '<div style="display: flex; gap: 1rem;">'
#         metrics = [
#             {"name": "Score", "value": f"{feature_score:.2f}", "color": "#a9a9a9"},
#             {
#                 "name": "Bayesian",
#                 "value": f"{calibrated_score * 100:.0f}%",
#                 "color": "#51cf66" if calibrated_score < 0.5 else "#d32f2f",
#             },
#             {
#                 "name": "CDF",
#                 "value": f"{cdf:.2f}",
#                 "color": "#a9a9a9" if cdf < 0.5 else "#8b00ff",
#             },
#         ]
#         for metric in metrics:
#             metric_name = metric["name"]
#             metric_value = metric["value"]
#             metric_color = metric["color"]
#             metric_html = f"""<div style="flex: 1; background-color: {metric_color}20; padding: 0.5rem 0.75rem; border-radius: 6px; border-left: 3px solid {metric_color}; text-align: center;">
#                 <div style="font-size: 0.7rem; color: #666; margin-bottom: 0.2rem;">{metric_name}</div>
#                 <div style="font-size: 1.1rem; font-weight: 600; color: {metric_color};">{metric_value}</div></div>"""
#             metrics_html += metric_html
#         metrics_html += "</div>"

#         st.markdown(
#             f"""
#             <div style="background-color: #fafafa; padding: 1.25rem; border-radius: 10px 10px 0 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
#                 <div style="display: flex; justify-content: space-between; align-items: center;">
#                     <h4 style="margin: 0; color: #1976d2; font-weight: 600;">Feature #{feature_index}</h4>
#                     <hr style="margin: 1rem 0 0 0; border: none; border-top: 1px solid #e0e0e0;">
#                     {metrics_html}
#             </div>""",
#             unsafe_allow_html=True,
#         )

#         # Card body container
#         st.markdown(
#             '<div style="background-color: #fafafa; padding: 0 1.25rem 1.25rem 1.25rem; border-radius: 0 0 10px 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 1.5rem;">',
#             unsafe_allow_html=True,
#         )

#         # Explanation and Description
#         st.markdown(f"_{explanation}_")
#         st.info(f"`{description}`")

#         # Distribution plot (KDE only)
#         fig, ax = plt.subplots(figsize=(10, 3.5))

#         # Track max PDF value for y-axis limit
#         max_pdf = 0
#         for dist in plot_data["distributions"]:
#             # Plot PDF curve only
#             ax.plot(
#                 dist["pdf_x"],
#                 dist["pdf_y"],
#                 color=dist["color"][:3],
#                 lw=2.5,
#                 label=dist["label_name"],
#                 alpha=0.9,
#             )
#             # Fill under the curve for better visualization
#             ax.fill_between(
#                 dist["pdf_x"],
#                 dist["pdf_y"],
#                 alpha=0.2,
#                 color=dist["color"][:3],
#             )
#             max_pdf = max(max_pdf, dist["pdf_y"].max())

#         # Mark current score
#         ax.axvline(
#             plot_data["current_score"],
#             color="red",
#             linestyle="--",
#             linewidth=2,
#             label="Current Score",
#             alpha=0.8,
#         )

#         # Set axis limits for better visualization
#         ax.set_xlim(plot_data["global_min"], plot_data["global_max"])
#         ax.set_ylim(0, max_pdf * 1.1)

#         ax.set_xlabel("Feature Score", fontsize=10)
#         ax.set_ylabel("Density", fontsize=10)
#         ax.legend(loc="upper right", fontsize=9)
#         ax.grid(True, alpha=0.3, linestyle="--")
#         ax.tick_params(labelsize=9)
#         plt.tight_layout()

#         st.pyplot(fig)
#         plt.close()


def render_feature_card(
    feature_index: int,
    explanation: str,
    description: str,
    plot_data: dict,
    feature_score: float,
    calibrated_score: float,
    cdf: float,
    highlighted_text_html: str = "",
    explanation_max_len: int = 250,
    description_max_len: int = 500,
):
    """Render a single feature card (with metrics, explanation beside title, highlighted text, and chart)."""

    explanation = explanation.strip() or "N/A"
    explanation = explanation.replace("**", "")
    if len(explanation) > explanation_max_len:
        explanation = explanation[:explanation_max_len] + "..."

    description = description.strip() or "N/A"
    description = description.replace("**", "")
    if len(description) > description_max_len:
        description = description[:description_max_len] + "..."

    # --- Metrics HTML ---
    metrics = [
        {"name": "Score", "value": f"{feature_score:.2f}", "color": "#a9a9a9"},
        {
            "name": "AI Probability (Bayesian)",
            "value": f"{calibrated_score * 100:.0f}%",
            "color": "#51cf66" if calibrated_score < 0.5 else "#d32f2f",
        },
        {
            "name": "Significance (CDF)",
            "value": f"{cdf * 100:.0f}%",
            "color": "#a9a9a9" if cdf < 0.5 else "#8b00ff",
        },
    ]

    metrics_html = '<div style="display: flex; gap: 1rem; margin-top: 0.75rem;">'
    for metric in metrics:
        metrics_html += f"""
        <div style="flex: 1; background-color: {metric['color']}20;
                    padding: 0.5rem 0.75rem; border-radius: 6px;
                    border-left: 3px solid {metric['color']};
                    text-align: center;">
            <div style="font-size: 0.7rem; color: #666; margin-bottom: 0.2rem;">
                {metric['name']}
            </div>
            <div style="font-size: 1.1rem; font-weight: 600; color: {metric['color']};">
                {metric['value']}
            </div>
        </div>
        """
    metrics_html += "</div>"

    # --- Matplotlib chart ---
    fig, ax = plt.subplots(figsize=(8, 3.0))
    max_pdf = 0
    for dist in plot_data["distributions"]:
        ax.plot(
            dist["pdf_x"],
            dist["pdf_y"],
            color=dist["color"][:3],
            lw=2.5,
            label=dist["label_name"],
            alpha=0.9,
        )
        ax.fill_between(
            dist["pdf_x"],
            dist["pdf_y"],
            alpha=0.2,
            color=dist["color"][:3],
        )
        max_pdf = max(max_pdf, dist["pdf_y"].max())

    ax.axvline(
        plot_data["current_score"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Current Score",
        alpha=0.8,
    )
    ax.set_xlim(plot_data["global_min"], plot_data["global_max"])
    ax.set_ylim(0, max_pdf * 1.1)
    ax.set_xlabel("Feature Score", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    # --- Encode plot as base64 ---
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    img_html = f'<img src="data:image/png;base64,{data}" style="width:100%; border-radius: 8px; margin-top: 0.5rem;">'

    # --- Build highlighted text section if available ---
    highlighted_text_section = ""
    if highlighted_text_html:
        # highlighted_text_section = f"""
        # <div style="margin-top: 1rem; background-color: #ffffff; padding: 0.75rem; border-radius: 6px; border: 1px solid #e0e0e0;">
        #     <div style="font-size: 0.85rem; color: #666; margin-bottom: 0.5rem; font-weight: 500;">Token Activation:</div>
        #     {highlighted_text_html}
        # </div>
        # """
        highlighted_text_section = f"""
        <div style="margin-top: 1rem; background-color: #ffffff; padding: 0.75rem; border-radius: 6px; color: #333; border: 1px solid #e0e0e0;">
            {highlighted_text_html}
        </div>
        """

    # --- Full Card HTML ---
    card_html = f"""
    <div style="
        background-color: #f4f4f4;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        padding: 1.25rem;
    ">
        <!-- Header: Feature title + Explanation -->
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem;">
            <h4 style="margin: 0; color: #1976d2; font-weight: 600; white-space: nowrap;">
                Feature #{feature_index}
            </h4>
            <p style="margin: 0; font-weight: 600; font-style: italic; color: #000; font-size: 0.9rem; flex: 1;">
                {explanation}
            </p>
        </div>

        <div style="background-color: #ffffff; font-style: italic; padding: 0.75rem; border-radius: 6px; color: #333; border: 1px solid #e0e0e0;">
            {description}
        </div>

        {metrics_html}

        <hr style="margin: 1rem 0; border: none; border-top: 1px solid #ddd;">
        
        {highlighted_text_section}

        {img_html}
    </div>
    """

    # Adjust height based on whether highlighted text is included
    card_height = 650 if highlighted_text_html else 520
    components.html(card_html, height=card_height, scrolling=True)


def render_explanation_interface(
    feature_scores: dict,
    scores_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    checkpoint: str,
    layer: int,
    pred: int,
    text_input: str,
    sae: SparseAutoencoder,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
):
    """Render the feature explanation interface (bottom)."""
    st.markdown("---")

    # feature_scores already contains only live features with score > 0
    if not feature_scores:
        st.info("No active features detected for this text.")
        return

    # Initialize TokenActivationScorer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_scorer = TokenActivationScorer(model=sae, tokenizer=tokenizer, device=device)

    # Compute calibrated scores for all active features
    feature_data = []
    for feature_index, feature_score in feature_scores.items():
        calibrated_score = compute_bayesian_prob(
            feature_score=feature_score,
            scores_df=scores_df,
            meta_df=meta_df,
            feature_index=feature_index,
            label="pred",
        )

        # Calculate PCC to determine AI-like vs Human-like
        pcc_label = calculate_pcc(
            scores_df=scores_df,
            meta_df=meta_df,
            feature_index=feature_index,
            key="label",
        )

        # CDF value
        cdf = calculate_cdf(
            scores_df=scores_df,
            feature_index=feature_index,
            feature_score=feature_score,
            emperical=True,
        )

        feature_data.append(
            {
                "feature_index": feature_index,
                "feature_score": feature_score,
                "calibrated_score": calibrated_score,
                "cdf": cdf,
                "pcc_label": pcc_label,
                "is_ai_like": pcc_label > 0,  # Positive PCC means AI-like
            }
        )

    # Filter by 'Significance (CDF)' >= 0.1
    feature_data = [f for f in feature_data if f["cdf"] >= 0.1]

    # Sort by 'Significance (CDF)' -> Split into AI-like and Human-like features and apply TOP_K limit
    feature_data = sorted(feature_data, key=lambda x: x["cdf"], reverse=True)

    human_like_features = [f for f in feature_data if not f["is_ai_like"]][
        :TOP_K_FEATURES_TO_SHOW
    ]
    ai_like_features = [f for f in feature_data if f["is_ai_like"]][
        :TOP_K_FEATURES_TO_SHOW
    ]

    # if pred == 1:
    #     # Prediction is 'AI'
    #     feature_data = sorted(feature_data, key=lambda x: x["calibrated_score"], reverse=True)
    #     human_like_features = [f for f in feature_data if not f["is_ai_like"]][:TOP_K_FEATURES_TO_SHOW]
    #     ai_like_features = [f for f in feature_data if f["is_ai_like"]][:TOP_K_FEATURES_TO_SHOW]
    # else:
    #     # Prediction is 'Human'
    #     feature_data = sorted(feature_data, key=lambda x: abs(x["calibrated_score"]), reverse=False)
    #     human_like_features = [f for f in feature_data if not f["is_ai_like"]][:TOP_K_FEATURES_TO_SHOW]
    #     ai_like_features = [f for f in feature_data if f["is_ai_like"]][:TOP_K_FEATURES_TO_SHOW]

    # Render AI-like and Human-like features
    max_rows = max(len(ai_like_features), len(human_like_features))

    c = st.container()
    with c:
        col_human, col_ai = st.columns(2)
        with col_human:
            st.markdown(
                '<h4 style="margin-bottom: 1rem;">🙋🏻‍♂️ Human-like Features</h4>',
                unsafe_allow_html=True,
            )
            if not human_like_features:
                st.info("No Human-like features detected.")
        with col_ai:
            st.markdown(
                '<h4 style="margin-bottom: 1rem;">🖥️ AI-like Features</h4>',
                unsafe_allow_html=True,
            )
            if not ai_like_features:
                st.info("No AI-like features detected.")

    for i in range(max_rows):
        c = st.container()
        with c:
            col_human, col_ai = st.columns(2)
            with col_human:
                if i < len(human_like_features):
                    feat = human_like_features[i]
                    explanation, description = load_feature_explanation(
                        checkpoint, layer, feat["feature_index"]
                    )
                    plot_data = create_distribution_plot(
                        scores_df=scores_df,
                        meta_df=meta_df,
                        feature_index=feat["feature_index"],
                        current_score=feat["feature_score"],
                        label="label",
                    )

                    # Compute token scores for this feature
                    token_scores, _ = token_scorer.compute_token_scores(
                        text=text_input,
                        feature_idx=feat["feature_index"],
                        layer_idx=layer,
                        lm_model=model,
                        exclude_special_tokens=True,
                    )

                    # Generate highlighted HTML
                    highlighted_html = _highlight_tokens_html(
                        text=text_input,
                        token_scores=token_scores,
                        tokenizer=tokenizer,
                    )

                    render_feature_card(
                        feature_index=feat["feature_index"],
                        feature_score=feat["feature_score"],
                        calibrated_score=feat["calibrated_score"],
                        cdf=feat["cdf"],
                        explanation=explanation,
                        description=description,
                        plot_data=plot_data,
                        highlighted_text_html=highlighted_html,
                    )
            with col_ai:
                if i < len(ai_like_features):
                    feat = ai_like_features[i]
                    explanation, description = load_feature_explanation(
                        checkpoint, layer, feat["feature_index"]
                    )
                    plot_data = create_distribution_plot(
                        scores_df=scores_df,
                        meta_df=meta_df,
                        feature_index=feat["feature_index"],
                        current_score=feat["feature_score"],
                        label="label",
                    )

                    # Compute token scores for this feature
                    token_scores, _ = token_scorer.compute_token_scores(
                        text=text_input,
                        feature_idx=feat["feature_index"],
                        layer_idx=layer,
                        lm_model=model,
                        exclude_special_tokens=True,
                    )

                    # Generate highlighted HTML
                    highlighted_html = _highlight_tokens_html(
                        text=text_input,
                        token_scores=token_scores,
                        tokenizer=tokenizer,
                    )

                    render_feature_card(
                        feature_index=feat["feature_index"],
                        feature_score=feat["feature_score"],
                        calibrated_score=feat["calibrated_score"],
                        cdf=feat["cdf"],
                        explanation=explanation,
                        description=description,
                        plot_data=plot_data,
                        highlighted_text_html=highlighted_html,
                    )


# ============================================================================
# Main Page Logic
# ============================================================================


def main():
    render_header(
        title=page_title,
        icon=page_icon,
        description=page_desc,
        gradient=True,
        add_floor=False,
    )

    # Settings panel
    render_settings_panel()

    # Get current settings
    checkpoint = get_checkpoint()
    layer = st.session_state.get("layer", 21)

    # Load models
    with st.spinner("Loading models..."):
        model, tokenizer = load_classification_model()
        sae = load_sae_model(checkpoint, layer)

        if sae is None:
            st.error(f"SAE model not found for {checkpoint} at layer {layer}")
            return

        scores_df, meta_df = load_feature_metadata(checkpoint, layer)
        live_features = get_live_features(checkpoint, layer)

    # Main interface layout
    st.markdown("---")

    # Text input and prediction in columns
    col_text, col_pred = st.columns([1, 1])

    with col_text:
        text_input, submit_button = render_text_interface()

    # Process submission
    if submit_button and text_input.strip():
        with st.spinner("Processing..."):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 1. Get prediction
            pred_label, pred_prob = predict_text(text_input, model, tokenizer, device)

            # 2. Extract feature scores (filtered by live features)
            feature_scores = extract_feature_scores(
                text=text_input,
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                layer=layer,
                live_feature_indices=live_features,
                device=device,
            )

        # Display prediction
        with col_pred:
            render_prediction_interface(pred_label, pred_prob)

        # Display feature explanations
        render_explanation_interface(
            feature_scores=feature_scores,
            scores_df=scores_df,
            meta_df=meta_df,
            checkpoint=checkpoint,
            layer=layer,
            pred=pred_label,
            text_input=text_input,
            sae=sae,
            tokenizer=tokenizer,
            model=model,
        )

    elif submit_button:
        st.warning("Please enter some text to analyze.")

    render_footer(add_ceiling=False)


if __name__ == "__main__":
    main()
