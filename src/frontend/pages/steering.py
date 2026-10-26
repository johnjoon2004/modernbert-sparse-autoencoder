import os
from pathlib import Path
from typing import Dict, List, Tuple

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

from src.features import load_score_data
from src.frontend.components.footer import render_footer
from src.frontend.components.header import render_header
from src.frontend.components.setting_panel import render_settings_panel
from src.frontend.components.styles import add_base_styles, add_page_styles
from src.frontend.config import APP_ICON, APP_TITLE, LAYOUT, MODULE_MAP
from src.frontend.utils.inference_utils import extract_feature_scores, predict_text
from src.frontend.utils.session_fetcher import get_ckpt as get_checkpoint
from src.global_settings import settings
from src.sae import SparseAutoencoder
from src.utils.explanation_loader import load_feature_explanation
from src.utils.metrics import (
    calculate_bayesian_prob,
    calculate_cdf,
    calculate_pcc,
    calculate_significances,
)
from src.utils.model_loader import load_sae
from transformers import AutoModelForSequenceClassification, AutoTokenizer

META_DIR = settings.feature_meta_dir
EXPLANATION_DIR = settings.feature_explanation_dir

THRESHOLD_MIN_ACTIVE_SAMPLES = 200

module = MODULE_MAP["steering"]
page_icon = module.get("icon", "🎛️")
page_title = module.get("name", "Steering")
page_desc = module.get("description", "")

st.set_page_config(
    page_title=f"{APP_TITLE.lower()}/steering",
    page_icon=APP_ICON,
    layout=LAYOUT,
)
add_base_styles()
add_page_styles()


# ============================================================================
# Cached Model Loading Functions
# ============================================================================


@st.cache_resource(show_spinner=False)
def load_classification_model() -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
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
def load_feature_metadata(checkpoint: str, layer: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scores_df = pd.DataFrame(load_score_data(checkpoint, layer, metric_name="total_post", meta_dir=Path(META_DIR)))
    meta_df = pd.read_csv(settings.text_dataset_path)
    return scores_df, meta_df


@st.cache_data(show_spinner=False)
def get_live_features(
    checkpoint: str, layer: int, min_active_samples: int = THRESHOLD_MIN_ACTIVE_SAMPLES
) -> List[int]:
    scores_df, _ = load_feature_metadata(checkpoint, layer)
    sample_counts = (scores_df > 0).sum(axis=0)
    sample_counts = sample_counts[sample_counts > 0]
    live_feature_indices = sample_counts[sample_counts > min_active_samples].index.tolist()
    return live_feature_indices


@st.cache_data(show_spinner=False)
def cache_significance_scores(_scores_df: pd.DataFrame, _meta_df: pd.DataFrame) -> pd.Series:
    """Cache significance scores for all features."""
    return calculate_significances(_scores_df, _meta_df, name="significance")


# ============================================================================
# Rendering Functions
# ============================================================================


def render_text_interface():
    """Render the text input interface (left/top side)."""
    st.markdown("### Text Input")
    text_input = st.text_area(
        label="empty",
        label_visibility="collapsed",
        height=200,
        placeholder="Type or paste your text here...",
        key="text_input",
    )

    submit_button = st.button("Submit", type="primary", width="stretch")

    return text_input, submit_button


def render_prediction_interface(
    pred_label: int,
    pred_prob: float,
    title: str = "Prediction",
):
    st.markdown(f"### {title}")

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


def render_steered_prediction_interface(
    pred_label: int,
    pred_prob: float,
    title: str = "Prediction",
    original_prob: float = None,
):
    st.markdown(f"### {title}")

    label_text = "AI" if pred_label == 1 else "Human"
    color = "#FF6B6B" if pred_label == 1 else "#51CF66"

    prob_change_html = ""
    box_height = 160  # 기본 높이

    if original_prob is not None:
        diff = pred_prob - original_prob
        if diff < 0:
            arrow = "▼"
            diff_color = "#DC143C"  # Crimson (Red)
        else:
            arrow = "▲"
            diff_color = "#4169E1"  # Royal Blue

        prob_change_html = f"""
            <p style="color: {diff_color}; font-size: 1.0rem; margin-top: 0.5rem; font-weight: normal;">
                ({arrow} {abs(diff) * 100:.1f}%)
            </p>
        """
        box_height = 190  # 변화 텍스트가 있을 때 높이 증가

    final_html = f"""
    <div style="text-align: center; padding: 2rem; background-color: {color}20; border-radius: 10px; margin: 1rem 0;"><h1 style="color: {color}; font-size: 3rem; margin: 0; font-weight: bold;">{label_text}</h1><p style="color: {color}; font-size: 1.5rem; margin-top: 0.5rem;">({pred_prob * 100:.2f}%)</p>{prob_change_html}</div>
    """

    components.html(final_html, height=240)


def render_feature_table(
    feature_scores: Dict[int, float],
    scores_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    checkpoint: str,
    layer: int,
    significance_cache: pd.Series,
    deactivated_features: set = None,
    forced_deactivated: set = None,
) -> pd.DataFrame:
    """Render feature table with checkboxes for selection."""
    if deactivated_features is None:
        deactivated_features = set()
    if forced_deactivated is None:
        forced_deactivated = set()

    st.markdown("### Active Features")

    if not feature_scores:
        st.info("No active features detected for this text.")
        return None, pd.DataFrame()

    # Prepare feature data
    feature_data = []
    for feature_index, feature_score in feature_scores.items():
        explanation, description = load_feature_explanation(checkpoint, layer, feature_index, postprocess=True)

        # Get significance from cache
        significance = significance_cache[feature_index] if feature_index in significance_cache.index else 0.0

        # Calculate CDF
        cdf = calculate_cdf(
            scores_df=scores_df,
            feature_index=feature_index,
            feature_score=feature_score,
            emperical=True,
        )

        # Calculate AI probability
        ai_prob = calculate_bayesian_prob(
            feature_score=feature_score,
            scores_df=scores_df,
            meta_df=meta_df,
            feature_index=feature_index,
            label="pred",
        )

        pcc_pred = calculate_pcc(
            scores_df=scores_df,
            meta_df=meta_df,
            feature_index=feature_index,
            key="pred",
            flop_negative=False,
        )
        if pcc_pred > 0.1:
            side = "💻"
        elif pcc_pred < -0.1:
            side = "👤"
        else:
            side = ""

        is_deactivated = feature_index in deactivated_features or feature_score == 0
        is_forced = feature_index in forced_deactivated

        feature_data.append(
            {
                "feature_index": feature_index,
                "explanation": explanation[:100] + "..." if len(explanation) > 100 else explanation,
                "description": description[:150] + "..." if len(description) > 150 else description,
                "side": side,
                "significance": significance,
                "score": feature_score,
                "cdf": cdf * 100,
                "cdf_raw": cdf,
                "ai_prob": ai_prob * 100,
                "ai_prob_raw": ai_prob,
                "is_deactivated": is_deactivated,
                "is_forced": is_forced,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(feature_data)
    df = df.sort_values("significance", ascending=False)
    df = df.reset_index(drop=True)

    display_df = df[
        [
            "feature_index",
            "explanation",
            "description",
            "side",
            "significance",
            "score",
            "cdf",
            "ai_prob",
        ]
    ].copy()

    # Format numeric columns for display (use display_df, not df)
    display_df["significance"] = display_df["significance"].apply(lambda x: f"{x:.4f}")
    display_df["score"] = display_df["score"].apply(lambda x: f"{x:.2f}")

    # Add checkbox column with persistent state
    checkbox_data = []
    for idx, row in df.iterrows():
        if row["is_deactivated"]:
            checkbox_data.append(False)
        else:
            checkbox_data.append(row["feature_index"] in st.session_state.get("selected_features", set()))

    display_df.insert(0, "Select", checkbox_data)

    # Build disabled columns list (all columns except Select)
    disabled_cols = [col for col in display_df.columns if col != "Select"]

    # For deactivated rows, also disable the Select checkbox
    disabled = {}
    for idx, row in df.iterrows():
        if row["is_deactivated"]:
            disabled[idx] = ["Select"]

    # Function to determine CDF bar color
    def get_cdf_color(cdf_value):
        if cdf_value >= 75:
            return "#FFA500"  # Orange
        elif cdf_value >= 50:
            return "#FFD700"  # Yellow/Gold
        else:
            return "#D3D3D3"  # Gray

    # Function to determine AI Prob background color
    def get_ai_prob_color(ai_prob_value):
        if ai_prob_value >= 50:
            return "#FFDCDC"  # Light Red
        else:
            return "#DCFFDC"  # Light Green

    st.markdown('<div class="feature-table-container">', unsafe_allow_html=True)

    max_sig = df["significance"].max()
    min_sig = df["significance"].min()

    # Display the table
    edited_df = st.data_editor(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select features to deactivate",
                default=False,
            ),
            "feature_index": st.column_config.NumberColumn("Feature #", format="%d"),
            "explanation": st.column_config.TextColumn("Explanation", width="medium"),
            "description": st.column_config.TextColumn("Description", width="large"),
            "side": st.column_config.TextColumn("Side", help="This feature's correlation side"),
            # "significance": st.column_config.TextColumn("Significance"),
            "significance": st.column_config.ProgressColumn(
                "Significance",
                help="Significance level of the feature",
                format="%.4f",
                min_value=min_sig,
                max_value=max_sig,
            ),
            "score": st.column_config.TextColumn("Score", help="Feature's raw activation score"),
            "cdf": st.column_config.ProgressColumn(
                "CDF",
                help="Cumulative Distribution Function",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "ai_prob": st.column_config.ProgressColumn(
                "AI Probability",
                help="Bayesian probability based on this single feature's score",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
        disabled=disabled_cols,
        key="feature_table",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Apply custom CSS for CDF and AI Probability coloring
    custom_css = """
    <style>
    """

    for idx, row in df.iterrows():
        cdf_val = row["cdf"]
        ai_prob_percent = row["ai_prob"]

        # CDF bar color
        cdf_color = get_cdf_color(cdf_val)

        # AI Probability background color
        ai_prob_bg = get_ai_prob_color(ai_prob_percent)

        custom_css += f"""
        /* CDF Progress bar color */
        .feature-table-container div[data-testid="stDataFrame"] tbody tr:nth-child({idx + 1}) td:nth-child(7) div[role="progressbar"] > div {{
            background-color: {cdf_color} !important;
        }}

        /* AI Probability cell background */
        .feature-table-container div[data-testid="stDataFrame"] tbody tr:nth-child({idx + 1}) td:nth-child(8) {{
            background-color: {ai_prob_bg} !important;
            font-weight: 600 !important;
            color: #333 !important;
        }}
        .feature-table-container div[data-testid="stDataFrame"] tbody tr:nth-child({idx + 1}) td:nth-child(8) > div {{
            background-color: {ai_prob_bg} !important;
        }}
        """

    custom_css += """
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)

    # Apply custom styling for forced and deactivated rows
    for idx, row in df.iterrows():
        if row["is_forced"]:
            st.markdown(
                f"""
                <style>
                .feature-table-container div[data-testid="stDataFrame"] tbody tr:nth-child({idx + 1}) {{
                    background-color: rgba(255, 200, 200, 0.3) !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

        elif row["is_deactivated"]:
            st.markdown(
                f"""
                <style>
                .feature-table-container div[data-testid="stDataFrame"] tbody tr:nth-child({idx + 1}) {{
                    background-color: rgba(200, 200, 200, 0.2) !important;
                    opacity: 0.6;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

    return edited_df, df


def perform_steering(
    text_input: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    sae: SparseAutoencoder,
    layer: int,
    feature_indices: List[int],
    feature_scores: Dict[int, float],
    device: torch.device,
) -> Tuple[int, float, Dict[int, float]]:
    """Perform steering by deactivating selected features simultaneously."""
    from contextlib import contextmanager
    from typing import Callable

    inputs = tokenizer(
        text_input,
        max_length=settings.prev_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    # Create a custom hook that deactivates multiple features at once
    def create_multi_feature_hook(
        sae: SparseAutoencoder,
        feature_indices: List[int],
        device: torch.device,
    ) -> Callable:
        def hook(module, inputs_tuple):
            activation = inputs_tuple[0].to(device)
            sae.to(device)

            with torch.no_grad():
                # (1) Feature Extraction (Encoding)
                processed_input = sae._preprocess_input(activation)
                encoded = sae.encoder(processed_input)
                features_org = sae._apply_activation(encoded)

                # (2) Feature Steering - deactivate ALL selected features
                features_rec = features_org.clone()
                for feature_idx in feature_indices:
                    features_rec[..., feature_idx] = 0.0

                # (3) Reconstruction (Decoding)
                activation_rec = sae.decoder(features_rec)
                if sae.apply_preprocessing:
                    activation_rec = activation_rec + sae.activation_mean.to(device)

            activation_rec = activation_rec.to(activation.device)
            return (activation_rec,) + inputs_tuple[1:]

        return hook

    # Get target module for the specified layer
    try:
        target_module = model.model.layers[layer]  # ModernBERT
    except AttributeError:
        try:
            target_module = model.bert.encoder.layer[layer]  # BERT
        except Exception as e:
            raise ValueError(f"Unsupported model architecture. Error: {e}")

    # Register hook and perform steering
    hook_fn = create_multi_feature_hook(sae, feature_indices, device)
    hook_handle = target_module.register_forward_pre_hook(hook_fn)

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits_steered = outputs.logits.detach().cpu()

        # Calculate final prediction
        probs = torch.softmax(logits_steered, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        pred_prob = probs[0, pred_label].item()

        # If we always want the prob of 'AI' (label 1), use this instead:
        # pred_prob_ai = probs[0, 1].item()
        # But the original code tracked the prob of the *predicted* label, so we keep that.

    finally:
        # Remove hook
        if hook_handle:
            hook_handle.remove()

    # Extract new feature scores after steering
    new_feature_scores = extract_feature_scores(
        text=text_input,
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        layer=layer,
        live_feature_indices=list(feature_scores.keys()),
        device=device,
    )

    return pred_label, pred_prob, new_feature_scores


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

    render_settings_panel()
    checkpoint = get_checkpoint()
    layer = st.session_state.get("layer", 21)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    with st.spinner("Loading models..."):
        model, tokenizer = load_classification_model()
        model.to(device)
        sae = load_sae_model(checkpoint, layer)
        sae.to(device)

        if sae is None:
            st.error(f"SAE model not found for {checkpoint} at layer {layer}")
            return

    with st.spinner("Initializing feature metadata..."):
        scores_df, meta_df = load_feature_metadata(checkpoint, layer)
        live_features = get_live_features(checkpoint, layer)
        significance_cache = cache_significance_scores(scores_df, meta_df)

    # Initialize session state
    if "original_prediction" not in st.session_state:
        st.session_state.original_prediction = None
    if "original_feature_scores" not in st.session_state:
        st.session_state.original_feature_scores = {}
    if "steered_prediction" not in st.session_state:
        st.session_state.steered_prediction = None
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = set()
    if "forced_deactivated" not in st.session_state:
        st.session_state.forced_deactivated = set()
    if "current_text" not in st.session_state:
        st.session_state.current_text = ""

    # Main interface layout
    st.markdown("---")

    # Section 1 & 2: Text input and original prediction
    col_text, col_pred = st.columns([1, 1])
    with col_text:
        text_input, submit_button = render_text_interface()

    # Process submission
    if submit_button and text_input.strip():
        # Reset state for new text
        if text_input != st.session_state.current_text:
            st.session_state.current_text = text_input
            st.session_state.selected_features = set()
            st.session_state.forced_deactivated = set()
            st.session_state.steered_prediction = None

        with st.spinner("Processing..."):
            # Get original prediction
            pred_label, pred_prob = predict_text(text_input, model, tokenizer, device)

            # Extract feature scores
            feature_scores = extract_feature_scores(
                text=text_input,
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                layer=layer,
                live_feature_indices=live_features,
                device=device,
            )

            st.session_state.original_prediction = (pred_label, pred_prob)
            st.session_state.original_feature_scores = feature_scores

        with col_pred:
            render_prediction_interface(pred_label, pred_prob, title="Original Prediction")

    elif st.session_state.original_prediction is not None:
        # Display existing prediction
        with col_pred:
            pred_label, pred_prob = st.session_state.original_prediction
            render_prediction_interface(pred_label, pred_prob, title="Original Prediction")

    # # Section 3: Feature list with checkboxes
    if st.session_state.original_feature_scores:
        st.markdown("---")

        # Get deactivated features (forced or naturally zero)
        deactivated_features = set()

        # if st.session_state.steered_prediction is not None:
        # After steering, mark features with score 0 as deactivated
        for feat_idx, score in st.session_state.original_feature_scores.items():
            if score == 0:
                deactivated_features.add(feat_idx)

        # Also add previously forced ones, even if they had a score
        # (This handles the state after steering)
        deactivated_features.update(st.session_state.forced_deactivated)

        edited_df, original_df = render_feature_table(
            feature_scores=st.session_state.original_feature_scores,
            scores_df=scores_df,
            meta_df=meta_df,
            checkpoint=checkpoint,
            layer=layer,
            significance_cache=significance_cache,
            deactivated_features=deactivated_features,
            forced_deactivated=st.session_state.forced_deactivated,
        )

        # Extract selected features from edited DataFrame
        if edited_df is not None and not original_df.empty:
            selected_indices = []
            for idx, row in edited_df.iterrows():
                if idx in original_df.index:
                    if row["Select"] and not original_df.iloc[idx]["is_deactivated"]:
                        selected_indices.append(original_df.iloc[idx]["feature_index"])

            st.session_state.selected_features = set(selected_indices)

        # Retry button
        col_button1, _, _, col_button2 = st.columns([1, 1, 1, 1])
        with col_button1:
            retry_button = st.button(
                "Apply Steering",
                type="primary",
                width="stretch",
                disabled=len(st.session_state.selected_features) == 0,
            )

        with col_button2:
            is_reset_disabled = not (st.session_state.selected_features or st.session_state.steered_prediction)
            reset_button = st.button(
                "Reset Selected",
                use_container_width=True,
                disabled=is_reset_disabled,
            )

        if reset_button:
            st.session_state.selected_features = set()
            st.session_state.forced_deactivated = set()
            st.session_state.steered_prediction = None
            st.rerun()

        # Section 4: Steered prediction
        if retry_button and st.session_state.selected_features:
            with st.spinner("Applying steering..."):
                selected_list = list(st.session_state.selected_features)
                pred_label_steered, pred_prob_steered, new_feature_scores = perform_steering(
                    text_input=st.session_state.current_text,
                    model=model,
                    tokenizer=tokenizer,
                    sae=sae,
                    layer=layer,
                    feature_indices=selected_list,
                    feature_scores=st.session_state.original_feature_scores,
                    device=device,
                )

                st.session_state.steered_prediction = (
                    pred_label_steered,
                    pred_prob_steered,
                )
                st.session_state.forced_deactivated = st.session_state.selected_features.copy()

            st.rerun()

        # Display steered prediction if available
        if st.session_state.steered_prediction is not None:
            st.markdown("---")
            st.info(f"**Deactivated Features:** {', '.join(map(str, sorted(st.session_state.forced_deactivated)))}")

            col_steered1, col_steered2 = st.columns([1, 1])

            pred_label, pred_prob = st.session_state.original_prediction

            with col_steered1:
                render_prediction_interface(pred_label, pred_prob, title="Original Prediction")

            with col_steered2:
                pred_label_steered, pred_prob_steered = st.session_state.steered_prediction
                original_label, original_prob = st.session_state.original_prediction
                render_steered_prediction_interface(
                    pred_label_steered,
                    pred_prob_steered,
                    title="Steered Prediction",
                    original_prob=original_prob,
                )

    render_footer(add_ceiling=False)


if __name__ == "__main__":
    main()
