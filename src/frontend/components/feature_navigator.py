import random
from typing import List

import streamlit as st
from src.auto_explainer.explanations import FeatureExplanations
from src.features import get_available_features
from src.frontend.config import MAX_FEATURE_INDEX
from src.global_settings import settings


@st.cache_data(show_spinner=False)
def get_available_features_cached(checkpoint: str, layer: int) -> List[int]:
    return get_available_features(checkpoint, layer, strict=False)


def get_previous_available_feature_index(current_feature_index: int, available_features: List[int]) -> int:
    previous_features = [f for f in available_features if f < current_feature_index]
    if previous_features:
        return max(previous_features)
    return available_features[0]


def get_next_available_feature_index(current_feature_index: int, available_features: List[int]) -> int:
    next_features = [f for f in available_features if f > current_feature_index]
    if next_features:
        return min(next_features)
    return available_features[-1]


def render_feature_navigation(model, hparam, layer, current_feature):
    checkpoint = f"{model}_{hparam}"
    checkpoint_layer = f"{checkpoint}_L{layer}"

    available_features = get_available_features_cached(checkpoint, layer)

    st.markdown(
        f'<h1 class="page-header"> <b>{checkpoint_layer}</b> &nbsp - &nbsp <b><u>#{current_feature}</u></b> </h1>',
        unsafe_allow_html=True,
    )

    nav_cols = st.columns([2.5, 1, 1, 1, 2.5])

    # Move Buttons
    with nav_cols[1]:
        if st.button("⬅️ Previous", use_container_width=True):
            if current_feature > 0:
                previous_feature = get_previous_available_feature_index(current_feature, available_features)
                st.session_state.feature_index = previous_feature
                st.rerun()

    with nav_cols[2]:
        if st.button("🔄 Random", use_container_width=True):
            if available_features:
                st.session_state.feature_index = random.choice(available_features)
                st.rerun()

    with nav_cols[3]:
        if st.button("➡️ Next", use_container_width=True):
            next_feature = get_next_available_feature_index(current_feature, available_features)
            st.session_state.feature_index = next_feature
            st.rerun()

    st.markdown(
        f'<h2 class="section-header"> 🌐 Feature Navigation </h2>',
        unsafe_allow_html=True,
    )

    explanation_path = settings.feature_explanation_dir / checkpoint / str(layer) / f"{current_feature}.jsonl"
    try:
        feature_explanations = FeatureExplanations.load_from_jsonl(explanation_path)
        best_explanation = (
            f"`{feature_explanations.get_best_explanation().explanation}`"
            if feature_explanations.get_best_explanation()
            else "`N/A`"
        )
        description = (
            f"`{feature_explanations.explanations[1].explanation}`"
            if len(feature_explanations.explanations) > 1
            else "``"
        )

    except FileNotFoundError:
        best_explanation = "N/A"
        description = ""

    # except Exception as e:
    #     st.error(f"❌ Error loading explanations: {e}")
    #     best_explanation = "N/A"
    #     description = ""

    # Current Feature Info
    a, b, c, d, e = st.columns(5)
    a.info(f"Checkpoint: `{checkpoint}`")
    b.info(f"Layer: `#{layer}`")
    c.info(f"Current Feature: `#{st.session_state.feature_index}`")
    d.info(f"Available Features: `{len(available_features)} / {MAX_FEATURE_INDEX[model]}`")
    e.info(f"Feature Signifiance: `-`")

    st.info(f"Explanation: {best_explanation} \n\n {description}")
