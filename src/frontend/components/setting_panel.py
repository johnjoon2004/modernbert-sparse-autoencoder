import streamlit as st
from src.frontend.components.feature_navigator import get_available_features_cached
from src.frontend.config import (
    ALLOWED_DOMAINS,
    ALLOWED_HPARAMSET_TAGS,
    ALLOWED_LAYERS,
    ALLOWED_MODEL_TAGS,
    FEATURE_RECORD_DIR,
    MAX_FEATURE_INDEX,
    MIN_FEATURE_INDEX,
)


def render_settings_panel(show_feature_selector: bool = True):
    # model configuration
    st.sidebar.subheader("⚙️ Model Configuration")

    model = st.sidebar.selectbox(
        "Model",
        options=ALLOWED_MODEL_TAGS,
        index=ALLOWED_MODEL_TAGS.index(st.session_state.get("model", ALLOWED_MODEL_TAGS[0])),
        key="model_selector",
    )

    layer = st.sidebar.selectbox(
        "Layer",
        options=ALLOWED_LAYERS,
        index=ALLOWED_LAYERS.index(st.session_state.get("layer", ALLOWED_LAYERS[0])),
        key="layer_selector",
    )

    hparam = st.sidebar.selectbox(
        "Hyperparameter",
        options=ALLOWED_HPARAMSET_TAGS,
        index=ALLOWED_HPARAMSET_TAGS.index(st.session_state.get("hyperparameter", ALLOWED_HPARAMSET_TAGS[0])),
        key="hparam_selector",
    )

    domain = st.sidebar.selectbox(
        "Domain",
        options=ALLOWED_DOMAINS,
        index=ALLOWED_DOMAINS.index(st.session_state.get("domain", ALLOWED_DOMAINS[0])),
        key="domain_selector",
        help="This option doesn't affect the features page",
    )

    # update session state
    st.session_state.model = model
    st.session_state.layer = layer
    st.session_state.hyperparameter = hparam
    st.session_state.domain = domain

    st.sidebar.markdown("---")

    # feature index selection
    if show_feature_selector:
        st.sidebar.subheader("⚙️ Feature Selection")

        checkpoint = f"{model}_{hparam}"
        checkpoint_layer = f"{model}_{hparam}_L{layer}"

        available_features = get_available_features_cached(checkpoint=checkpoint, layer=layer)

        max_idx = MAX_FEATURE_INDEX.get(model, 960) - 1
        current_idx = st.session_state.get("feature_index", MIN_FEATURE_INDEX)

        if available_features:
            if current_idx not in available_features:
                st.session_state.feature_index = available_features[0]
        else:
            if current_idx > max_idx:
                st.session_state.feature_index = max_idx

        if available_features:
            try:
                list_index = available_features.index(st.session_state.feature_index)
            except ValueError:
                list_index = 0

        def update_feature_index_from_select():
            st.session_state.feature_index = st.session_state.feature_index_select

        st.sidebar.selectbox(
            "Feature Index",
            options=available_features,
            index=list_index,
            key="feature_index_select",
            on_change=update_feature_index_from_select,  # session state callback
            help=f"Select number ({len(available_features)} available)",
        )

        # display current selection (checkpoint/feature)

        st.sidebar.info(
            f"✅ **Selected**:\n\n**checkpoint:** `{checkpoint_layer}`\n\n**feature #:** `{st.session_state.feature_index}`"
        )
