import streamlit as st


def get_checkpoint() -> str:
    """Get current checkpoint. (model/hparam/layer)"""
    model = st.session_state.get("model", "f3072_k8")
    hparam = st.session_state.get("hyperparameter", "alpha")
    layer = st.session_state.get("layer", 21)
    return f"{model}_{hparam}_L{layer}"


def get_ckpt() -> str:
    model = st.session_state.get("model", "f3072_k8")
    hparam = st.session_state.get("hyperparameter", "alpha")
    return f"{model}_{hparam}"


def get_feature_index() -> int:
    """Get current feature index."""
    return st.session_state.get("feature_index", 0)
