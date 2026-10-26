import streamlit as st
from src.frontend.components.footer import render_footer
from src.frontend.components.header import render_header
from src.frontend.components.modules import render_module_cards
from src.frontend.components.styles import add_base_styles, clear_css_cache
from src.frontend.config import (
    APP_ICON,
    APP_TITLE,
    DEFAULT_DOMAIN,
    DEFAULT_HPARAMSET_TAG,
    DEFAULT_LAYER,
    DEFAULT_MODEL_TAG,
    INITIAL_SIDEBAR_STATE,
    LAYOUT,
)

st.set_page_config(
    page_title=APP_TITLE.lower(),
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE,
)
add_base_styles()


def init_session_state():
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL_TAG
    if "layer" not in st.session_state:
        st.session_state.layer = DEFAULT_LAYER
    if "hyperparameter" not in st.session_state:
        st.session_state.hyperparameter = DEFAULT_HPARAMSET_TAG
    if "domain" not in st.session_state:
        st.session_state.domain = DEFAULT_DOMAIN
    if "feature_index" not in st.session_state:
        st.session_state.feature_index = 0


def main():
    clear_css_cache()  # TODO: delete this line

    init_session_state()
    render_header(
        title=APP_TITLE,
        icon=APP_ICON,
        description="Interpret Language Model using Sparse Autoencoder",
        gradient=True,
        add_floor=True,
    )

    # homepage content
    render_module_cards()

    render_footer(add_ceiling=False)


if __name__ == "__main__":
    main()
