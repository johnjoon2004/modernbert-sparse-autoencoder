import streamlit as st
from src.frontend.components.styles import add_footer_styles
from src.frontend.config import VERSION


def render_footer(add_ceiling: bool = True):
    add_footer_styles()

    if add_ceiling:
        st.markdown("---")

    st.markdown(
        f"""
        <div class='footer'>
            <p>SAE Explorer v{VERSION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
