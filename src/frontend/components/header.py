from typing import Optional

import streamlit as st
from src.frontend.components.styles import add_header_styles


def render_header(
    title: str,
    icon: Optional[str] = None,
    description: Optional[str] = None,
    gradient: bool = False,
    add_floor: bool = True,
):
    add_header_styles()
    header_html = '<div class="page-header">'
    title_class = "gradient-text header-title-gradient" if gradient else "header-title-plain"

    if icon:
        header_html += f"""
            <h1 class="header-with-icon">
                <span class="header-icon">{icon}</span>
                <span class="{title_class} header-title">{title}</span>
            </h1>
        """
    else:
        header_html += f'<h1 class="{title_class}">{title}</h1>'

    if description:
        header_html += f'<div class="header-description">{description}</div>'

    header_html += "</div>"
    st.markdown(header_html, unsafe_allow_html=True)

    if add_floor:
        st.markdown("---")
