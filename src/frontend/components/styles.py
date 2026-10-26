from functools import lru_cache
from pathlib import Path

import streamlit as st


@lru_cache(maxsize=None)
def load_css(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"CSS file not found: {file_path}")
        return ""


def _apply_css(css_content: str):
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def apply_css(file_path: Path):
    css_content = load_css(file_path)
    if css_content:
        _apply_css(css_content)


CSS_DIR = Path(__file__).parent / "css"
BASE_CSS = CSS_DIR / "base.css"
HEADER_CSS = CSS_DIR / "header.css"
FOOTER_CSS = CSS_DIR / "footer.css"
HOMEPAGE_CSS = CSS_DIR / "homepage.css"
PAGES_CSS = CSS_DIR / "pages.css"


def add_base_styles():
    apply_css(BASE_CSS)


def add_header_styles():
    apply_css(HEADER_CSS)


def add_footer_styles():
    apply_css(FOOTER_CSS)


def add_homepage_styles():
    apply_css(HOMEPAGE_CSS)


def add_page_styles():
    apply_css(PAGES_CSS)


def clear_css_cache():
    load_css.cache_clear()
