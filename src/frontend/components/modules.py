"""
Module-cards component in homepage (main.py).

In v1.0, there are 4 top-level modules:
1. About
2. Features
3. Inference
4. Steering
# 5. Experiments (deleted)
"""

import streamlit as st
from src.frontend.components.styles import add_homepage_styles
from src.frontend.config import TOP_MODULES


def render_module_card(module: dict):
    disabled = module.get("disabled", False)

    if disabled:
        # unclickable
        card_html = f"""
        <div class="module-card disabled" style="background-color: {module['color']};">
            <div class="card-content">
                <div class="card-icon">{module['icon']}</div>
                <h3 class="card-title">{module['name']}</h3>
                <p class="card-description">{module['description']}</p>
                <div class="coming-soon-badge">Coming Soon</div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    else:
        # pop-up a new page when clicking an active card
        card_html = f"""
        <a href="?page={module['id']}" class="module-card" style="background-color: {module['color']};">
            <div class="card-content">
                <div class="card-icon">{module['icon']}</div>
                <h3 class="card-title">{module['name']}</h3>
                <p class="card-description">{module['description']}</p>
            </div>
        </a>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        query_params = st.query_params
        if "page" in query_params and query_params["page"] == module["id"]:
            st.switch_page(f"pages/{module['id']}.py")


def render_module_cards():
    add_homepage_styles()

    COLUMNS_PER_ROW = 2
    total_modules = len(TOP_MODULES)
    num_rows = (total_modules + COLUMNS_PER_ROW - 1) // COLUMNS_PER_ROW

    for row in range(num_rows):
        start_idx = row * COLUMNS_PER_ROW
        end_idx = min(start_idx + COLUMNS_PER_ROW, total_modules)
        row_modules = TOP_MODULES[start_idx:end_idx]
        num_modules_in_row = len(row_modules)

        if num_modules_in_row < COLUMNS_PER_ROW:
            empty_cols = COLUMNS_PER_ROW - num_modules_in_row
            col_ratios = [1] * empty_cols + [2] * num_modules_in_row + [1] * empty_cols
            cols = st.columns(col_ratios, gap="large")

            for i in range(empty_cols):
                with cols[i]:
                    st.empty()

            for i, module in enumerate(row_modules):
                with cols[empty_cols + i]:
                    render_module_card(module)

            for i in range(empty_cols + num_modules_in_row, len(cols)):
                with cols[i]:
                    st.empty()

        else:
            cols = st.columns(COLUMNS_PER_ROW, gap="large")
            for i, module in enumerate(row_modules):
                with cols[i]:
                    render_module_card(module)

        if row < num_rows - 1:
            st.markdown("<div style='margin: 4.5rem 0;'></div>", unsafe_allow_html=True)
