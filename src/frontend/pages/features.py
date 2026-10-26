from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
from src.auto_explainer.explanations import FeatureExplanations
from src.features import (
    FeatureRecord,
    FeatureRecordWrapper,
    SampleRecord,
    load_sample_records,
    load_score_data,
)
from src.frontend.components.feature_navigator import render_feature_navigation
from src.frontend.components.feature_overview import render_feature_stats
from src.frontend.components.feature_samples import render_feature_samples
from src.frontend.components.footer import render_footer
from src.frontend.components.header import render_header
from src.frontend.components.setting_panel import render_settings_panel
from src.frontend.components.styles import add_base_styles, add_page_styles
from src.frontend.config import APP_ICON, APP_TITLE, LAYOUT, MODULE_MAP
from src.frontend.utils.session_fetcher import get_feature_index
from src.global_settings import settings

META_DIR = settings.feature_meta_dir
RECORD_DIR = settings.feature_record_dir
EXPLANATION_DIR = settings.feature_explanation_dir

module = MODULE_MAP["features"]
page_icon = module.get("icon", "🔬")
page_title = module.get("name", "Features")
page_desc = module.get("description", "")


st.set_page_config(
    page_title=f"{APP_TITLE.lower()}/features",
    page_icon=APP_ICON,
    layout=LAYOUT,
)
add_base_styles()
add_page_styles()


@st.cache_data(show_spinner=False)
def load_all_features_scores(checkpoint: str, layer: int) -> np.ndarray:
    score_matrix = load_score_data(checkpoint, layer, metric_name="total_post", meta_dir=Path(META_DIR))
    return score_matrix  # shape: (num_samples, num_features)


@st.cache_data(show_spinner=False)
def load_scores(checkpoint: str, layer: int, feature_index: int) -> np.ndarray:
    score_matrix = load_all_features_scores(checkpoint, layer)
    scores = score_matrix[:, feature_index]
    return scores  # shape: (num_samples,)


@st.cache_data(show_spinner=False)
def load_feature_data(
    model: str,
    hparam: str,
    layer: int,
    feature_index: int,
) -> FeatureRecordWrapper | None:
    """
    Load feature sample data for frontend display.

    Returns:
        FeatureRecordWrapper: wrapped FeatureRecord for frontend use(added filtering and stats)
    """
    checkpoint = f"{model}_{hparam}"

    try:
        scores = load_scores(checkpoint, layer, feature_index)

        positive_samples: List[SampleRecord] = load_sample_records(
            checkpoint=checkpoint,
            layer=layer,
            feature_index=feature_index,
            sample_type="positive",
            as_frame=False,
            record_dir=Path(RECORD_DIR),
        )
        negative_samples: List[SampleRecord] = load_sample_records(
            checkpoint=checkpoint,
            layer=layer,
            feature_index=feature_index,
            sample_type="negative",
            as_frame=False,
            record_dir=Path(RECORD_DIR),
        )
        feature_record = FeatureRecord(
            layer_index=layer,
            feature_index=feature_index,
            scores=scores,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
        )

        return FeatureRecordWrapper(feature_record=feature_record)

    except FileNotFoundError:
        print(f"Feature #{feature_index} of checkpoint {checkpoint}_L{layer} not found.")
        return None

    except Exception as e:
        return None


# ============================================================================
# Header
# ============================================================================
render_header(
    title=page_title,
    icon=page_icon,
    description=page_desc,
    gradient=False,
    add_floor=True,
)

# ============================================================================
# Settings Panel (left sidebar)
# ============================================================================
render_settings_panel(show_feature_selector=True)


# ============================================================================
# Main
# ============================================================================

# Fetch current settings
model = st.session_state.model
layer = st.session_state.layer
hparam = st.session_state.hyperparameter
feature_index = get_feature_index()


# Render feature navigator (utility)
render_feature_navigation(model, hparam, layer, feature_index)

# Load data
with st.spinner("Loading feature data..."):
    feature_data = load_feature_data(model, hparam, layer, feature_index)

    if feature_data is not None:
        # Show feature overview stats
        render_feature_stats(feature_data)

        # Show samples
        render_feature_samples(feature_data)

    else:
        st.info("⚠️ The feature file may not exist. Try selecting a different feature index.")


# ============================================================================
# Footer
# ============================================================================
render_footer(add_ceiling=False)
