import streamlit as st
from src.frontend.components.header import render_header
from src.frontend.components.styles import add_base_styles, add_page_styles
from src.frontend.config import APP_ICON, APP_TITLE, LAYOUT, MODULE_MAP

module = MODULE_MAP["about"]
page_icon = module.get("icon", "ℹ️")
page_title = module.get("name", "About")
page_desc = module.get("description", "")

st.set_page_config(
    page_title=f"{APP_TITLE.lower()}/about",
    page_icon=APP_ICON,
    layout=LAYOUT,
)
add_base_styles()
add_page_styles()


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
# Contents
# ============================================================================

## 🎯 What is SAE Explorer?
st.markdown('<h2 class="section-header">🎯 What is SAE Explorer?</h2>', unsafe_allow_html=True)
st.markdown(
    """SAE Explorer is an **interpretability tool** designed to help researchers and developers understand \
    how language models process and represent information internally."""
)
st.markdown("<br></br>", unsafe_allow_html=True)

## ✨ Key Features
st.markdown('<h2 class="section-header">✨ Key Features</h2>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
    <div class="feature-box">
        <h3>🔬 Feature Exploration</h3>
        <p>Browse and analyze thousands of interpretable features extracted from your language model.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
    <div class="feature-box">
        <h3>✍️ Interactive Testing</h3>
        <p>Test model behavior with custom text inputs and see which features activate in real-time.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
st.markdown("<br></br>", unsafe_allow_html=True)

## 🔧 How it Works
st.markdown('<h2 class="section-header">🔧 How It Works</h2>', unsafe_allow_html=True)
st.markdown(
    """
### Sparse Autoencoders (SAE)

Language models learn to represent information in <span class="highlight">dense, entangled representations</span> 
that are difficult to interpret. SAEs help us decompose these representations into 
<span class="highlight">sparse, interpretable features</span>.

""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="info-box">
<strong>💡 The SAE Process:</strong>
<ol>
    <li><strong>Extract</strong> activations from a specific layer of the language model</li>
    <li><strong>Train</strong> a sparse autoencoder to reconstruct these activations</li>
    <li><strong>Decode</strong> the learned features to understand what concepts they represent</li>
    <li><strong>Analyze</strong> when and how these features activate during inference</li>
</ol>
</div>
""",
    unsafe_allow_html=True,
)

## ⚙️ Technical Details
st.markdown('<h2 class="section-header">⚙️ Technical Details</h2>', unsafe_allow_html=True)
st.markdown(
    """
### Model Architecture

- **Base Model**: ModernBERT sequence classifier
- **SAE Configuration**: 
  - Features: 960 latent dimensions with k=8 sparsity
  - Training: Top-k activation with reconstruction objective
  - Layers: Analyzed across multiple transformer layers

### Evaluation Metrics

We evaluate our SAE using several metrics:
- **Reconstruction Loss**: How well the SAE reconstructs original activations
- **Sparsity**: Average number of active features per input
- **Feature Interpretability**: Human evaluation of decoded features
"""
)


## 📚 Getting Started
st.markdown('<h2 class="section-header">📚 Getting Started</h2>', unsafe_allow_html=True)
st.markdown(
    """
1. **Configure Settings**: Select your model, layer, and hyperparameter settings from the sidebar
2. **Explore Features**: Browse the feature catalog to understand what the model has learned
3. **Test Inference**: Input your own text to see which features activate
4. **Analyze Results**: Examine activation patterns and feature interactions

> 💡 **Tip**: Start with the Features page to get familiar with common feature types before testing custom inputs.
"""
)

## References & Links
st.markdown("---")
st.markdown(
    """
### 📖 References

- **Sparse Autoencoders**: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- **Mechanistic Interpretability**: [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html)

### 🔗 Links

- [GitHub Repository](#)
- [Documentation](#)
- [Paper](#)
"""
)

# Back to Home
if st.button("← Back to Home", type="secondary"):
    st.switch_page("main.py")


# ============================================================================
# Footer
# ============================================================================
st.markdown(
    """
    <div style='text-align: center; color: #a0aec0; padding: 3rem 0 1rem; margin-top: 4rem; border-top: 1px solid #e2e8f0;'>
        <p style='margin: 0;'>SAE Explorer v1.0 | Research interface</p>
    </div>
    """,
    unsafe_allow_html=True,
)
