from dataclasses import asdict
from typing import Dict, List, Literal

import streamlit as st
from src.features import FeatureRecordWrapper, SampleRecord
from src.global_settings import settings
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(settings.lm_weights_path)


def _highlight_text(
    sample: Dict, sample_type: Literal["positive", "negative"] = "positive", color_map: Dict = None
) -> str:

    # TODO: Add token highlighting based on the feature score distribution

    text = sample.get("text")
    active_tokens = sample.get("active_tokens", [])
    active_token_map = {t["token_idx"]: t["token_score"] for t in active_tokens if t.get("token_score") is not None}

    if not active_token_map:
        return f"<div>{text}</div>"

    all_tokens = tokenizer.tokenize(text, truncation=True, max_length=settings.prev_seq_len)

    # HACK: temporary by min-max
    min_score = min(active_token_map.values())
    max_score = max(active_token_map.values())
    score_range = max_score - min_score if max_score != min_score else 1.0

    highlighted_tokens = []
    for idx_, token in enumerate(all_tokens):
        idx = idx_ + 1  # without cls
        if idx == 1:
            prefix = ""
            _token = token
        elif token.startswith("##"):
            prefix = ""
            _token = token[2:]
        else:
            prefix = " "
            _token = token

        if idx in active_token_map:
            score = active_token_map[idx]
            intensity = (score - min_score) / score_range
            color = f"rgba(255, {255 - int(intensity * 100)}, 0, {0.3 + intensity * 0.7})"
            token_html = (
                f'<span style="background-color: {color}; padding: 1px 1.5px; '
                f'border-radius: 3px; cursor: help;" '
                f'title="Token: {token}\nScore: {score:.2f}">{_token}</span>'
            )
        else:
            token_html = f"<span>{_token}</span>"

        highlighted_tokens.append(f"<span>{prefix}</span>")
        highlighted_tokens.append(token_html)

    html = (
        '<div style="line-height:1.8; font-size:1.05rem; font-family:inherit; '
        'overflow-wrap:break-word; word-break:break-all;">'
        f"{"".join(highlighted_tokens)}</div>"
    )
    return html


def _render_sample_card(
    sample: Dict, sample_type: Literal["positive", "negative"] = "positive", color_map: Dict = None
):
    meta_cols = st.columns(7)
    with meta_cols[0]:
        st.metric("id", sample["text_idx"])
    with meta_cols[1]:
        st.metric("label", sample["label"])
    with meta_cols[2]:
        st.metric("pred", sample["pred"])
    with meta_cols[3]:
        st.metric("model", sample["model"])
    with meta_cols[4]:
        st.metric("prompt", sample["prompt"])
    with meta_cols[5]:
        st.metric("score", sample["total_score_post"])
    with meta_cols[6]:
        st.metric("score (avg)", sample["avg_score_pre"])

    if sample.get("text"):
        # st.markdown("**text:**")
        st.markdown(f"> {sample['text']}", unsafe_allow_html=True)

        highlighted_text = _highlight_text(sample, sample_type=sample_type, color_map=color_map)
        st.markdown(highlighted_text, unsafe_allow_html=True)
    else:
        st.markdown("> No text available for this sample.")

    with st.expander("View Token Scores", expanded=False):
        reverse = sample_type == "positive"
        if sample.get("active_tokens"):
            # tokens = sorted(sample["active_tokens"], key=lambda x: x.get("token_score"), reverse=reverse)
            # for token in tokens:
            #     st.text(f"{str(token["token_idx"]):20s} | {token["token_str"]:20s} | {token["token_score"]:.2f}")
            # st.info(f"[Debug]\n\n tokens = {tokens}")

            tokens = sorted(sample["active_tokens"], key=lambda x: x.get("token_idx"))
            st.json(tokens)

        else:
            st.info("No active tokens for this sample.")


def _render_sample_cards(samples: List[Dict] | List[SampleRecord], sample_type=Literal["positive", "negative"]):
    if isinstance(samples[0], SampleRecord):
        samples = [asdict(sample) for sample in samples]

    if not samples:
        st.warning("No samples are available.")
        return

    if sample_type == "positive":
        score_key = "total_score_post"
        valid_samples = [sample for sample in samples if sample.get(score_key) > 0]
    else:  # negative
        score_key = "avg_score_pre"
        valid_samples = [sample for sample in samples if sample.get(score_key) < 0]

    for sample in valid_samples:
        # is_correct = sample["pred"] == sample["label"]
        # icon_pred = "✅" if is_correct else "❌"
        label = f"label: {'🤖' if sample['label'] == 1 else '👤'}"
        pred = f"pred: {'🤖' if sample['pred'] == 1 else '👤'}"
        expander = (
            f"text_idx: `{sample['text_idx']}` | "
            f"score: `{sample[score_key]:.2f}` | "
            f"{label} | {pred} | "
            f"model: `{sample['model']}` | "
            f"prompt: `{sample['prompt']}` "
        )

        with st.expander(expander, expanded=False):
            # st.markdown(expander, unsafe_allow_html=True)
            _render_sample_card(sample, color_map=None)


def render_feature_samples(feature_data: FeatureRecordWrapper):
    st.markdown("---")
    st.markdown('<h2 class="section-header">📝 Samples </h2>', unsafe_allow_html=True)

    st.markdown('<h3 class="subsection-header"> 🔥 Positive Samples </h3>', unsafe_allow_html=True)
    _render_sample_cards(feature_data.positive_samples, sample_type="positive")

    st.markdown("<br></br>", unsafe_allow_html=True)

    st.markdown('<h3 class="subsection-header"> ❄️ Negative Samples </h3>', unsafe_allow_html=True)
    _render_sample_cards(feature_data.negative_samples, sample_type="negative")
