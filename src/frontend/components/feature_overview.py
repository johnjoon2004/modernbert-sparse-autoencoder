import plotly.graph_objects as go
import streamlit as st
from src.features import FeatureRecordWrapper

COLOR_MAPS = {
    "pred": {
        "0": "rgba(144, 238, 144, 0.7)",  # green
        "1": "rgba(240, 128, 128, 0.7)",  # red
        "Human (0)": "rgba(144, 238, 144, 0.7)",
        "AI (1)": "rgba(240, 128, 128, 0.7)",
    },
    "label": {
        "0": "rgba(144, 238, 144, 0.7)",
        "1": "rgba(240, 128, 128, 0.7)",
        "Human (0)": "rgba(144, 238, 144, 0.7)",
        "AI (1)": "rgba(240, 128, 128, 0.7)",
    },
    "model": {
        "human": "rgba(144, 238, 144, 0.7)",
        "4o": "rgba(240, 128, 128, 0.7)",
        "gpt-3.5": "rgba(255, 182, 193, 0.7)",
        "DEFAULT": "rgba(255, 192, 203, 0.6)",
    },
    "prompt": {
        "human": "rgba(144, 238, 144, 0.7)",
        "fullwrite": "rgba(240, 128, 128, 0.7)",
        "zerowrite": "rgba(255, 182, 193, 0.7)",
        "DEFAULT": "rgba(255, 192, 203, 0.6)",
    },
}


def render_feature_stats(feature_data: FeatureRecordWrapper):
    st.markdown(
        f'<h2 class="section-header">📊 Feature Stats</h2>',
        unsafe_allow_html=True,
    )

    stats = feature_data.stats

    # Line 1
    cols = st.columns(3)
    with cols[0]:
        st.metric("Min / Max", f"{stats['min_score']:.2f} / {stats['max_score']:.2f}")
    with cols[1]:
        st.metric("Avg / Var", f"{stats['avg_score']:.2f} / {stats['variance']:.2f}")
    with cols[2]:
        # Active Rate: 얼마나 많은 샘플에서 해당 특징이 발현 되었는지
        ar_all = f'{stats["active_rate"]["all"]*100:.2f}%' or "-"
        ar_human = f'{stats["active_rate"]["human"]*100:.2f}%' or "-"
        ar_ai = f'{stats["active_rate"]["ai"]*100:.2f}%' or "-"
        st.metric("Active Rate (all/human/ai)", f"{ar_all} / {ar_human} / {ar_ai}")

    # Line 2
    cols = st.columns(3)
    with cols[0]:
        # AUROC: 특징의 score가 label=1과 label=0을 얼마나 잘 구분하는지
        st.metric("AUROC", f"{stats['auroc']:.2f}" if stats["auroc"] else "-")
    with cols[1]:
        # Pearson Corr: label(or pred)과 score 간의 상관관계
        corr = stats["pearson_corr"]
        corr_label = f"{corr["label"]:.2f}" if corr["label"] is not None else "-"
        corr_pred = f"{corr['pred']:.2f}" if corr["pred"] is not None else "-"
        st.metric("Pearson Corr (label/pred)", f"{corr_label} / {corr_pred}")
    with cols[2]:
        # Cohen's D: label=0 / label=1 (or pred) 두 그룹의 score가 얼마나 차이나는지
        cohens_d = stats["cohens_d"]
        cohens_d_label = f"{cohens_d['label']:.2f}" if cohens_d["label"] is not None else "-"
        cohens_d_pred = f"{cohens_d['pred']:.2f}" if cohens_d["pred"] is not None else "-"
        st.metric("Cohen's D (Label/Pred)", f"{cohens_d_label} / {cohens_d_pred}")

    '''
    with cols[-1]:
        """정규성 Moments
        Skewness: 왜도 (분포의 비대칭성, >0: 오른쪽 꼬리, <0: 왼쪽 꼬리)
        Kurtosis: 첨도 (분포의 뾰족함, >0: 뾰족, <0: 평평)
        """
        skewness = f"{stats['skewness']:.2f}" if stats["skewness"] else "-"
        kurtosis = f"{stats['kurtosis']:.2f}" if stats["kurtosis"] else "-"
        st.metric("Skewness / Kurtosis", f"{skewness} / {kurtosis}")
    '''

    cols = st.columns([2, 3, 2, 3, 2])
    with cols[1]:
        # Active Matrix: 상황별((label=0, pred=0), (label=0, pred=1), ...) 해당 특징이 발현(score > 0)되는 비율
        cm_df = stats["active_matrix"]
        st.metric("Active Matrix", "")
        st.dataframe(cm_df.style.format("{:.2f}").background_gradient(cmap="Blues"), width="stretch")
    with cols[3]:
        # Score Matrix: 상황별((label=0, pred=0), (label=0, pred=1), ...) 해당 특징이 갖는 점수의 평균 값 (*발현 샘플 only)
        cm_df = stats["score_matrix"]
        st.metric("Score Matrix", "")
        st.dataframe(cm_df.style.format("{:.2f}").background_gradient(cmap="Greens"), width="stretch")

    # ============================================================
    # Score distribution
    # ============================================================

    # Attribution_key selection UI
    attr_key = st.selectbox(
        "Compare score distribution by:",
        options=["pred", "label", "model", "prompt"],
        index=0,  # default to 'pred'
        key="score_dist_attr_key",
        help="Show score distributions split by the selected attribute.",
    )

    comp_data = stats.get("score_dist_data_attrs", {}).get(attr_key)

    if comp_data and comp_data.get("bins") is not None and comp_data.get("groups"):
        bin_edges = comp_data["bins"]
        groups = comp_data["groups"]

        x_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)]

        fig = go.Figure()

        color_map = COLOR_MAPS.get(attr_key, {})
        default_color = color_map.get("DEFAULT", "rgba(128, 128, 128, 0.7)")

        for group_name, counts in groups.items():
            if attr_key in ["label", "pred"]:
                name = f"{'AI' if group_name == '1' else 'Human'} ({group_name})"
            else:
                name = group_name

            group_color = color_map.get(str(group_name), default_color)

            fig.add_trace(
                go.Bar(
                    name=name,
                    x=x_labels,
                    y=counts,
                    marker_color=group_color,
                )
            )

        fig.update_layout(
            barmode="overlay",
            title=f"Score Distribution by '{attr_key}'",
            xaxis_title="Score Range",
            xaxis_tickangle=-45,
            yaxis_title="Count",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            legend_title_text=attr_key.capitalize(),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # fallback
        score_data = stats.get("score_dist_data")
        if score_data and score_data.get("bins") is not None and score_data.get("counts") is not None:
            bin_edges = score_data["bins"]
            counts = score_data["counts"]
            x_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(counts))]

            fig = go.Figure(data=[go.Bar(x=x_labels, y=counts)])
            fig.update_layout(
                title=f"Score Distribution",
                xaxis_title="Score Range",
                xaxis_tickangle=-45,
                yaxis_title="Count",
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # Other charts row
    # ============================================================

    keys = ["label", "pred", "model", "prompt"]

    chart_cols = st.columns(len(keys))

    for i, key in enumerate(keys):
        dist = stats[f"{key}_dist"]
        color_map = COLOR_MAPS.get(key, {})

        with chart_cols[i]:
            if key == "model":
                labels = list(dist.keys())
                values = list(dist.values())
                bar_colors = [color_map.get(k, color_map.get("DEFAULT")) for k in labels]
                fig = go.Figure(data=[go.Bar(x=labels, y=values, marker=dict(color=bar_colors))])
            else:
                labels = ["Human (0)", "AI (1)"]
                values = [dist.get(0, 0), dist.get(1, 0)]
                pie_colors = [color_map.get("0"), color_map.get("1")]
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, marker=dict(colors=pie_colors))])

            fig.update_layout(
                title=f"{key}".capitalize(),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
