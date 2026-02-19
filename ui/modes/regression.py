"""
Drift Auditor — Regression Analysis Mode
==========================================
Statistical patterns across batch-audited conversations.
Scatter plots: conversation length vs OLI, instruction count vs drift,
model vs correction failure rate.
"""
import os
import json
import glob as glob_mod

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from ui.theme import THEMES
from ui.components import _racing_stickmen_html, chart_export_png


def render_regression_mode() -> None:
    """Render the Regression Analysis mode UI."""
    T = THEMES[st.session_state.get("theme_name", "Ember")]
    PLOTLY_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["chart_text"], family="JetBrains Mono, DM Sans, sans-serif", size=12),
        margin=dict(l=50, r=30, t=40, b=40),
    )

    st.markdown("## 📊 Regression Analysis")
    st.caption("Statistical patterns across 512 audited conversations — conversation length vs OLI, "
               "instruction count vs drift, model vs correction failure rate.")

    _reg_race = st.empty()
    _reg_race.markdown(_racing_stickmen_html(T), unsafe_allow_html=True)

    # Load batch data
    batch_data = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(base_dir, "..", "..")
    for batch_dir in ["batch_results", "batch_results_chatgpt"]:
        batch_path = os.path.join(repo_root, batch_dir)
        json_files = glob_mod.glob(os.path.join(batch_path, "batch_results*.json"))
        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    results = json.load(f)
                for r in results:
                    if "model" not in r:
                        r["model"] = "claude" if "uuid" in r else "unknown"
                    batch_data.append(r)
            except Exception:
                pass

    _reg_race.empty()

    if not batch_data:
        st.warning("No batch results found. Run `batch_audit.py` and/or `batch_audit_chatgpt.py` first.")
        return

    df = pd.DataFrame(batch_data)
    df["commission_flags"] = df["commission_flags"].fillna(0).astype(int)
    df["omission_flags"] = df["omission_flags"].fillna(0).astype(int)
    df["corrections_failed"] = df.get("corrections_failed", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["corrections_total"] = df.get("corrections_total", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["total_flags"] = df["commission_flags"] + df["omission_flags"]
    df["correction_fail_rate"] = df["corrections_failed"] / df["corrections_total"].clip(lower=1)

    st.success(f"Loaded **{len(df)} conversations** across **{df['model'].nunique()} models**: "
               f"{', '.join(df['model'].unique())}")

    # Plot 1: Conversation Length vs Drift Score
    st.markdown("### 1. Conversation Length → Drift Score")
    st.caption("Do longer conversations drift more?")
    fig1 = go.Figure()
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        fig1.add_trace(go.Scatter(
            x=mdf["message_count"], y=mdf["overall_score"],
            mode="markers", name=model,
            marker=dict(size=6, opacity=0.6),
            hovertext=[f"{model} | {r.get('name', r.get('title', ''))[:30]}... | "
                        f"{r['message_count']} msgs, drift {r['overall_score']}"
                        for _, r in mdf.iterrows()],
            hoverinfo="text",
        ))
    x_all = df["message_count"].values.astype(float)
    y_all = df["overall_score"].values.astype(float)
    mask = ~(np.isnan(x_all) | np.isnan(y_all))
    if mask.sum() > 2:
        z = np.polyfit(x_all[mask], y_all[mask], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x_all[mask].min(), x_all[mask].max(), 100)
        fig1.add_trace(go.Scatter(
            x=x_trend, y=p(x_trend), mode="lines",
            name=f"Trend (slope={z[0]:.4f})",
            line=dict(color="#f59e0b", width=3, dash="dash"),
        ))
        corr = np.corrcoef(x_all[mask], y_all[mask])[0, 1]
        st.markdown(f"**Pearson r = {corr:.3f}** — "
                    f"{'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'} "
                    f"{'positive' if corr > 0 else 'negative'} correlation")
    fig1.update_layout(**PLOTLY_LAYOUT, height=450,
        xaxis=dict(title="Message Count", gridcolor="#2a2623", zeroline=False),
        yaxis=dict(title="Overall Drift Score (1-10)", range=[0, 11], gridcolor="#2a2623", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})
    chart_export_png(fig1, "regression_length_vs_drift.png", "Download Scatter PNG")

    # Plot 2: Instruction Count vs Drift Score
    st.markdown("### 2. Instruction Count → Drift Score")
    st.caption("Does instruction complexity predict drift?")
    fig2 = go.Figure()
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        fig2.add_trace(go.Scatter(
            x=mdf["instructions_extracted"], y=mdf["overall_score"],
            mode="markers", name=model, marker=dict(size=6, opacity=0.6),
        ))
    x2 = df["instructions_extracted"].values.astype(float)
    y2 = df["overall_score"].values.astype(float)
    mask2 = ~(np.isnan(x2) | np.isnan(y2))
    if mask2.sum() > 2:
        z2 = np.polyfit(x2[mask2], y2[mask2], 1)
        p2 = np.poly1d(z2)
        x2_trend = np.linspace(x2[mask2].min(), x2[mask2].max(), 100)
        fig2.add_trace(go.Scatter(
            x=x2_trend, y=p2(x2_trend), mode="lines",
            name=f"Trend (slope={z2[0]:.4f})",
            line=dict(color="#f59e0b", width=3, dash="dash"),
        ))
        corr2 = np.corrcoef(x2[mask2], y2[mask2])[0, 1]
        st.markdown(f"**Pearson r = {corr2:.3f}** — "
                    f"{'strong' if abs(corr2) > 0.5 else 'moderate' if abs(corr2) > 0.3 else 'weak'} "
                    f"{'positive' if corr2 > 0 else 'negative'} correlation")
    fig2.update_layout(**PLOTLY_LAYOUT, height=450,
        xaxis=dict(title="Instructions Extracted", gridcolor="#2a2623", zeroline=False),
        yaxis=dict(title="Overall Drift Score (1-10)", range=[0, 11], gridcolor="#2a2623", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
    chart_export_png(fig2, "regression_instructions_vs_drift.png", "Download Scatter PNG")

    # Plot 3: Model Type vs Correction Failure Rate
    st.markdown("### 3. Model Type → Correction Failure Rate")
    st.caption("Which model fails to hold corrections more often?")
    model_cfr = df.groupby("model").agg(
        avg_fail_rate=("correction_fail_rate", "mean"),
        median_fail_rate=("correction_fail_rate", "median"),
        conversations=("model", "count"),
        avg_drift=("overall_score", "mean"),
    ).reset_index()
    fig3 = go.Figure()
    bar_colors = ["#f59e0b", "#22c55e", "#a855f7", "#3b82f6", "#ef4444"]
    for i, (_, row) in enumerate(model_cfr.iterrows()):
        color = bar_colors[i % len(bar_colors)]
        fig3.add_trace(go.Bar(
            x=[row["model"]], y=[row["avg_fail_rate"]],
            name=row["model"], marker_color=color,
            text=[f"{row['avg_fail_rate']:.1%}"], textposition="outside",
            textfont=dict(color="#e8dfd0", size=14),
            hovertext=f"{row['model']}: {row['avg_fail_rate']:.1%} avg failure rate<br>"
                      f"Median: {row['median_fail_rate']:.1%}<br>"
                      f"{int(row['conversations'])} conversations<br>"
                      f"Avg drift: {row['avg_drift']:.1f}/10",
            hoverinfo="text",
        ))
    fig3.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False,
        xaxis=dict(title="Model", gridcolor="#2a2623"),
        yaxis=dict(title="Avg Correction Failure Rate", gridcolor="#2a2623",
                   tickformat=".0%", zeroline=False),
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displaylogo": False})
    chart_export_png(fig3, "regression_model_correction_rate.png", "Download Bar Chart PNG")

    # Plot 4: Conversation Length vs Total Flags
    st.markdown("### 4. Conversation Length → Total Flags")
    st.caption("Flag accumulation rate across conversation length.")
    fig4 = go.Figure()
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        fig4.add_trace(go.Scatter(
            x=mdf["message_count"], y=mdf["total_flags"],
            mode="markers", name=model, marker=dict(size=6, opacity=0.6),
        ))
    x4 = df["message_count"].values.astype(float)
    y4 = df["total_flags"].values.astype(float)
    mask4 = ~(np.isnan(x4) | np.isnan(y4))
    if mask4.sum() > 2:
        z4 = np.polyfit(x4[mask4], y4[mask4], 1)
        p4 = np.poly1d(z4)
        x4_trend = np.linspace(x4[mask4].min(), x4[mask4].max(), 100)
        fig4.add_trace(go.Scatter(
            x=x4_trend, y=p4(x4_trend), mode="lines",
            name=f"Trend ({z4[0]:.2f} flags/msg)",
            line=dict(color="#f59e0b", width=3, dash="dash"),
        ))
        corr4 = np.corrcoef(x4[mask4], y4[mask4])[0, 1]
        st.markdown(f"**Pearson r = {corr4:.3f}** — "
                    f"{'strong' if abs(corr4) > 0.5 else 'moderate' if abs(corr4) > 0.3 else 'weak'} "
                    f"{'positive' if corr4 > 0 else 'negative'} correlation. "
                    f"Slope: **{z4[0]:.2f} flags per message** on average.")
    fig4.update_layout(**PLOTLY_LAYOUT, height=450,
        xaxis=dict(title="Message Count", gridcolor="#2a2623", zeroline=False),
        yaxis=dict(title="Total Flags (Commission + Omission)", gridcolor="#2a2623", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displaylogo": False})
    chart_export_png(fig4, "regression_length_vs_flags.png", "Download Scatter PNG")

    # Summary Stats Table
    st.markdown("### Summary Statistics by Model")
    summary = df.groupby("model").agg(
        conversations=("model", "count"),
        avg_messages=("message_count", "mean"),
        avg_drift=("overall_score", "mean"),
        avg_instructions=("instructions_extracted", "mean"),
        avg_flags=("total_flags", "mean"),
        avg_corrections=("corrections_total", "mean"),
        avg_fail_rate=("correction_fail_rate", "mean"),
        avg_void_events=("void_events", "mean"),
    ).reset_index()
    summary.columns = ["Model", "Conversations", "Avg Messages", "Avg Drift Score",
                        "Avg Instructions", "Avg Flags", "Avg Corrections",
                        "Avg Correction Fail Rate", "Avg Void Events"]
    for col in summary.columns[2:]:
        summary[col] = summary[col].round(3)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Key Findings
    st.markdown("### Key Findings")
    findings = []
    if mask.sum() > 2:
        corr_val = np.corrcoef(x_all[mask], y_all[mask])[0, 1]
        if abs(corr_val) > 0.3:
            findings.append(f"• **Conversation length and drift are correlated** (r={corr_val:.3f}). "
                            f"Longer conversations {'accumulate more drift' if corr_val > 0 else 'show less drift'}.")
        else:
            findings.append(f"• Conversation length has **weak correlation** with drift (r={corr_val:.3f}). "
                            f"Drift emerges regardless of conversation length.")
    if mask2.sum() > 2:
        corr_val2 = np.corrcoef(x2[mask2], y2[mask2])[0, 1]
        if abs(corr_val2) > 0.3:
            findings.append(f"• **Instruction complexity correlates with drift** (r={corr_val2:.3f}). "
                            f"More instructions → {'more drift' if corr_val2 > 0 else 'less drift'}.")
    if len(model_cfr) > 1:
        best = model_cfr.loc[model_cfr["avg_fail_rate"].idxmin()]
        worst = model_cfr.loc[model_cfr["avg_fail_rate"].idxmax()]
        findings.append(f"• **{worst['model']}** has the highest correction failure rate "
                        f"({worst['avg_fail_rate']:.1%}), vs **{best['model']}** ({best['avg_fail_rate']:.1%}).")
    if findings:
        for finding in findings:
            st.markdown(finding)
    else:
        st.info("Insufficient data to draw conclusions. Add more batch results.")
