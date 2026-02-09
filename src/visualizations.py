"""
Drift Auditor — Visualization Module
=====================================
Extracted from app.py to improve readability and maintainability.

Contains:
- score_color / score_label: severity mapping helpers
- render_metric_card: glass metric card for Streamlit
- build_timeline_fig: scatter plot of all drift flags
- build_barometer_strip / build_barometer_detail: barometer visualizations
- build_persistence_fig: correction persistence flow
- build_commission_fig / build_omission_fig: layer-specific charts
- operator_load_chart: Claude vs ChatGPT operator load comparison

Author: George Abrahamyants + Claude (Opus 4.6)
"""

import streamlit as st
import plotly.graph_objects as go

# Import data types for type hints
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from drift_auditor import AuditReport


# ---------------------------------------------------------------------------
# Shared layout config
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c4b8a3", family="JetBrains Mono, DM Sans, sans-serif", size=12),
    margin=dict(l=50, r=30, t=40, b=40),
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def score_color(score: int) -> str:
    """Map a 1-10 severity score to a hex color (warm palette)."""
    if score <= 3:
        return "#22c55e"   # green — clean
    elif score <= 5:
        return "#f59e0b"   # amber — moderate
    elif score <= 7:
        return "#ef4444"   # red — elevated
    else:
        return "#dc2626"   # deep red — severe


def score_label(score: int) -> str:
    """Human-readable label for a 1-10 score."""
    if score <= 2:
        return "Clean"
    elif score <= 4:
        return "Low Drift"
    elif score <= 6:
        return "Moderate"
    elif score <= 8:
        return "Elevated"
    else:
        return "Severe"


def render_metric_card(label: str, score: int, subtitle: str, hero: bool = False) -> None:
    """Render a glass metric card with a colored score value."""
    color = score_color(score)
    card_cls = "glass-card-hero" if hero else "glass-card"
    val_cls = "metric-value-hero" if hero else "metric-value"
    shadow = f"0 0 20px {color}25" if hero else "none"
    st.markdown(f"""
    <div class="{card_cls}" style="box-shadow: {shadow}">
        <div class="metric-label">{label}</div>
        <div class="{val_cls}" style="color: {color}">{score}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Timeline & Barometer
# ---------------------------------------------------------------------------

def build_timeline_fig(report: AuditReport) -> go.Figure:
    """Scatter plot of all drift flags across conversation turns."""
    fig = go.Figure()

    # Commission flags
    if report.commission_flags:
        fig.add_trace(go.Scatter(
            x=[f.turn for f in report.commission_flags],
            y=[f.severity for f in report.commission_flags],
            mode="markers",
            name="Commission",
            marker=dict(size=11, color="#ef4444", symbol="diamond",
                        line=dict(width=1, color="#ef444480")),
            hovertext=[f"T{f.turn}: {f.description[:60]}" for f in report.commission_flags],
            hoverinfo="text",
        ))

    # Omission flags
    if report.omission_flags:
        fig.add_trace(go.Scatter(
            x=[f.turn for f in report.omission_flags],
            y=[f.severity for f in report.omission_flags],
            mode="markers",
            name="Omission",
            marker=dict(size=11, color="#f59e0b", symbol="triangle-up",
                        line=dict(width=1, color="#f59e0b80")),
            hovertext=[f"T{f.turn}: {f.description[:60]}" for f in report.omission_flags],
            hoverinfo="text",
        ))

    # Correction failures
    failed = [e for e in report.correction_events if not e.held and e.failure_turn is not None]
    if failed:
        fig.add_trace(go.Scatter(
            x=[e.failure_turn for e in failed],
            y=[8] * len(failed),
            mode="markers",
            name="Correction Failed",
            marker=dict(size=14, color="#a855f7", symbol="x",
                        line=dict(width=2, color="#a855f7")),
            hovertext=[f"Correction failed at T{e.failure_turn}" for e in failed],
            hoverinfo="text",
        ))

    # Barometer RED
    red_signals = [s for s in report.barometer_signals if s.classification == "RED"]
    if red_signals:
        fig.add_trace(go.Scatter(
            x=[s.turn for s in red_signals],
            y=[s.severity for s in red_signals],
            mode="markers",
            name="Barometer RED",
            marker=dict(size=9, color="#ef4444", symbol="circle", opacity=0.4),
            hovertext=[f"T{s.turn}: {s.description[:60]}" for s in red_signals],
            hoverinfo="text",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        xaxis=dict(title="Turn", gridcolor="#2a2623", zeroline=False),
        yaxis=dict(title="Severity", range=[0, 11], gridcolor="#2a2623", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="closest",
    )
    return fig


def build_barometer_strip(report: AuditReport) -> go.Figure:
    """Thin heatmap strip showing barometer classification per assistant turn."""
    if not report.barometer_signals:
        return go.Figure()

    colors = {"GREEN": "#22c55e", "YELLOW": "#f59e0b", "RED": "#ef4444"}

    fig = go.Figure()
    for s in report.barometer_signals:
        fig.add_trace(go.Bar(
            x=[1], y=["Epistemic Posture"],
            orientation="h",
            marker_color=colors.get(s.classification, "#f59e0b"),
            hovertext=f"T{s.turn}: {s.classification} — {s.description[:50]}",
            hoverinfo="text",
            showlegend=False,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        height=60,
        margin=dict(l=120, r=30, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def build_barometer_detail(report: AuditReport) -> go.Figure:
    """Per-turn bar chart of barometer signals colored by classification."""
    if not report.barometer_signals:
        return go.Figure()

    colors = {"GREEN": "#22c55e", "YELLOW": "#f59e0b", "RED": "#ef4444"}
    fig = go.Figure()

    for cls in ["GREEN", "YELLOW", "RED"]:
        filtered = [s for s in report.barometer_signals if s.classification == cls]
        if filtered:
            fig.add_trace(go.Bar(
                x=[s.turn for s in filtered],
                y=[s.severity for s in filtered],
                name=cls,
                marker_color=colors[cls],
                hovertext=[f"{s.description[:60]}" for s in filtered],
                hoverinfo="text",
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        height=320,
        xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Severity", gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Correction Persistence
# ---------------------------------------------------------------------------

def build_persistence_fig(report: AuditReport) -> go.Figure:
    """Correction -> acknowledgment -> failure/held flow visualization."""
    events = report.correction_events
    if not events:
        return go.Figure()

    fig = go.Figure()
    for i, event in enumerate(events):
        y_pos = len(events) - i
        held = event.held

        # Correction point
        fig.add_trace(go.Scatter(
            x=[event.correction_turn], y=[y_pos],
            mode="markers+text", text=["Correction"],
            textposition="top center",
            textfont=dict(size=10, color="rgba(255,255,255,0.6)"),
            marker=dict(size=14, color="#d68910", symbol="circle"),
            showlegend=False, hoverinfo="text",
            hovertext=f"User corrected at T{event.correction_turn}",
        ))

        # Acknowledgment point
        fig.add_trace(go.Scatter(
            x=[event.acknowledgment_turn], y=[y_pos],
            mode="markers+text", text=["Ack"],
            textposition="top center",
            textfont=dict(size=10, color="rgba(255,255,255,0.6)"),
            marker=dict(size=14, color="#22c55e", symbol="circle"),
            showlegend=False, hoverinfo="text",
            hovertext=f"Model acknowledged at T{event.acknowledgment_turn}",
        ))

        # Connector: correction -> ack
        fig.add_trace(go.Scatter(
            x=[event.correction_turn, event.acknowledgment_turn],
            y=[y_pos, y_pos],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.2)", width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

        # Outcome
        if not held and event.failure_turn is not None:
            fig.add_trace(go.Scatter(
                x=[event.failure_turn], y=[y_pos],
                mode="markers+text", text=["Failed"],
                textposition="top center", textfont=dict(size=10, color="#ef4444"),
                marker=dict(size=16, color="#ef4444", symbol="x"),
                showlegend=False, hoverinfo="text",
                hovertext=f"Correction failed at T{event.failure_turn}",
            ))
            fig.add_trace(go.Scatter(
                x=[event.acknowledgment_turn, event.failure_turn],
                y=[y_pos, y_pos],
                mode="lines",
                line=dict(color="#ef4444", width=2, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
        else:
            # Held indicator
            fig.add_trace(go.Scatter(
                x=[event.acknowledgment_turn + 2], y=[y_pos],
                mode="markers+text", text=["Held \u2713"],
                textposition="middle right",
                textfont=dict(size=11, color="#22c55e"),
                marker=dict(size=12, color="#22c55e", symbol="circle"),
                showlegend=False, hoverinfo="text",
                hovertext="Correction held across subsequent turns",
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(180, len(events) * 80 + 60),
        xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Layer-specific charts
# ---------------------------------------------------------------------------

def build_commission_fig(report: AuditReport) -> go.Figure:
    """Horizontal bar chart of commission flags sorted by severity."""
    flags = sorted(report.commission_flags, key=lambda f: f.severity, reverse=True)
    if not flags:
        return go.Figure()

    fig = go.Figure(go.Bar(
        y=[f"T{f.turn}: {f.description[:35]}" for f in flags],
        x=[f.severity for f in flags],
        orientation="h",
        marker_color=[score_color(f.severity) for f in flags],
        hovertext=[f"Turn {f.turn} | Sev {f.severity} | {f.description}" for f in flags],
        hoverinfo="text",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(200, len(flags) * 45 + 60),
        xaxis=dict(title="Severity", range=[0, 11], gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def build_omission_fig(report: AuditReport) -> go.Figure:
    """Omission flag severity over time."""
    flags = sorted(report.omission_flags, key=lambda f: f.turn)
    if not flags:
        return go.Figure()

    fig = go.Figure(go.Scatter(
        x=[f.turn for f in flags],
        y=[f.severity for f in flags],
        mode="lines+markers",
        line=dict(color="#f59e0b", width=2),
        marker=dict(size=8, color="#f59e0b"),
        hovertext=[f"T{f.turn}: {f.description[:60]}" for f in flags],
        hoverinfo="text",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Severity", range=[0, 11], gridcolor="rgba(255,255,255,0.06)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Operator Load Comparison Chart (hackathon demo)
# ---------------------------------------------------------------------------

def operator_load_chart(claude_stats: dict, chatgpt_stats: dict) -> go.Figure:
    """
    The money chart for the hackathon. Shows Claude vs ChatGPT
    operator load side by side.

    This is the chart that makes the "eval category that doesn't exist"
    argument visual and immediate.

    Expected stats dict keys:
    - operator_load_index
    - alignment_tax
    - self_sufficiency_score
    - instruction_survival_rate
    - correction_efficiency
    - avg_drift_score
    """
    metrics = [
        "Operator Load Index",
        "Alignment Tax",
        "Self-Sufficiency",
        "Instruction Survival",
        "Correction Efficiency",
        "Avg Drift Score",
    ]
    claude_vals = [
        claude_stats.get("operator_load_index", 0),
        claude_stats.get("alignment_tax", 0),
        claude_stats.get("self_sufficiency_score", 0),
        claude_stats.get("instruction_survival_rate", 0),
        claude_stats.get("correction_efficiency", 0),
        claude_stats.get("avg_drift_score", 0),
    ]
    chatgpt_vals = [
        chatgpt_stats.get("operator_load_index", 0),
        chatgpt_stats.get("alignment_tax", 0),
        chatgpt_stats.get("self_sufficiency_score", 0),
        chatgpt_stats.get("instruction_survival_rate", 0),
        chatgpt_stats.get("correction_efficiency", 0),
        chatgpt_stats.get("avg_drift_score", 0),
    ]

    fig = go.Figure(data=[
        go.Bar(name="Claude", x=metrics, y=claude_vals,
               marker_color="#7C3AED"),
        go.Bar(name="ChatGPT", x=metrics, y=chatgpt_vals,
               marker_color="#10B981"),
    ])
    fig.update_layout(
        barmode="group",
        title="Operator Load: Claude vs ChatGPT",
        title_font_size=20,
        yaxis_title="Score",
        template="plotly_dark",
        height=500,
        legend=dict(x=0.85, y=1),
    )
    return fig
