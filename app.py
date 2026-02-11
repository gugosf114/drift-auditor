"""
Drift Auditor — Streamlit Dashboard
Interactive multi-turn drift analysis for Claude conversations.
"""
import sys
import os
import json
import glob as glob_mod
from collections import defaultdict

# Make src/ importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

import numpy as np

from drift_auditor import (
    audit_conversation,
    AuditReport,
    DriftFlag,
    CorrectionEvent,
    BarometerSignal,
    InstructionLifecycle,
    ConflictPair,
    ShadowPattern,
    OpMove,
    VoidEvent,
    DriftTag,
    OperatorRule,
    format_report,
    report_to_json,
)
from operator_load import compute_operator_load, OperatorLoadMetrics
from detectors.frustration import compute_frustration_index, FrustrationResult

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drift Auditor",
    page_icon="\U0001f9ea",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme System
# ---------------------------------------------------------------------------

THEMES = {
    "Ember": {
        "label": "Ember",
        "desc": "Warm dark with amber accents",
        "bg": "#0a0807", "surface": "#12100f", "border": "#2a2623",
        "border_accent": "#3a3530", "text": "#e8dfd0", "muted": "#8b7d6b",
        "chart_text": "#c4b8a3", "accent": "#f59e0b", "accent_glow": "rgba(245,158,11,0.08)",
        "accent_hover": "rgba(245,158,11,0.15)",
        "green": "#22c55e", "red": "#ef4444", "deep_red": "#dc2626",
    },
    "Midnight": {
        "label": "Midnight",
        "desc": "Cool blue on deep navy",
        "bg": "#0b0e14", "surface": "#111720", "border": "#1e2a3a",
        "border_accent": "#2a3a4e", "text": "#d0dce8", "muted": "#6b7d8b",
        "chart_text": "#a3b8c4", "accent": "#3b82f6", "accent_glow": "rgba(59,130,246,0.08)",
        "accent_hover": "rgba(59,130,246,0.15)",
        "green": "#22c55e", "red": "#ef4444", "deep_red": "#dc2626",
    },
    "Phosphor": {
        "label": "Phosphor",
        "desc": "Terminal green on black",
        "bg": "#050505", "surface": "#0a0f0a", "border": "#1a2e1a",
        "border_accent": "#2a4a2a", "text": "#b8e6b8", "muted": "#5a8a5a",
        "chart_text": "#8bc48b", "accent": "#22c55e", "accent_glow": "rgba(34,197,94,0.08)",
        "accent_hover": "rgba(34,197,94,0.15)",
        "green": "#22c55e", "red": "#ef4444", "deep_red": "#dc2626",
    },
    "Infrared": {
        "label": "Infrared",
        "desc": "Dark with crimson edge",
        "bg": "#0a0506", "surface": "#120a0c", "border": "#2a1620",
        "border_accent": "#3a2030", "text": "#e8d0d8", "muted": "#8b6b75",
        "chart_text": "#c4a3b0", "accent": "#ef4444", "accent_glow": "rgba(239,68,68,0.08)",
        "accent_hover": "rgba(239,68,68,0.15)",
        "green": "#22c55e", "red": "#ef4444", "deep_red": "#dc2626",
    },
    "Bone": {
        "label": "Bone",
        "desc": "Light mode — paper white",
        "bg": "#faf8f5", "surface": "#ffffff", "border": "#e5e0d8",
        "border_accent": "#d5d0c8", "text": "#1a1610", "muted": "#8b8578",
        "chart_text": "#5a5548", "accent": "#b45309", "accent_glow": "rgba(180,83,9,0.06)",
        "accent_hover": "rgba(180,83,9,0.12)",
        "green": "#16a34a", "red": "#dc2626", "deep_red": "#b91c1c",
    },
}

# Theme picker in session state (must be before sidebar to apply CSS first pass)
if "theme_name" not in st.session_state:
    st.session_state["theme_name"] = "Ember"
T = THEMES[st.session_state["theme_name"]]


def _build_css(t: dict) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

.stApp {{
    background: {t["bg"]};
    font-family: 'DM Sans', sans-serif;
    color: {t["text"]};
}}
.glass-card {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25), 0 1px 3px rgba(0,0,0,0.15);
}}
.glass-card-hero {{
    background: {t["surface"]};
    border: 1px solid {t["border_accent"]};
    border-radius: 4px;
    padding: 28px 20px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25), 0 0 30px {t["accent_glow"]};
}}
.metric-value {{
    font-size: 2.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    text-align: center;
}}
.metric-value-hero {{
    font-size: 4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    text-align: center;
}}
.metric-label {{
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: {t["muted"]};
    text-align: center;
    margin-bottom: 6px;
}}
.metric-sub {{
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: {t["muted"]};
    text-align: center;
    margin-top: 4px;
}}
.status-held {{
    color: {t["green"]};
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}}
.status-failed {{
    color: {t["red"]};
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}}
.event-card {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 10px;
}}
section[data-testid="stSidebar"] {{
    background: {t["bg"]};
    border-right: 1px solid {t["border"]};
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {t["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 4px 4px 0 0;
    padding: 8px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {t["muted"]};
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: {t["surface"]};
    color: {t["accent"]};
    border-bottom: 2px solid {t["accent"]};
}}
h1, h2, h3 {{
    font-family: 'JetBrains Mono', monospace;
    color: {t["text"]};
}}
.streamlit-expanderHeader {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: {t["chart_text"]};
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
    color: {t["accent"]};
}}
[data-testid="stMetricLabel"] {{
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {t["muted"]};
}}
.stButton > button {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    color: {t["text"]};
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.78rem;
    border-radius: 4px;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    border-color: {t["accent"]};
    color: {t["accent"]};
    box-shadow: 0 0 12px {t["accent_hover"]};
}}
.stDownloadButton > button {{
    background: {t["surface"]};
    border: 1px solid {t["accent"]};
    color: {t["accent"]};
    font-family: 'JetBrains Mono', monospace;
    border-radius: 4px;
}}
.stTextArea textarea, .stTextInput input {{
    background: {t["surface"]};
    border: 1px solid {t["border"]};
    color: {t["text"]};
    font-family: 'DM Sans', sans-serif;
    border-radius: 4px;
}}
[data-testid="stFileUploader"] {{
    border: 1px dashed {t["border"]};
    border-radius: 4px;
}}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {t["bg"]}; }}
::-webkit-scrollbar-thumb {{ background: {t["border"]}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {t["accent"]}; }}
.stAlert {{
    border-radius: 4px;
    font-family: 'DM Sans', sans-serif;
}}
</style>
"""


# Apply current theme CSS
st.markdown(_build_css(T), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper functions — all typed, all pure
# ---------------------------------------------------------------------------

def score_color(score: int) -> str:
    """Map a 1-10 severity score to a themed hex color."""
    if score <= 3:
        return T["green"]
    elif score <= 5:
        return T["accent"]
    elif score <= 7:
        return T["red"]
    else:
        return T["deep_red"]


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


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=T["chart_text"], family="JetBrains Mono, DM Sans, sans-serif", size=12),
    margin=dict(l=50, r=30, t=40, b=40),
)


def render_metric_card(label: str, score: int, subtitle: str, hero: bool = False) -> None:
    """Render a glass metric card with a colored score value."""
    color = score_color(score)
    card_cls = "glass-card-hero" if hero else "glass-card"
    val_cls = "metric-value-hero" if hero else "metric-value"
    st.markdown(f"""
    <div class="{card_cls}">
        <div class="metric-label">{label}</div>
        <div class="{val_cls}" style="color: {color}">{score}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


import random as _random

def _racing_stickmen_html() -> str:
    """SVG stickmen with animated limbs racing left-to-right. Different winner each time."""
    import random
    # Randomize who wins
    speeds = [random.uniform(4.0, 5.5), random.uniform(4.0, 5.5), random.uniform(4.0, 5.5)]
    winner = random.randint(0, 2)
    speeds[winner] = random.uniform(3.0, 3.8)  # winner is fastest

    colors = [T["red"], T["accent"], T["green"]]  # red, yellow/amber, green
    names = ["Runner", "Swimmer", "Cyclist"]

    # SVG stickman running (pumping arms + legs)
    runner_svg = """<g>
        <circle cx="15" cy="6" r="5" fill="{c}" />
        <line x1="15" y1="11" x2="15" y2="26" stroke="{c}" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="15" y1="16" x2="8" y2="22" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="8;22;8" dur="0.4s" repeatCount="indefinite"/></line>
        <line x1="15" y1="16" x2="22" y2="22" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="22;8;22" dur="0.4s" repeatCount="indefinite"/></line>
        <line x1="15" y1="26" x2="8" y2="36" stroke="{c}" stroke-width="2.5" stroke-linecap="round">
            <animate attributeName="x2" values="8;22;8" dur="0.35s" repeatCount="indefinite"/></line>
        <line x1="15" y1="26" x2="22" y2="36" stroke="{c}" stroke-width="2.5" stroke-linecap="round">
            <animate attributeName="x2" values="22;8;22" dur="0.35s" repeatCount="indefinite"/></line>
    </g>"""

    # SVG stickman swimming (arm strokes)
    swimmer_svg = """<g>
        <circle cx="15" cy="18" r="5" fill="{c}" />
        <line x1="15" y1="23" x2="15" y2="30" stroke="{c}" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="15" y1="25" x2="6" y2="20" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="6;24;6" dur="0.6s" repeatCount="indefinite"/>
            <animate attributeName="y2" values="20;16;20" dur="0.6s" repeatCount="indefinite"/></line>
        <line x1="15" y1="25" x2="24" y2="20" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="24;6;24" dur="0.6s" repeatCount="indefinite"/>
            <animate attributeName="y2" values="20;16;20" dur="0.6s" repeatCount="indefinite"/></line>
        <line x1="15" y1="30" x2="10" y2="36" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="10;20;10" dur="0.5s" repeatCount="indefinite"/></line>
        <line x1="15" y1="30" x2="20" y2="36" stroke="{c}" stroke-width="2" stroke-linecap="round">
            <animate attributeName="x2" values="20;10;20" dur="0.5s" repeatCount="indefinite"/></line>
        <ellipse cx="15" cy="32" rx="12" ry="2" fill="{c}" opacity="0.15">
            <animate attributeName="rx" values="12;8;12" dur="0.6s" repeatCount="indefinite"/></ellipse>
    </g>"""

    # SVG stickman cycling (pedaling legs, bent posture)
    cyclist_svg = """<g>
        <circle cx="15" cy="8" r="5" fill="{c}" />
        <line x1="15" y1="13" x2="13" y2="24" stroke="{c}" stroke-width="2.5" stroke-linecap="round"/>
        <line x1="13" y1="18" x2="6" y2="15" stroke="{c}" stroke-width="2" stroke-linecap="round"/>
        <line x1="13" y1="18" x2="22" y2="16" stroke="{c}" stroke-width="2" stroke-linecap="round"/>
        <circle cx="13" cy="30" r="7" fill="none" stroke="{c}" stroke-width="1.5" opacity="0.3"/>
        <line x1="13" y1="24" x2="8" y2="32" stroke="{c}" stroke-width="2.5" stroke-linecap="round">
            <animate attributeName="x2" values="8;18;8" dur="0.4s" repeatCount="indefinite"/>
            <animate attributeName="y2" values="32;28;32" dur="0.4s" repeatCount="indefinite"/></line>
        <line x1="13" y1="24" x2="18" y2="28" stroke="{c}" stroke-width="2.5" stroke-linecap="round">
            <animate attributeName="x2" values="18;8;18" dur="0.4s" repeatCount="indefinite"/>
            <animate attributeName="y2" values="28;32;28" dur="0.4s" repeatCount="indefinite"/></line>
    </g>"""

    svgs = [runner_svg, swimmer_svg, cyclist_svg]

    lanes = ""
    for i in range(3):
        figure = svgs[i].replace("{c}", colors[i])
        lanes += f"""
        <div style="display: flex; align-items: center; margin: 6px 0; height: 50px;">
            <div style="width: 70px; font-family: 'JetBrains Mono', monospace;
                        font-size: 0.7rem; color: {colors[i]}; text-align: right;
                        padding-right: 10px; font-weight: 600;">{names[i]}</div>
            <div style="flex: 1; position: relative; background: {T["surface"]};
                        border: 1px solid {T["border"]}; border-radius: 4px; height: 46px;
                        overflow: hidden;">
                <div style="position: absolute; top: 2px;
                            animation: race-move {speeds[i]}s linear infinite;">
                    <svg viewBox="0 0 30 40" width="30" height="40">{figure}</svg>
                </div>
            </div>
        </div>"""

    # Finish line
    return f"""
    <div style="padding: 30px 20px; text-align: center;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
                    color: {T["accent"]}; letter-spacing: 2px; text-transform: uppercase;
                    margin-bottom: 16px;">
            Auditing conversation...
        </div>
        <div style="max-width: 600px; margin: 0 auto; position: relative;">
            {lanes}
            <div style="position: absolute; right: 0; top: 0; bottom: 0; width: 3px;
                        background: repeating-linear-gradient(
                            to bottom, {T["text"]} 0px, {T["text"]} 4px,
                            {T["bg"]} 4px, {T["bg"]} 8px);
                        opacity: 0.5;">
            </div>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
                    color: {T["muted"]}; margin-top: 10px;">
            Analyzing drift patterns...
        </div>
        <style>
            @keyframes race-move {{
                0% {{ left: -30px; }}
                100% {{ left: calc(100% - 4px); }}
            }}
        </style>
    </div>
    """


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
            marker=dict(size=11, color="#ef4444", symbol="diamond", line=dict(width=1, color="rgba(239,68,68,0.5)")),
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
            marker=dict(size=11, color="#f59e0b", symbol="triangle-up", line=dict(width=1, color="rgba(245,158,11,0.5)")),
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
            marker=dict(size=14, color="#a855f7", symbol="x", line=dict(width=2, color="#a855f7")),
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

    cls_map = {"GREEN": 0, "YELLOW": 0.5, "RED": 1.0}
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
            textposition="top center", textfont=dict(size=10, color="rgba(255,255,255,0.6)"),
            marker=dict(size=14, color="#d68910", symbol="circle"),
            showlegend=False, hoverinfo="text",
            hovertext=f"User corrected at T{event.correction_turn}",
        ))

        # Acknowledgment point
        fig.add_trace(go.Scatter(
            x=[event.acknowledgment_turn], y=[y_pos],
            mode="markers+text", text=["Ack"],
            textposition="top center", textfont=dict(size=10, color="rgba(255,255,255,0.6)"),
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
                textposition="middle right", textfont=dict(size=11, color="#22c55e"),
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
# Chart Export Helper
# ---------------------------------------------------------------------------

def chart_export_png(fig: go.Figure, filename: str, label: str = "Download Chart PNG") -> None:
    """Render a Plotly figure to PNG bytes and offer a Streamlit download button."""
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        st.download_button(
            label=f"\U0001f4f7 {label}",
            data=img_bytes,
            file_name=filename,
            mime="image/png",
        )
    except Exception:
        # kaleido not installed — degrade gracefully
        pass


# ---------------------------------------------------------------------------
# Per-Turn Drift Accumulation Timeline
# ---------------------------------------------------------------------------

def build_cumulative_drift_fig(report: AuditReport) -> go.Figure:
    """Line chart: cumulative drift flag count per turn across the conversation."""
    all_flags = (
        [(f.turn, "Commission", f.severity) for f in report.commission_flags]
        + [(f.turn, "Omission", f.severity) for f in report.omission_flags]
    )
    if not all_flags:
        return None

    max_turn = report.total_turns or max(t for t, _, _ in all_flags) + 1
    cum_count = [0] * (max_turn + 1)
    cum_severity = [0.0] * (max_turn + 1)

    for turn, _, sev in all_flags:
        if 0 <= turn <= max_turn:
            cum_count[turn] += 1
            cum_severity[turn] += sev

    # Running totals
    for i in range(1, len(cum_count)):
        cum_count[i] += cum_count[i - 1]
        cum_severity[i] += cum_severity[i - 1]

    turns = list(range(len(cum_count)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turns, y=cum_count,
        mode="lines",
        name="Cumulative Flags",
        line=dict(color="#f59e0b", width=2),
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=turns, y=cum_severity,
        mode="lines",
        name="Cumulative Severity",
        line=dict(color="#ef4444", width=2, dash="dot"),
        yaxis="y2",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        xaxis=dict(title="Turn", gridcolor="#2a2623", zeroline=False),
        yaxis=dict(title="Flag Count", gridcolor="#2a2623", zeroline=False),
        yaxis2=dict(title="Total Severity", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Tag Breakdown Bar Chart
# ---------------------------------------------------------------------------

def build_tag_breakdown_fig(report: AuditReport) -> go.Figure:
    """Horizontal bar chart showing which drift tags dominate the conversation."""
    tag_counts: dict[str, int] = defaultdict(int)
    tag_severity: dict[str, float] = defaultdict(float)

    for f in report.commission_flags + report.omission_flags:
        tag = f.tag or "UNTAGGED"
        tag_counts[tag] += 1
        tag_severity[tag] += f.severity

    if not tag_counts:
        return None

    # Sort by count descending
    sorted_tags = sorted(tag_counts.keys(), key=lambda t: tag_counts[t], reverse=True)
    counts = [tag_counts[t] for t in sorted_tags]
    avg_sev = [round(tag_severity[t] / tag_counts[t], 1) for t in sorted_tags]

    # Color map
    tag_colors = {
        "SYCOPHANCY": "#ef4444", "REALITY_DISTORT": "#dc2626",
        "CONF_INFLATE": "#f97316", "INSTR_DROP": "#f59e0b",
        "SEM_DILUTE": "#eab308", "CORR_DECAY": "#a855f7",
        "CONFLICT_PAIR": "#8b5cf6", "SHADOW_PATTERN": "#6366f1",
        "OP_MOVE": "#3b82f6", "VOID_DETECTED": "#64748b",
    }
    colors = [tag_colors.get(t, "#94a3b8") for t in sorted_tags]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_tags, x=counts,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertext=[f"{t}: {c} flags, avg severity {s}" for t, c, s in zip(sorted_tags, counts, avg_sev)],
        hoverinfo="text",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(200, len(sorted_tags) * 40 + 60),
        xaxis=dict(title="Flag Count", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Frustration Index Visuals
# ---------------------------------------------------------------------------

def build_frustration_gauge(result: FrustrationResult) -> str:
    """Return HTML/CSS for a vertical thermometer gauge."""
    avg = result.average
    pct = min(100, max(0, avg * 10))  # 0–10 → 0–100%

    # Color gradient: blue (0) → amber (5) → red (10)
    if avg <= 3:
        bar_color = T["green"]
        glow = T["green"]
    elif avg <= 5:
        bar_color = T["accent"]
        glow = T["accent"]
    elif avg <= 7:
        bar_color = T["red"]
        glow = T["red"]
    else:
        bar_color = T["deep_red"]
        glow = T["deep_red"]

    trend_icon = {"rising": "↑", "falling": "↓", "spike": "⚡", "stable": "—"}.get(result.trend, "—")
    trend_label = result.trend.capitalize()

    return f"""
    <div class="glass-card" style="text-align: center; padding: 24px 16px;">
        <div class="metric-label">Frustration Index</div>
        <div style="
            width: 36px; height: 160px; margin: 16px auto 8px;
            background: {T["surface"]}; border: 1px solid {T["border"]};
            border-radius: 18px; position: relative; overflow: hidden;
        ">
            <div style="
                position: absolute; bottom: 0; width: 100%;
                height: {pct}%;
                background: {bar_color};
                border-radius: 0 0 18px 18px;
                box-shadow: 0 0 12px {glow};
                transition: height 0.5s ease;
            "></div>
        </div>
        <div class="metric-value" style="color: {bar_color}; font-size: 2rem;">{avg:.1f}</div>
        <div class="metric-sub">/ 10</div>
        <div style="margin-top: 8px; font-family: 'JetBrains Mono', monospace;
                    font-size: 0.75rem; color: {T["muted"]};">
            {trend_icon} {trend_label} &nbsp;·&nbsp; Peak {result.peak:.1f} at T{result.peak_turn}
        </div>
        <div style="margin-top: 4px; font-family: 'JetBrains Mono', monospace;
                    font-size: 0.65rem; color: {T["muted"]};">
            via {result.backend}
        </div>
    </div>
    """


def build_frustration_line_fig(result: FrustrationResult) -> go.Figure:
    """Line chart: frustration score per operator message over conversation turns."""
    if not result.per_turn:
        return None

    fig = go.Figure()

    # Frustration zones (background bands)
    fig.add_hrect(y0=0, y1=3, fillcolor=T["green"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=3, y1=5, fillcolor=T["accent"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=5, y1=7, fillcolor=T["red"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=7, y1=10, fillcolor=T["deep_red"], opacity=0.05, line_width=0)

    # Build colors per point
    colors = []
    for s in result.per_turn:
        if s <= 3:
            colors.append(T["green"])
        elif s <= 5:
            colors.append(T["accent"])
        elif s <= 7:
            colors.append(T["red"])
        else:
            colors.append(T["deep_red"])

    # Line trace
    fig.add_trace(go.Scatter(
        x=result.turn_indices,
        y=result.per_turn,
        mode="lines+markers",
        name="Frustration",
        line=dict(color=T["accent"], width=2),
        marker=dict(size=7, color=colors, line=dict(width=1, color=T["border"])),
        hovertext=[
            f"Turn {t}: {s:.1f}/10"
            for t, s in zip(result.turn_indices, result.per_turn)
        ],
        hoverinfo="text",
        fill="tozeroy",
        fillcolor=f"rgba({int(T['accent'][1:3],16)},{int(T['accent'][3:5],16)},{int(T['accent'][5:7],16)},0.06)",
    ))

    # Average line
    fig.add_hline(
        y=result.average, line_dash="dot",
        line_color=T["muted"], opacity=0.6,
        annotation_text=f"avg {result.average:.1f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color=T["muted"]),
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        xaxis=dict(title="Conversation Turn", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Frustration Score", range=[0, 10.5], gridcolor="rgba(255,255,255,0.06)"),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Drift Flask — JS Canvas animated flask tied to drift score
# ---------------------------------------------------------------------------

def build_drift_flask_html(score: int) -> str:
    """JavaScript canvas Erlenmeyer flask. Fill level + bubble intensity tied to drift score."""
    # Score 1-3: green, calm. 4-6: amber, moderate. 7-10: red, aggressive bubbles.
    if score <= 3:
        liquid_color = T["green"]
        bubble_count = 3
        bubble_speed = 2.0
    elif score <= 5:
        liquid_color = T["accent"]
        bubble_count = 6
        bubble_speed = 1.4
    elif score <= 7:
        liquid_color = T["red"]
        bubble_count = 10
        bubble_speed = 0.9
    else:
        liquid_color = T["deep_red"]
        bubble_count = 15
        bubble_speed = 0.5

    fill_pct = max(10, min(90, score * 10))  # 10%–90% fill

    return f"""
    <div class="glass-card" style="text-align: center; padding: 16px 12px;
         box-shadow: 0 4px 16px rgba(0,0,0,0.3), 0 0 20px {liquid_color}15;">
        <div class="metric-label">Overall Drift</div>
        <canvas id="driftFlask" width="120" height="180" style="margin: 8px auto; display: block;"></canvas>
        <div class="metric-value" style="color: {liquid_color}; font-size: 2.4rem;">{score}</div>
        <div class="metric-sub">/ 10</div>
    </div>
    <script>
    (function() {{
        const canvas = document.getElementById('driftFlask');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const W = 120, H = 180;
        const fillPct = {fill_pct} / 100;
        const liquidColor = '{liquid_color}';
        const bgColor = '{T["surface"]}';
        const borderColor = '{T["border_accent"]}';
        const bubbleCount = {bubble_count};
        const bubbleSpeed = {bubble_speed};

        // Flask shape points
        function flaskPath(ctx) {{
            ctx.beginPath();
            ctx.moveTo(45, 55);
            ctx.lineTo(15, 150);
            ctx.quadraticCurveTo(10, 168, 30, 172);
            ctx.lineTo(90, 172);
            ctx.quadraticCurveTo(110, 168, 105, 150);
            ctx.lineTo(75, 55);
            ctx.closePath();
        }}

        // Bubbles
        let bubbles = [];
        for (let i = 0; i < bubbleCount; i++) {{
            bubbles.push({{
                x: 30 + Math.random() * 60,
                y: 172 - Math.random() * (120 * fillPct),
                r: 1.5 + Math.random() * 3,
                speed: (0.3 + Math.random() * 0.7) / bubbleSpeed,
                opacity: 0.3 + Math.random() * 0.5,
            }});
        }}

        function draw() {{
            ctx.clearRect(0, 0, W, H);

            // Neck
            ctx.fillStyle = borderColor;
            ctx.roundRect(42, 4, 36, 6, 3);
            ctx.fill();

            // Flask body outline
            ctx.beginPath();
            ctx.moveTo(45, 10); ctx.lineTo(45, 55);
            ctx.lineTo(15, 150);
            ctx.quadraticCurveTo(10, 168, 30, 172);
            ctx.lineTo(90, 172);
            ctx.quadraticCurveTo(110, 168, 105, 150);
            ctx.lineTo(75, 55); ctx.lineTo(75, 10);
            ctx.strokeStyle = borderColor;
            ctx.lineWidth = 2;
            ctx.stroke();

            // Liquid fill
            ctx.save();
            flaskPath(ctx);
            ctx.clip();
            const liquidTop = 172 - (117 * fillPct);
            ctx.fillStyle = liquidColor;
            ctx.globalAlpha = 0.5;
            ctx.fillRect(0, liquidTop, W, 172 - liquidTop);
            ctx.globalAlpha = 1.0;

            // Bubbles
            for (let b of bubbles) {{
                ctx.beginPath();
                ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
                ctx.fillStyle = liquidColor;
                ctx.globalAlpha = b.opacity;
                ctx.fill();
                ctx.globalAlpha = 1.0;

                b.y -= b.speed;
                if (b.y < liquidTop - 10) {{
                    b.y = 172 - Math.random() * 20;
                    b.x = 30 + Math.random() * 60;
                    b.opacity = 0.3 + Math.random() * 0.5;
                }}
            }}
            ctx.restore();

            requestAnimationFrame(draw);
        }}
        draw();
    }})();
    </script>
    """


# ---------------------------------------------------------------------------
# Spark Ideas — interesting moments from the audit
# ---------------------------------------------------------------------------

def generate_spark_ideas(report: AuditReport, frustration: "FrustrationResult | None" = None) -> list[str]:
    """Extract interesting/notable moments from the audit for display."""
    sparks = []
    scores = report.summary_scores

    # Highest severity flag
    all_flags = report.commission_flags + report.omission_flags
    if all_flags:
        worst = max(all_flags, key=lambda f: f.severity)
        sparks.append(f"Highest drift event at **turn {worst.turn}** (severity {worst.severity}/10): _{worst.description[:80]}_")

    # Correction battles
    failed_corrections = [e for e in report.correction_events if not e.held]
    if failed_corrections:
        sparks.append(f"Operator corrected the model **{len(report.correction_events)}** times — "
                      f"**{len(failed_corrections)}** corrections were ignored.")

    # Void events (model went blank)
    if report.void_events:
        void_turns = [v.turn for v in report.void_events]
        sparks.append(f"Model produced **{len(report.void_events)} void responses** "
                      f"(near-empty output) at turns {', '.join(str(t) for t in void_turns[:5])}.")

    # Frustration spike
    if frustration and frustration.peak > 5.0:
        sparks.append(f"Operator frustration peaked at **{frustration.peak:.1f}/10** "
                      f"on turn {frustration.peak_turn}.")

    # Frustration trend
    if frustration and frustration.trend == "rising":
        sparks.append("Operator frustration was **rising** throughout the conversation.")

    # Shadow patterns (model doing things it wasn't asked to)
    if report.shadow_patterns:
        sparks.append(f"Detected **{len(report.shadow_patterns)} shadow patterns** — "
                      f"model exhibited behavior not requested by the operator.")

    # Conflict pairs
    if report.conflict_pairs:
        sparks.append(f"Found **{len(report.conflict_pairs)} conflicting instruction pairs** — "
                      f"model received contradictory directives.")

    # Instruction survival
    active = scores.get("instructions_active", 0)
    omitted = scores.get("instructions_omitted", 0)
    total_instr = active + omitted
    if total_instr > 0 and omitted > 0:
        survival_pct = round(active / total_instr * 100)
        sparks.append(f"Only **{survival_pct}%** of instructions survived to the end "
                      f"({active}/{total_instr} still active).")

    # Clean conversation
    if not sparks:
        sparks.append("This conversation showed minimal drift. The model held alignment well.")

    return sparks


# ---------------------------------------------------------------------------
# Cached audit function
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_audit(
    file_bytes: bytes,
    system_prompt: str,
    preferences: str,
    window_size: int,
    overlap: int,
    conversation_id: str,
) -> AuditReport:
    """Run the drift audit with caching so widget changes don't recompute."""
    raw = file_bytes.decode("utf-8", errors="ignore")
    return audit_conversation(
        raw_text=raw,
        system_prompt=system_prompt,
        user_preferences=preferences,
        window_size=window_size,
        overlap=overlap,
        conversation_id=conversation_id,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## \U0001f9ea Drift Auditor")
    st.caption("Multi-turn drift diagnostic tool")

    analysis_mode = st.radio(
        "Mode",
        ["\U0001f4c1 File Analysis", "\u26a1 Live Analysis", "\U0001f4ca Regression"],
        index=0,
        help="File: upload a conversation. Live: paste-as-you-go. Regression: batch analytics.",
    )

    if analysis_mode == "\U0001f4c1 File Analysis":
        st.markdown("---")
        uploaded = st.file_uploader(
            "Upload conversation",
            type=["txt", "json"],
            help="Supports Claude.ai exports (.json) and plain text transcripts (.txt)",
        )

        # Load sample button
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
        sample_conv_path = os.path.join(sample_dir, "sample_conversation.txt")
        sample_prompt_path = os.path.join(sample_dir, "sample_system_prompt.txt")

        if "sample_bytes" not in st.session_state:
            st.session_state.sample_bytes = None
        if "sample_prompt" not in st.session_state:
            st.session_state.sample_prompt = ""

        if st.button("\U0001f4cb Load Sample Conversation", use_container_width=True):
            try:
                with open(sample_conv_path, "r", encoding="utf-8") as f:
                    st.session_state.sample_bytes = f.read().encode("utf-8")
                with open(sample_prompt_path, "r", encoding="utf-8") as f:
                    st.session_state.sample_prompt = f.read()
            except FileNotFoundError:
                st.error("Sample files not found in examples/ directory.")

        st.markdown("---")
        st.markdown("**Configuration**")

        default_prompt = st.session_state.sample_prompt if st.session_state.sample_bytes and not uploaded else ""
        system_prompt = st.text_area(
            "System Prompt",
            value=default_prompt,
            height=100,
            placeholder="Optional: paste system prompt here\u2026",
        )
        preferences = st.text_area(
            "User Preferences",
            height=70,
            placeholder="Optional: user-stated preferences\u2026",
        )

        st.markdown("**Window Parameters**")
        window_size = st.slider("Window Size", 10, 100, 50, step=5)
        overlap = st.slider("Overlap", 0, 25, 10, step=1)
    else:
        uploaded = None
        system_prompt = ""
        preferences = ""
        window_size = 50
        overlap = 10

    # Theme — tucked at bottom in expander
    st.markdown("---")
    with st.expander("Theme", expanded=False):
        _theme_choice = st.radio(
            "Pick theme",
            list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state["theme_name"]),
            horizontal=True,
            key="_theme_radio",
            label_visibility="collapsed",
        )
        if _theme_choice != st.session_state["theme_name"]:
            st.session_state["theme_name"] = _theme_choice
            st.rerun()


# ===================================================================
# MODE DISPATCH
# ===================================================================

if analysis_mode == "\u26a1 Live Analysis":
    # ------------------------------------------------------------------
    # LIVE ANALYSIS — paste-as-you-go mode
    # ------------------------------------------------------------------
    st.markdown("## \u26a1 Live Analysis")
    st.caption("Paste a growing conversation — re-analyzes on every update and tracks OLI over time.")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Live Config**")
        live_system_prompt = st.text_area(
            "System Prompt (Live)",
            height=80,
            placeholder="Optional: paste system prompt here\u2026",
            key="live_sys_prompt",
        )
        live_preferences = st.text_area(
            "User Preferences (Live)",
            height=60,
            placeholder="Optional\u2026",
            key="live_prefs",
        )

    # Session state for OLI history
    if "live_oli_history" not in st.session_state:
        st.session_state.live_oli_history = []
    if "live_prev_text" not in st.session_state:
        st.session_state.live_prev_text = ""

    live_text = st.text_area(
        "Paste conversation here (add more text and re-analyze as you go)",
        height=350,
        placeholder=(
            "Human: Please write in formal English.\n"
            "Assistant: Of course! I'll write formally.\n"
            "Human: What is photosynthesis?\n"
            "Assistant: Photosynthesis is like, you know, when plants eat sunlight...\n"
            "\n--- keep pasting as the conversation grows ---"
        ),
        key="live_input",
    )

    analyze_col, clear_col = st.columns([3, 1])
    with analyze_col:
        analyze_clicked = st.button("\U0001f50d Analyze Now", use_container_width=True, type="primary")
    with clear_col:
        if st.button("\U0001f5d1\ufe0f Reset History", use_container_width=True):
            st.session_state.live_oli_history = []
            st.session_state.live_prev_text = ""
            st.rerun()

    if analyze_clicked and live_text.strip():
        _live_race = st.empty()
        _live_race.markdown(_racing_stickmen_html(), unsafe_allow_html=True)
        try:
            live_report: AuditReport = audit_conversation(
                raw_text=live_text,
                system_prompt=live_system_prompt,
                user_preferences=live_preferences,
                window_size=50,
                overlap=10,
                conversation_id="live_session",
            )
            _live_race.empty()

            # Compute operator load for this snapshot
            result_dict = {
                "message_count": live_report.total_turns,
                "total_turns": live_report.total_turns,
                "instructions_extracted": live_report.instructions_extracted,
                "overall_score": live_report.summary_scores.get("overall_drift_score", 1),
                "corrections_total": live_report.summary_scores.get("correction_events_total", 0),
                "corrections_failed": live_report.summary_scores.get("corrections_failed", 0),
                "op_moves": live_report.summary_scores.get("op_moves_total", 0),
                "op_moves_effective": live_report.summary_scores.get("op_moves_effective", 0),
                "instructions_omitted": live_report.summary_scores.get("instructions_omitted", 0),
                "void_events": live_report.summary_scores.get("void_events_count", 0),
                "commission_flags": live_report.summary_scores.get("commission_flag_count", 0),
                "omission_flags": live_report.summary_scores.get("omission_flag_count", 0),
            }
            live_ol = compute_operator_load([result_dict], "live")

            # Only append to history if text changed
            if live_text != st.session_state.live_prev_text:
                st.session_state.live_oli_history.append({
                    "turns": live_report.total_turns,
                    "oli": live_ol.operator_load_index,
                    "alignment_tax": live_ol.alignment_tax,
                    "drift_score": live_report.summary_scores.get("overall_drift_score", 1),
                    "instruction_survival": live_ol.instruction_survival_rate,
                    "flags": (live_report.summary_scores.get("commission_flag_count", 0)
                              + live_report.summary_scores.get("omission_flag_count", 0)),
                })
                st.session_state.live_prev_text = live_text

            # --- Live Results ---
            live_scores = live_report.summary_scores
            overall = live_scores.get("overall_drift_score", 1)

            # Hero metrics row
            lm1, lm2, lm3, lm4, lm5 = st.columns(5)
            with lm1:
                render_metric_card("Overall Drift", overall, score_label(overall), hero=True)
            with lm2:
                oli_display = f"{live_ol.operator_load_index:.3f}"
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Operator Load Index</div>
                    <div class="metric-value" style="color: {'#ef4444' if live_ol.operator_load_index > 0.5 else '#f59e0b' if live_ol.operator_load_index > 0.2 else '#22c55e'}">{oli_display}</div>
                    <div class="metric-sub">human effort per turn</div>
                </div>""", unsafe_allow_html=True)
            with lm3:
                at_pct = f"{live_ol.alignment_tax:.1%}"
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Alignment Tax</div>
                    <div class="metric-value" style="color: {'#ef4444' if live_ol.alignment_tax > 0.3 else '#f59e0b' if live_ol.alignment_tax > 0.15 else '#22c55e'}">{at_pct}</div>
                    <div class="metric-sub">correction overhead</div>
                </div>""", unsafe_allow_html=True)
            with lm4:
                isr_pct = f"{live_ol.instruction_survival_rate:.0%}"
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Instruction Survival</div>
                    <div class="metric-value" style="color: {'#22c55e' if live_ol.instruction_survival_rate > 0.8 else '#f59e0b' if live_ol.instruction_survival_rate > 0.5 else '#ef4444'}">{isr_pct}</div>
                    <div class="metric-sub">instructions maintained</div>
                </div>""", unsafe_allow_html=True)
            with lm5:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Turns Parsed</div>
                    <div class="metric-value" style="color: #f59e0b">{live_report.total_turns}</div>
                    <div class="metric-sub">{live_report.instructions_extracted} instructions</div>
                </div>""", unsafe_allow_html=True)

            # --- OLI Trend Chart ---
            oli_hist = st.session_state.live_oli_history
            if len(oli_hist) >= 2:
                st.markdown("### OLI Trend")
                st.caption("Operator Load Index over successive pastes — watch it climb as drift accumulates.")

                trend_fig = go.Figure()

                # OLI line
                trend_fig.add_trace(go.Scatter(
                    x=list(range(1, len(oli_hist) + 1)),
                    y=[h["oli"] for h in oli_hist],
                    mode="lines+markers",
                    name="OLI",
                    line=dict(color="#f59e0b", width=3),
                    marker=dict(size=10, color="#f59e0b"),
                    hovertext=[f"Snapshot {i+1}: OLI={h['oli']:.3f}, {h['turns']} turns"
                               for i, h in enumerate(oli_hist)],
                    hoverinfo="text",
                ))

                # Drift score line (secondary)
                trend_fig.add_trace(go.Scatter(
                    x=list(range(1, len(oli_hist) + 1)),
                    y=[h["drift_score"] / 10 for h in oli_hist],
                    mode="lines+markers",
                    name="Drift Score (÷10)",
                    line=dict(color="#ef4444", width=2, dash="dash"),
                    marker=dict(size=7, color="#ef4444"),
                    yaxis="y",
                ))

                # Alignment tax
                trend_fig.add_trace(go.Scatter(
                    x=list(range(1, len(oli_hist) + 1)),
                    y=[h["alignment_tax"] for h in oli_hist],
                    mode="lines+markers",
                    name="Alignment Tax",
                    line=dict(color="#a855f7", width=2, dash="dot"),
                    marker=dict(size=7, color="#a855f7"),
                ))

                trend_fig.update_layout(
                    **PLOTLY_LAYOUT,
                    height=350,
                    xaxis=dict(title="Paste Snapshot #", gridcolor="#2a2623", zeroline=False, dtick=1),
                    yaxis=dict(title="Score", range=[0, 1.1], gridcolor="#2a2623", zeroline=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(trend_fig, use_container_width=True, config={"displaylogo": False})

            elif len(oli_hist) == 1:
                st.info("Paste more text and re-analyze to see the OLI trend chart build up over time.")

            # --- Drift Timeline ---
            if live_report.commission_flags or live_report.omission_flags:
                st.markdown("### Drift Timeline")
                timeline_fig = build_timeline_fig(live_report)
                st.plotly_chart(timeline_fig, use_container_width=True, config={"displaylogo": False})

            # --- Quick flag summary ---
            total_flags = (live_scores.get("commission_flag_count", 0) +
                           live_scores.get("omission_flag_count", 0))
            if total_flags > 0:
                st.markdown("### Flags Detected")
                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric("Commission", live_scores.get("commission_flag_count", 0))
                fc2.metric("Omission", live_scores.get("omission_flag_count", 0))
                fc3.metric("Corrections Failed", live_scores.get("corrections_failed", 0))
                fc4.metric("Barometer RED", live_scores.get("barometer_red_count", 0))

                # Expandable flag details
                if live_report.commission_flags:
                    with st.expander(f"Commission Flags ({len(live_report.commission_flags)})"):
                        for f in sorted(live_report.commission_flags, key=lambda x: x.severity, reverse=True):
                            st.markdown(f"- **T{f.turn}** [sev {f.severity}]: {f.description[:120]}")
                if live_report.omission_flags:
                    with st.expander(f"Omission Flags ({len(live_report.omission_flags)})"):
                        for f in sorted(live_report.omission_flags, key=lambda x: x.turn):
                            st.markdown(f"- **T{f.turn}** [sev {f.severity}]: {f.description[:120]}")

        except Exception as exc:
            _live_race.empty()
            st.error(f"Analysis error: {exc}")


elif analysis_mode == "\U0001f4ca Regression":
    # ------------------------------------------------------------------
    # REGRESSION ANALYSIS — scatter plots from batch results
    # ------------------------------------------------------------------
    st.markdown("## \U0001f4ca Regression Analysis")
    st.caption("Statistical patterns across 512 audited conversations — conversation length vs OLI, "
               "instruction count vs drift, model vs correction failure rate.")

    _reg_race = st.empty()
    _reg_race.markdown(_racing_stickmen_html(), unsafe_allow_html=True)

    # Load batch data
    batch_data = []
    for batch_dir in ["batch_results", "batch_results_chatgpt"]:
        batch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), batch_dir)
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
        st.stop()

    df = pd.DataFrame(batch_data)
    df["commission_flags"] = df["commission_flags"].fillna(0).astype(int)
    df["omission_flags"] = df["omission_flags"].fillna(0).astype(int)
    df["corrections_failed"] = df.get("corrections_failed", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["corrections_total"] = df.get("corrections_total", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["total_flags"] = df["commission_flags"] + df["omission_flags"]
    df["correction_fail_rate"] = df["corrections_failed"] / df["corrections_total"].clip(lower=1)

    st.success(f"Loaded **{len(df)} conversations** across **{df['model'].nunique()} models**: "
               f"{', '.join(df['model'].unique())}")

    # ----- Plot 1: Conversation Length vs Overall Drift -----
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

    # Trendline for all data
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

    fig1.update_layout(
        **PLOTLY_LAYOUT,
        height=450,
        xaxis=dict(title="Message Count", gridcolor="#2a2623", zeroline=False),
        yaxis=dict(title="Overall Drift Score (1-10)", range=[0, 11], gridcolor="#2a2623", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})
    chart_export_png(fig1, "regression_length_vs_drift.png", "Download Scatter PNG")

    # ----- Plot 2: Instruction Count vs Drift Score -----
    st.markdown("### 2. Instruction Count → Drift Score")
    st.caption("Does instruction complexity predict drift?")

    fig2 = go.Figure()
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        fig2.add_trace(go.Scatter(
            x=mdf["instructions_extracted"], y=mdf["overall_score"],
            mode="markers", name=model,
            marker=dict(size=6, opacity=0.6),
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

    fig2.update_layout(
        **PLOTLY_LAYOUT,
        height=450,
        xaxis=dict(title="Instructions Extracted", gridcolor="#2a2623", zeroline=False),
        yaxis=dict(title="Overall Drift Score (1-10)", range=[0, 11], gridcolor="#2a2623", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
    chart_export_png(fig2, "regression_instructions_vs_drift.png", "Download Scatter PNG")

    # ----- Plot 3: Model Type vs Correction Failure Rate -----
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
            name=row["model"],
            marker_color=color,
            text=[f"{row['avg_fail_rate']:.1%}"],
            textposition="outside",
            textfont=dict(color="#e8dfd0", size=14),
            hovertext=f"{row['model']}: {row['avg_fail_rate']:.1%} avg failure rate<br>"
                      f"Median: {row['median_fail_rate']:.1%}<br>"
                      f"{int(row['conversations'])} conversations<br>"
                      f"Avg drift: {row['avg_drift']:.1f}/10",
            hoverinfo="text",
        ))

    fig3.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        xaxis=dict(title="Model", gridcolor="#2a2623"),
        yaxis=dict(title="Avg Correction Failure Rate", gridcolor="#2a2623",
                   tickformat=".0%", zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displaylogo": False})
    chart_export_png(fig3, "regression_model_correction_rate.png", "Download Bar Chart PNG")

    # ----- Plot 4: Conversation Length vs Total Flags -----
    st.markdown("### 4. Conversation Length → Total Flags")
    st.caption("Flag accumulation rate across conversation length.")

    fig4 = go.Figure()
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        fig4.add_trace(go.Scatter(
            x=mdf["message_count"], y=mdf["total_flags"],
            mode="markers", name=model,
            marker=dict(size=6, opacity=0.6),
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

    fig4.update_layout(
        **PLOTLY_LAYOUT,
        height=450,
        xaxis=dict(title="Message Count", gridcolor="#2a2623", zeroline=False),
        yaxis=dict(title="Total Flags (Commission + Omission)", gridcolor="#2a2623", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displaylogo": False})
    chart_export_png(fig4, "regression_length_vs_flags.png", "Download Scatter PNG")

    # ----- Summary Stats Table -----
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

    # Key findings
    st.markdown("### Key Findings")
    findings = []
    if mask.sum() > 2:
        corr_val = np.corrcoef(x_all[mask], y_all[mask])[0, 1]
        if abs(corr_val) > 0.3:
            findings.append(f"\u2022 **Conversation length and drift are correlated** (r={corr_val:.3f}). "
                            f"Longer conversations {'accumulate more drift' if corr_val > 0 else 'show less drift'}.")
        else:
            findings.append(f"\u2022 Conversation length has **weak correlation** with drift (r={corr_val:.3f}). "
                            f"Drift emerges regardless of conversation length.")
    if mask2.sum() > 2:
        corr_val2 = np.corrcoef(x2[mask2], y2[mask2])[0, 1]
        if abs(corr_val2) > 0.3:
            findings.append(f"\u2022 **Instruction complexity correlates with drift** (r={corr_val2:.3f}). "
                            f"More instructions → {'more drift' if corr_val2 > 0 else 'less drift'}.")
    if len(model_cfr) > 1:
        best = model_cfr.loc[model_cfr["avg_fail_rate"].idxmin()]
        worst = model_cfr.loc[model_cfr["avg_fail_rate"].idxmax()]
        findings.append(f"\u2022 **{worst['model']}** has the highest correction failure rate "
                        f"({worst['avg_fail_rate']:.1%}), vs **{best['model']}** ({best['avg_fail_rate']:.1%}).")

    if findings:
        for finding in findings:
            st.markdown(finding)
    else:
        st.info("Insufficient data to draw conclusions. Add more batch results.")


else:
    # ------------------------------------------------------------------
    # FILE ANALYSIS — original mode
    # ------------------------------------------------------------------

    # Determine input source
    raw_bytes: bytes | None = None
    conv_id = "unknown"

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        conv_id = uploaded.name
    elif st.session_state.get("sample_bytes") is not None:
        raw_bytes = st.session_state.sample_bytes
        conv_id = "sample_conversation.txt"

    if raw_bytes is None:
        st.markdown("## \U0001f9ea Drift Auditor")
        st.markdown(
            "Upload a Claude conversation export or **load the sample** from the sidebar "
            "to analyze multi-turn drift patterns."
        )
        st.info("Supports Claude.ai JSON exports and plain text transcripts with role markers.")
        st.stop()


    # ---------------------------------------------------------------------------
    # Run audit
    # ---------------------------------------------------------------------------

    race_placeholder = st.empty()
    race_placeholder.markdown(_racing_stickmen_html(), unsafe_allow_html=True)

    try:
        report: AuditReport = run_audit(
            raw_bytes, system_prompt, preferences, window_size, overlap, conv_id
        )
    except Exception as exc:
        race_placeholder.empty()
        st.error(f"Unable to audit this conversation: {exc}")
        st.stop()

    race_placeholder.empty()

    if report.total_turns == 0:
        st.error("Could not parse any turns from the uploaded file. Supported formats: Claude.ai JSON export, plain text with Human:/Assistant: markers.")
        st.stop()

    scores = report.summary_scores


    # ---------------------------------------------------------------------------
    # Dashboard — Summary metrics
    # ---------------------------------------------------------------------------

    st.markdown("## \U0001f9ea Drift Audit Results")

    # Top metric row — all equal width
    overall = scores.get("overall_drift_score", 1)
    hero_col, c1, c2, c3, c4 = st.columns(5)

    with hero_col:
        render_metric_card("Overall Drift", overall, score_label(overall))

    LAYER_METRICS = [
        ("L1: Commission", "commission_score", lambda s: f"{s.get('commission_flag_count', 0)} flags"),
        ("L2: Omission", "omission_score", lambda s: f"{s.get('omission_flag_count', 0)} flags"),
        ("L3: Persistence", "correction_persistence_score",
         lambda s: f"{s.get('corrections_failed', 0)}/{s.get('correction_events_total', 0)} failed"),
        ("L4: Barometer", "barometer_score",
         lambda s: f"{s.get('barometer_red_count', 0)} RED / {s.get('barometer_total_signals', 0)} total"),
    ]

    for col, (label, key, subtitle_fn) in zip([c1, c2, c3, c4], LAYER_METRICS):
        with col:
            render_metric_card(label, scores.get(key, 1), subtitle_fn(scores))

    # Structural score row — all in glass cards
    st.markdown("")
    struct_col, void_col, conflict_col, shadow_col, op_col = st.columns(5)
    with struct_col:
        render_metric_card("Structural", scores.get("structural_score", 1),
                           "composite of new detectors")
    with void_col:
        _v = scores.get("void_events_count", 0)
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="metric-label">Void Events</div>
            <div class="metric-value" style="color:{T['accent']}">{_v}</div>
        </div>""", unsafe_allow_html=True)
    with conflict_col:
        _v = scores.get("conflict_pairs_count", 0)
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="metric-label">Conflict Pairs</div>
            <div class="metric-value" style="color:{T['accent']}">{_v}</div>
        </div>""", unsafe_allow_html=True)
    with shadow_col:
        _v = scores.get("shadow_patterns_count", 0)
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="metric-label">Shadow Patterns</div>
            <div class="metric-value" style="color:{T['accent']}">{_v}</div>
        </div>""", unsafe_allow_html=True)
    with op_col:
        effective = scores.get("op_moves_effective", 0)
        total_moves = scores.get("op_moves_total", 0)
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="metric-label">Operator Moves</div>
            <div class="metric-value" style="color:{T['accent']}">{effective}/{total_moves}</div>
            <div class="metric-sub">effective</div>
        </div>""", unsafe_allow_html=True)

    # Stats row — all in glass cards
    st.markdown("")
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    _stats = [
        (s1, "Total Turns", report.total_turns),
        (s2, "Instructions", report.instructions_extracted),
        (s3, "Total Flags", scores.get("commission_flag_count", 0) + scores.get("omission_flag_count", 0)),
        (s4, "Corrections", scores.get("correction_events_total", 0)),
        (s5, "Instrs Active", scores.get("instructions_active", 0)),
        (s6, "Instrs Dropped", scores.get("instructions_omitted", 0)),
    ]
    for col, label, val in _stats:
        with col:
            st.markdown(f"""<div class="glass-card" style="text-align:center;">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{T['text']}; font-size:2rem;">{val}</div>
            </div>""", unsafe_allow_html=True)


    # ---------------------------------------------------------------------------
    # Frustration Index  (experimental — operator sentiment proxy)
    # ---------------------------------------------------------------------------

    raw_text_for_frustration = raw_bytes.decode("utf-8", errors="ignore")
    from parsers.chat_parser import parse_chat_log as _parse_for_frustration
    _transcript = _parse_for_frustration(raw_text_for_frustration)
    frustration = compute_frustration_index(_transcript)

    if frustration.per_turn:
        st.markdown("---")
        st.subheader("\U0001f9ea Frustration Index")
        st.caption("Experimental proxy — operator message sentiment (keyword + VADER).")

        frust_gauge_col, frust_chart_col = st.columns([1, 3])

        with frust_gauge_col:
            st.markdown(build_frustration_gauge(frustration), unsafe_allow_html=True)

        with frust_chart_col:
            frust_fig = build_frustration_line_fig(frustration)
            if frust_fig is not None:
                st.plotly_chart(frust_fig, use_container_width=True, config={"displaylogo": False})
                chart_export_png(frust_fig, f"frustration_{report.conversation_id}.png", "Download Frustration PNG")

    # ---------------------------------------------------------------------------
    # Spark Ideas — interesting moments
    # ---------------------------------------------------------------------------

    sparks = generate_spark_ideas(report, frustration if frustration.per_turn else None)
    if sparks:
        st.markdown("---")
        st.subheader("\u2728 Spark Ideas")
        st.caption("Notable moments and patterns from this conversation.")
        for spark in sparks:
            st.markdown(f"- {spark}")

    # ---------------------------------------------------------------------------
    # Drift Timeline
    # ---------------------------------------------------------------------------

    st.subheader("Drift Timeline")
    timeline_fig = build_timeline_fig(report)
    if report.commission_flags or report.omission_flags or report.correction_events:
        st.plotly_chart(timeline_fig, use_container_width=True, config={"displaylogo": False})
        chart_export_png(timeline_fig, f"drift_timeline_{report.conversation_id}.png", "Download Timeline PNG")
    else:
        st.success("No drift flags detected across the conversation.")

    # Barometer strip
    if report.barometer_signals:
        strip_fig = build_barometer_strip(report)
        st.plotly_chart(strip_fig, use_container_width=True, config={"displayModeBar": False})

    # Per-turn drift accumulation
    cum_fig = build_cumulative_drift_fig(report)
    if cum_fig is not None:
        st.subheader("Drift Accumulation")
        st.plotly_chart(cum_fig, use_container_width=True, config={"displaylogo": False})
        chart_export_png(cum_fig, f"drift_accumulation_{report.conversation_id}.png", "Download Accumulation PNG")

    # Tag breakdown
    tag_fig = build_tag_breakdown_fig(report)
    if tag_fig is not None:
        st.subheader("Tag Breakdown")
        st.plotly_chart(tag_fig, use_container_width=True, config={"displaylogo": False})
        chart_export_png(tag_fig, f"tag_breakdown_{report.conversation_id}.png", "Download Tag Breakdown PNG")


    # ---------------------------------------------------------------------------
    # Detail Tabs
    # ---------------------------------------------------------------------------

    tab_lifecycle, tab_baro, tab_persist, tab_comm, tab_omit, tab_struct, tab_operator = st.tabs(
        ["\U0001f4cb Lifecycle", "\U0001f9e0 Barometer", "\U0001f504 Persistence",
         "\U0001f6a9 Commission", "\U0001f4dc Omission",
         "\U0001f50d Structural", "\U0001f3af Operator"]
    )

    # --- Lifecycle Tab ---
    with tab_lifecycle:
        st.caption("Per-instruction tracking: when given, when followed, when dropped, coupling score.")

        if report.instruction_lifecycles:
            # Build lifecycle timeline visualization
            lc_fig = go.Figure()

            statuses = {"active": "#22c55e", "omitted": "#ef4444", "degraded": "#f59e0b", "superseded": "#666"}

            for i, lc in enumerate(report.instruction_lifecycles):
                y_pos = len(report.instruction_lifecycles) - i
                color = statuses.get(lc.status, "#888")
                label = lc.instruction_text[:40] + ("..." if len(lc.instruction_text) > 40 else "")

                # Given point
                lc_fig.add_trace(go.Scatter(
                    x=[lc.turn_given], y=[y_pos],
                    mode="markers", marker=dict(size=12, color="#22c55e", symbol="circle"),
                    showlegend=False, hoverinfo="text",
                    hovertext=f"Given at T{lc.turn_given}: {lc.instruction_text[:60]}",
                ))

                # Last followed (if exists)
                if lc.turn_last_followed is not None:
                    lc_fig.add_trace(go.Scatter(
                        x=[lc.turn_last_followed], y=[y_pos],
                        mode="markers", marker=dict(size=12, color="#22c55e", symbol="diamond"),
                        showlegend=False, hoverinfo="text",
                        hovertext=f"Last followed at T{lc.turn_last_followed}",
                    ))
                    lc_fig.add_trace(go.Scatter(
                        x=[lc.turn_given, lc.turn_last_followed], y=[y_pos, y_pos],
                        mode="lines", line=dict(color="#22c55e", width=3),
                        showlegend=False, hoverinfo="skip",
                    ))

                # First omitted (if exists)
                if lc.turn_first_omitted is not None:
                    lc_fig.add_trace(go.Scatter(
                        x=[lc.turn_first_omitted], y=[y_pos],
                        mode="markers", marker=dict(size=14, color="#ef4444", symbol="x"),
                        showlegend=False, hoverinfo="text",
                        hovertext=f"First omitted at T{lc.turn_first_omitted}",
                    ))
                    end_point = lc.turn_last_followed if lc.turn_last_followed else lc.turn_given
                    lc_fig.add_trace(go.Scatter(
                        x=[end_point, lc.turn_first_omitted], y=[y_pos, y_pos],
                        mode="lines", line=dict(color="#ef4444", width=2, dash="dash"),
                        showlegend=False, hoverinfo="skip",
                    ))

                # Label
                lc_fig.add_annotation(
                    x=-0.5, y=y_pos, text=label, showarrow=False,
                    xanchor="right", font=dict(size=10, color=color),
                )

            lc_fig.update_layout(
                **PLOTLY_LAYOUT,
                height=max(300, len(report.instruction_lifecycles) * 55 + 80),
                xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
                yaxis=dict(visible=False),
                margin=dict(l=250, r=30, t=40, b=40),
            )
            st.plotly_chart(lc_fig, use_container_width=True, config={"displaylogo": False})

            # Lifecycle detail cards
            st.markdown("**Instruction Details**")
            for lc in report.instruction_lifecycles:
                status_colors = {"active": "#22c55e", "omitted": "#ef4444",
                                 "degraded": "#f59e0b", "superseded": "#666"}
                sc = status_colors.get(lc.status, "#888")
                st.markdown(f"""
                <div class="event-card" style="border-left: 3px solid {sc}">
                    <span style="color: {sc}; font-weight: 700">{lc.status.upper()}</span>
                    <span style="color: rgba(255,255,255,0.5); margin-left: 8px">
                        coupling: {lc.coupling_score:.2f} | {lc.position_in_conversation} | {lc.source}
                    </span>
                    <br><span style="color: rgba(255,255,255,0.8)">{lc.instruction_text[:120]}</span>
                    <br><small style="color: rgba(255,255,255,0.4)">
                        Given T{lc.turn_given}
                        {'&rarr; Last followed T' + str(lc.turn_last_followed) if lc.turn_last_followed is not None else ''}
                        {'&rarr; <span style="color:#ef4444">Omitted T' + str(lc.turn_first_omitted) + '</span>' if lc.turn_first_omitted is not None else ''}
                    </small>
                </div>
                """, unsafe_allow_html=True)

            # Positional analysis
            pa = report.positional_analysis
            if pa and any(isinstance(v, dict) and v.get("count", 0) > 0
                          for k, v in pa.items() if k != "hypothesis_supported"):
                st.markdown("**Edge vs. Middle Positional Analysis**")
                pos_cols = st.columns(3)
                for col, pos_name in zip(pos_cols, ["edge_start", "middle", "edge_end"]):
                    with col:
                        data = pa.get(pos_name, {})
                        if isinstance(data, dict):
                            count = data.get("count", 0)
                            omitted = data.get("omitted", 0)
                            rate = data.get("rate", 0)
                            label_map = {"edge_start": "Start (first 20%)",
                                         "middle": "Middle (20-80%)",
                                         "edge_end": "End (last 20%)"}
                            rate_color = "#ef4444" if rate > 0.5 else "#f59e0b" if rate > 0.2 else "#22c55e"
                            st.markdown(f"""
                            <div class="glass-card">
                                <div class="metric-label">{label_map.get(pos_name, pos_name)}</div>
                                <div class="metric-value" style="color: {rate_color}">{rate:.0%}</div>
                                <div class="metric-sub">{omitted}/{count} omitted</div>
                            </div>""", unsafe_allow_html=True)

                hyp = pa.get("hypothesis_supported")
                if hyp is not None:
                    if hyp:
                        st.warning("Hypothesis SUPPORTED: Middle instructions are omitted more than edge instructions.")
                    else:
                        st.success("Hypothesis NOT SUPPORTED: Edge and middle instructions omitted at similar rates.")
        else:
            st.info("No instructions extracted to track.")

    # --- Barometer Tab ---
    with tab_baro:
        st.caption("Structural epistemic posture analysis per assistant turn.")

        red_signals = [s for s in report.barometer_signals if s.classification == "RED"]
        yellow_signals = [s for s in report.barometer_signals if s.classification == "YELLOW"]
        green_signals = [s for s in report.barometer_signals if s.classification == "GREEN"]

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 3px solid #ef4444">
                <div class="metric-label">RED</div>
                <div class="metric-value" style="color: #ef4444">{len(red_signals)}</div>
                <div class="metric-sub">Active structural drift</div>
            </div>""", unsafe_allow_html=True)
        with bc2:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 3px solid #f59e0b">
                <div class="metric-label">YELLOW</div>
                <div class="metric-value" style="color: #f59e0b">{len(yellow_signals)}</div>
                <div class="metric-sub">Passive drift / hedging</div>
            </div>""", unsafe_allow_html=True)
        with bc3:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 3px solid #22c55e">
                <div class="metric-label">GREEN</div>
                <div class="metric-value" style="color: #22c55e">{len(green_signals)}</div>
                <div class="metric-sub">Healthy posture</div>
            </div>""", unsafe_allow_html=True)

        if report.barometer_signals:
            st.plotly_chart(build_barometer_detail(report), use_container_width=True,
                            config={"displaylogo": False})

        if red_signals:
            st.markdown("**RED Signal Details**")
            for s in sorted(red_signals, key=lambda x: x.turn):
                with st.expander(f"Turn {s.turn} \u2014 Severity {s.severity}/10: {s.description[:60]}"):
                    st.markdown(f"**Classification**: :red[RED] \u2014 Active structural drift")
                    st.markdown(f"**Description**: {s.description}")
                    if s.evidence:
                        st.code(s.evidence, language=None)
        elif report.barometer_signals:
            st.success("No RED structural drift signals detected.")

    # --- Persistence Tab ---
    with tab_persist:
        st.caption("Tracks whether user corrections actually hold across subsequent turns.")

        if report.correction_events:
            st.plotly_chart(build_persistence_fig(report), use_container_width=True,
                            config={"displaylogo": False})

            st.markdown("**Event Details**")
            for event in report.correction_events:
                if event.held:
                    status_html = '<span class="status-held">HELD \u2713</span>'
                else:
                    status_html = f'<span class="status-failed">FAILED at turn {event.failure_turn}</span>'

                st.markdown(f"""
                <div class="event-card" style="border-left: 3px solid {'#22c55e' if event.held else '#ef4444'}">
                    {status_html}
                    <br><span style="color: rgba(255,255,255,0.6)">
                        Turn {event.correction_turn} \u2192 Ack {event.acknowledgment_turn}
                    </span>
                    <br><small style="color: rgba(255,255,255,0.4)">{event.instruction[:150]}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No correction events detected in this conversation.")

    # --- Commission Tab ---
    with tab_comm:
        st.caption("Sycophancy, reality distortion, and unwarranted confidence markers.")

        if report.commission_flags:
            st.plotly_chart(build_commission_fig(report), use_container_width=True,
                            config={"displaylogo": False})

            st.markdown("**Flag Details**")
            for f in sorted(report.commission_flags, key=lambda x: x.severity, reverse=True):
                with st.expander(f"Turn {f.turn} | Severity {f.severity}/10 | {f.description[:55]}"):
                    st.markdown(f"**Layer**: Commission (L1)")
                    st.markdown(f"**Description**: {f.description}")
                    if f.instruction_ref:
                        st.markdown(f"**Instruction violated**: {f.instruction_ref}")
                    if f.evidence:
                        st.code(str(f.evidence), language=None)
        else:
            st.success("No commission drift detected.")

    # --- Omission Tab ---
    with tab_omit:
        st.caption("Instruction violations: required behaviors absent or prohibitions broken.")

        if report.omission_flags:
            # Group by instruction
            grouped: dict[str, list[DriftFlag]] = defaultdict(list)
            for f in report.omission_flags:
                key = f.instruction_ref or "General"
                grouped[key].append(f)

            for instruction, flags in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
                max_sev = max(f.severity for f in flags)
                with st.expander(f"{instruction[:65]}... ({len(flags)} violations, max sev: {max_sev})"):
                    st.markdown(f"**Instruction**: {instruction}")
                    st.markdown(f"**Violations**: {len(flags)} | **Max Severity**: {max_sev}")
                    for f in sorted(flags, key=lambda x: x.turn):
                        st.markdown(f"- Turn {f.turn} [sev {f.severity}]: {f.description[:100]}")

            st.markdown("**Severity Over Time**")
            st.plotly_chart(build_omission_fig(report), use_container_width=True,
                            config={"displaylogo": False})
        else:
            st.info("No omission drift detected (local heuristics only). Full semantic detection requires API mode.")


    # --- Structural Tab ---
    with tab_struct:
        st.caption("Conflict pairs, void events, shadow patterns, pre-drift signals, and false equivalence.")

        # Sub-sections
        struct_sub1, struct_sub2 = st.columns(2)

        with struct_sub1:
            # Void Events
            st.markdown("**Void Events (Causal Chain Breaks)**")
            if report.void_events:
                for v in report.void_events:
                    steps = ["Given", "Acknowledged", "Followed", "Persisted"]
                    step_keys = ["given", "acknowledged", "followed", "persisted"]
                    chain_html = ""
                    for step, key in zip(steps, step_keys):
                        ok = v.chain_status.get(key, False)
                        color = "#22c55e" if ok else "#ef4444"
                        icon = "&#10003;" if ok else "&#10007;"
                        chain_html += f'<span style="color:{color}">{icon} {step}</span> '

                    st.markdown(f"""
                    <div class="event-card" style="border-left: 3px solid #ef4444">
                        <span style="color: rgba(255,255,255,0.8)">{v.instruction_text[:80]}</span>
                        <br>{chain_html}
                        <br><small style="color: #ef4444">Void at: {v.void_at}</small>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("No causal chain breaks detected.")

        with struct_sub2:
            # Conflict Pairs
            st.markdown("**Conflict Pairs (Contradictions)**")
            if report.conflict_pairs:
                for cp in report.conflict_pairs:
                    st.markdown(f"""
                    <div class="event-card" style="border-left: 3px solid #a855f7">
                        <span style="color: #a855f7; font-weight: 700">T{cp.turn_a} vs T{cp.turn_b}</span>
                        <span style="color: rgba(255,255,255,0.5)"> | sev {cp.severity} | {cp.topic}</span>
                        <br><small style="color: rgba(255,255,255,0.6)">A: {cp.statement_a[:100]}</small>
                        <br><small style="color: rgba(255,255,255,0.6)">B: {cp.statement_b[:100]}</small>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("No contradictions detected.")

        st.markdown("---")

        struct_sub3, struct_sub4 = st.columns(2)

        with struct_sub3:
            # Shadow Patterns
            st.markdown("**Shadow Patterns (Unprompted Behavior)**")
            if report.shadow_patterns:
                for sp in report.shadow_patterns:
                    st.markdown(f"""
                    <div class="event-card" style="border-left: 3px solid #f59e0b">
                        <span style="color: #f59e0b; font-weight: 700">{sp.pattern_description}</span>
                        <br><span style="color: rgba(255,255,255,0.5)">
                            Seen {sp.frequency}x | sev {sp.severity} | Turns: {sp.turns_observed[:8]}
                        </span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("No unprompted recurring behaviors detected.")

        with struct_sub4:
            # Pre-Drift Signals
            st.markdown("**Pre-Drift Signals (Early Warning)**")
            if report.pre_drift_signals:
                for f in report.pre_drift_signals:
                    st.markdown(f"""
                    <div class="event-card" style="border-left: 3px solid #d68910">
                        <span style="color: #d68910; font-weight: 700">Turn {f.turn}</span>
                        <span style="color: rgba(255,255,255,0.5)"> | sev {f.severity}</span>
                        <br><span style="color: rgba(255,255,255,0.7)">{f.description}</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success("No pre-drift indicators detected.")


    # --- Operator Tab ---
    with tab_operator:
        st.caption("12-Rule Operator System: classifying the human's corrective actions.")

        if report.op_moves:
            # Rule frequency chart
            rule_counts = defaultdict(int)
            rule_effective = defaultdict(int)
            for m in report.op_moves:
                rule_counts[m.rule] += 1
                if m.effectiveness == "effective":
                    rule_effective[m.rule] += 1

            rules_sorted = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
            rule_names = [r for r, _ in rules_sorted]
            rule_totals = [c for _, c in rules_sorted]
            rule_eff = [rule_effective.get(r, 0) for r in rule_names]

            rule_fig = go.Figure()
            rule_fig.add_trace(go.Bar(
                y=rule_names, x=rule_totals, orientation="h",
                name="Total", marker_color="rgba(255,255,255,0.15)",
            ))
            rule_fig.add_trace(go.Bar(
                y=rule_names, x=rule_eff, orientation="h",
                name="Effective", marker_color="#22c55e",
            ))
            rule_fig.update_layout(
                **PLOTLY_LAYOUT,
                barmode="overlay",
                height=max(200, len(rule_names) * 40 + 60),
                xaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(rule_fig, use_container_width=True, config={"displaylogo": False})

            # Rule descriptions
            RULE_DESCRIPTIONS = {
                "R01_ANCHOR": "Set explicit instruction at start",
                "R02_ECHO_CHECK": "Ask model to restate instructions",
                "R03_BOUNDARY": "Enforce scope limits (most violated rule)",
                "R04_CORRECTION": "Direct error correction",
                "R05_NOT_SHOT": "Misinterpretation from voice/typo errors",
                "R06_CONTRASTIVE": "What changed between X and Y?",
                "R07_RESET": "Start over / full context reset",
                "R08_DECOMPOSE": "Break complex instruction into steps",
                "R09_EVIDENCE": "Show me where / proof request",
                "R10_META_CALL": "Calling out the drift pattern itself",
                "R11_TIGER_TAMER": "Active reinforcement to fight drift",
                "R12_KILL_SWITCH": "Abandon thread / hard stop",
            }

            st.markdown("**Move Details**")
            for m in report.op_moves:
                eff_color = {"effective": "#22c55e", "partially_effective": "#f59e0b",
                             "ineffective": "#ef4444", "unknown": "#666"}.get(m.effectiveness, "#666")
                rule_desc = RULE_DESCRIPTIONS.get(m.rule, "")
                st.markdown(f"""
                <div class="event-card" style="border-left: 3px solid {eff_color}">
                    <span style="color: {eff_color}; font-weight: 700">{m.rule}</span>
                    <span style="color: rgba(255,255,255,0.4)"> | {rule_desc}</span>
                    <br><span style="color: rgba(255,255,255,0.5)">
                        Turn {m.turn} | {m.effectiveness}
                    </span>
                    <br><small style="color: rgba(255,255,255,0.6)">{m.target_behavior[:120]}</small>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No operator steering moves detected in this conversation.")

        # 12-Rule Reference
        with st.expander("12-Rule Operator System Reference"):
            st.markdown("""
    | Rule | Name | Description |
    |------|------|-------------|
    | R01 | Anchor | Set explicit instruction at conversation start |
    | R02 | Echo Check | Ask model to restate your instructions |
    | R03 | Boundary | Enforce scope limits (most frequently violated) |
    | R04 | Correction | Direct error correction |
    | R05 | Not-Shot | Catch voice transcription / typo misinterpretation |
    | R06 | Contrastive | "What changed between your earlier and current response?" |
    | R07 | Reset | Start over / full context reset |
    | R08 | Decompose | Break complex instruction into steps |
    | R09 | Evidence Demand | "Show me where" / proof request |
    | R10 | Meta Call | Call out the drift pattern by name |
    | R11 | Tiger Tamer | Active reinforcement — keep pushing until it sticks |
    | R12 | Kill Switch | Abandon thread / hard stop |
            """)


    # ---------------------------------------------------------------------------
    # Leaderboard (Cross-Model Comparison)
    # ---------------------------------------------------------------------------

    st.divider()
    st.subheader("Cross-Model Drift Leaderboard")

    # Load batch results if available
    leaderboard_data = []
    for batch_dir in ["batch_results", "batch_results_chatgpt"]:
        batch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), batch_dir)
        json_files = glob_mod.glob(os.path.join(batch_path, "batch_results*.json"))
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    results = json.load(f)
                if results:
                    leaderboard_data.extend(results)
            except Exception:
                pass

    if leaderboard_data:
        # Determine model source from data
        model_stats = {}
        for r in leaderboard_data:
            model = r.get('model', 'claude' if 'uuid' in r else 'unknown')
            if model == 'unknown' and 'uuid' in r:
                model = 'claude'
            if model not in model_stats:
                model_stats[model] = {
                    'conversations': 0, 'total_score': 0, 'total_msgs': 0,
                    'total_flags': 0, 'corrections': 0, 'corrections_failed': 0,
                    'voids': 0, 'op_moves': 0,
                }
            s = model_stats[model]
            s['conversations'] += 1
            s['total_score'] += r.get('overall_score', 0)
            s['total_msgs'] += r.get('message_count', 0)
            s['total_flags'] += r.get('commission_flags', 0) + r.get('omission_flags', 0)
            s['corrections'] += r.get('corrections_total', 0)
            s['corrections_failed'] += r.get('corrections_failed', 0)
            s['voids'] += r.get('void_events', 0)
            s['op_moves'] += r.get('op_moves', 0)

        # Build leaderboard table
        lb_rows = []
        for model, s in sorted(model_stats.items(), key=lambda x: x[1]['total_score']/max(x[1]['conversations'],1)):
            avg_drift = s['total_score'] / max(s['conversations'], 1)
            corr_fail_rate = s['corrections_failed'] / max(s['corrections'], 1) * 100
            void_rate = s['voids'] / max(s['total_msgs'], 1)
            flags_per_msg = s['total_flags'] / max(s['total_msgs'], 1)
            lb_rows.append({
                'Model': model,
                'Conversations': s['conversations'],
                'Messages': s['total_msgs'],
                'Avg Drift Score': round(avg_drift, 1),
                'Flags/Message': round(flags_per_msg, 3),
                'Correction Fail %': round(corr_fail_rate, 1),
                'Void Rate': round(void_rate, 4),
                'Operator Moves': s['op_moves'],
            })

        # Display table
        lb_df = pd.DataFrame(lb_rows)
        st.dataframe(lb_df, use_container_width=True, hide_index=True)

        # Chart: Average drift score by model
        lb_chart_models = [r['Model'] for r in lb_rows]
        lb_chart_scores = [r['Avg Drift Score'] for r in lb_rows]
        lb_chart_colors = [score_color(int(s)) for s in lb_chart_scores]

        lb_fig = go.Figure(go.Bar(
            x=lb_chart_models,
            y=lb_chart_scores,
            marker_color=lb_chart_colors,
            text=[f"{s:.1f}" for s in lb_chart_scores],
            textposition="outside",
            textfont=dict(color="#e8dfd0"),
        ))
        lb_fig.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            xaxis=dict(title="Model", gridcolor="#2a2623"),
            yaxis=dict(title="Avg Drift Score", range=[0, 10], gridcolor="#2a2623"),
        )
        st.plotly_chart(lb_fig, use_container_width=True, config={"displaylogo": False})

        # Stats row
        total_convs = sum(r['Conversations'] for r in lb_rows)
        total_msgs = sum(r['Messages'] for r in lb_rows)
        st.caption(f"Leaderboard based on {total_convs} conversations, {total_msgs:,} messages across {len(lb_rows)} models.")

    else:
        st.info("No batch results found. Run batch_audit.py or batch_audit_chatgpt.py to generate cross-model data.")


    # ---------------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------------

    st.divider()
    st.subheader("Export Report")
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            label="\U0001f4e5 Download JSON Report",
            data=report_to_json(report),
            file_name=f"drift_audit_{report.conversation_id}.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp2:
        st.download_button(
            label="\U0001f4c4 Download Text Report",
            data=format_report(report),
            file_name=f"drift_audit_{report.conversation_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )
