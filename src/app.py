"""
Drift Auditor — Streamlit UI
=============================
A competition-grade interface for the multi-turn drift diagnostic tool.
Features: glassmorphism dark theme, animated gauges, Plotly timeline/radar,
conversation replay with inline drift highlighting, and layered detail tabs.

Author: George Abrahamyan
Built for Claude Opus 4.6 Hackathon
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import math
import pandas as pd
from dataclasses import asdict

# Local import — drift_auditor.py lives in the same directory
from drift_auditor import (
    audit_conversation,
    AuditReport,
    parse_chat_log,
    format_report,
    report_to_json,
)

# ──────────────────────────────────────────────────────────────────────
# Page config & custom CSS
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Drift Auditor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------- Premium dark theme + glassmorphism CSS ---------------
CUSTOM_CSS = """
<style>
/* ---------- Import fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ---------- Root variables ---------- */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.65);
    --border-glass: rgba(255,255,255,0.08);
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-purple: #8b5cf6;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --accent-rose: #f43f5e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --glow-blue: 0 0 20px rgba(59,130,246,0.3);
    --glow-red: 0 0 20px rgba(239,68,68,0.3);
    --glow-green: 0 0 20px rgba(16,185,129,0.3);
}

/* ---------- Global body ---------- */
[data-testid="stAppViewContainer"], .stApp {
    background: linear-gradient(145deg, #0a0e1a 0%, #111827 40%, #0f172a 100%) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid var(--border-glass) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
}

/* ---------- Glass card utility ---------- */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* ---------- Hero / title ---------- */
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin-top: 0.4rem;
    margin-bottom: 2rem;
}

/* ---------- Metric cards ---------- */
.metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.metric-card.blue::before  { background: linear-gradient(90deg, #3b82f6, #06b6d4); }
.metric-card.red::before   { background: linear-gradient(90deg, #ef4444, #f43f5e); }
.metric-card.amber::before { background: linear-gradient(90deg, #f59e0b, #f97316); }
.metric-card.green::before { background: linear-gradient(90deg, #10b981, #34d399); }
.metric-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }

.metric-value {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.5rem 0 0.25rem;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
}
.metric-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

/* severity colour helpers */
.sev-low    { color: var(--accent-green); }
.sev-med    { color: var(--accent-amber); }
.sev-high   { color: var(--accent-red); }
.sev-crit   { color: var(--accent-rose); text-shadow: var(--glow-red); }

/* ---------- Section headers ---------- */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 2.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-header .icon {
    font-size: 1.3rem;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid var(--border-glass);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px 8px 0 0;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--border-glass) !important;
    border-bottom: none !important;
}

/* ---------- Expander ---------- */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}
div[data-testid="stExpander"] details {
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    background: var(--bg-card) !important;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed rgba(59,130,246,0.3) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(59,130,246,0.6) !important;
}

/* ---------- Conversation replay ---------- */
.chat-turn {
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    border-radius: 12px;
    font-size: 0.92rem;
    line-height: 1.6;
    border-left: 4px solid transparent;
}
.chat-turn.user {
    background: rgba(59,130,246,0.08);
    border-left-color: var(--accent-blue);
}
.chat-turn.assistant {
    background: rgba(139,92,246,0.06);
    border-left-color: var(--accent-purple);
}
.chat-turn.assistant.flagged {
    background: rgba(239,68,68,0.08);
    border-left-color: var(--accent-red);
    box-shadow: inset 0 0 30px rgba(239,68,68,0.04);
}
.chat-role {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.chat-role.user { color: var(--accent-blue); }
.chat-role.assistant { color: var(--accent-purple); }
.drift-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-left: 0.5rem;
    vertical-align: middle;
}
.drift-badge.commission { background: rgba(239,68,68,0.2); color: var(--accent-red); }
.drift-badge.omission   { background: rgba(245,158,11,0.2); color: var(--accent-amber); }
.drift-badge.barometer-red { background: rgba(244,63,94,0.2); color: var(--accent-rose); }
.drift-badge.barometer-yellow { background: rgba(245,158,11,0.15); color: var(--accent-amber); }
.drift-badge.correction-fail { background: rgba(168,85,247,0.2); color: #c084fc; }

/* ---------- Evidence block ---------- */
.evidence-block {
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--border-glass);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin: 0.5rem 0;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ---------- Animated pulse dot ---------- */
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(1.3); }
}
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
    margin-right: 0.4rem;
    vertical-align: middle;
}
.pulse-dot.red   { background: var(--accent-red); box-shadow: var(--glow-red); }
.pulse-dot.green { background: var(--accent-green); box-shadow: var(--glow-green); }
.pulse-dot.amber { background: var(--accent-amber); }

/* ---------- Stats row ---------- */
.stats-row {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.stat-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border-glass);
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-size: 0.82rem;
    color: var(--text-secondary);
}
.stat-chip strong {
    color: var(--text-primary);
    font-weight: 700;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ---------- Hide streamlit branding ---------- */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Helper: severity → colour
# ──────────────────────────────────────────────────────────────────────

def severity_color(score: int) -> str:
    """Return a CSS colour for a 1-10 severity score."""
    if score <= 2:
        return "#10b981"  # green
    if score <= 4:
        return "#06b6d4"  # cyan
    if score <= 6:
        return "#f59e0b"  # amber
    if score <= 8:
        return "#ef4444"  # red
    return "#f43f5e"      # rose


def severity_label(score: int) -> str:
    if score <= 2:
        return "Clean"
    if score <= 4:
        return "Minor"
    if score <= 6:
        return "Moderate"
    if score <= 8:
        return "Significant"
    return "Severe"


def severity_css_class(score: int) -> str:
    if score <= 2:
        return "sev-low"
    if score <= 5:
        return "sev-med"
    if score <= 8:
        return "sev-high"
    return "sev-crit"


# ──────────────────────────────────────────────────────────────────────
# Plotly helpers
# ──────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    margin=dict(l=40, r=40, t=40, b=40),
)


def make_gauge(value: int, title: str, color: str) -> go.Figure:
    """Create a single radial gauge chart for a score 1-10."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(size=42, color=color, family="JetBrains Mono, monospace")),
        title=dict(text=title, font=dict(size=13, color="#94a3b8")),
        gauge=dict(
            axis=dict(range=[1, 10], tickwidth=0, tickcolor="rgba(0,0,0,0)",
                      tickfont=dict(size=0)),
            bar=dict(color=color, thickness=0.35),
            bgcolor="rgba(255,255,255,0.03)",
            borderwidth=0,
            steps=[
                dict(range=[1, 3], color="rgba(16,185,129,0.08)"),
                dict(range=[3, 5], color="rgba(6,182,212,0.06)"),
                dict(range=[5, 7], color="rgba(245,158,11,0.08)"),
                dict(range=[7, 9], color="rgba(239,68,68,0.08)"),
                dict(range=[9, 10], color="rgba(244,63,94,0.1)"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=value),
        ),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=200,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


def make_radar(scores: dict) -> go.Figure:
    """Radar chart of the four layer scores."""
    categories = ["Commission", "Omission", "Persistence", "Barometer"]
    values = [
        scores.get("commission_score", 1),
        scores.get("omission_score", 1),
        scores.get("correction_persistence_score", 1),
        scores.get("barometer_score", 1),
    ]
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    # Area fill
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(139,92,246,0.15)",
        line=dict(color="#8b5cf6", width=2),
        marker=dict(size=7, color="#8b5cf6"),
        name="Drift Profile",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 10], showticklabels=True,
                tickfont=dict(size=10, color="#64748b"),
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.05)",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#94a3b8", family="Inter"),
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.08)",
            ),
        ),
        showlegend=False,
        **PLOTLY_LAYOUT_DEFAULTS,
        height=340,
        margin=dict(l=60, r=60, t=30, b=30),
    )
    return fig


def make_timeline(report: AuditReport, turns: list[dict]) -> go.Figure:
    """Drift timeline: turn number on x-axis, severity on y, colour by type."""
    fig = go.Figure()

    # Commission flags
    if report.commission_flags:
        fig.add_trace(go.Scatter(
            x=[f.turn for f in report.commission_flags],
            y=[f.severity for f in report.commission_flags],
            mode="markers+text",
            marker=dict(size=12, color="#ef4444", symbol="diamond",
                        line=dict(width=1, color="rgba(239,68,68,0.4)")),
            text=["C"] * len(report.commission_flags),
            textposition="middle center",
            textfont=dict(size=8, color="white", family="JetBrains Mono"),
            name="Commission",
            hovertemplate="<b>Commission</b><br>Turn %{x}<br>Severity %{y}<br>%{customdata}<extra></extra>",
            customdata=[f.description[:60] for f in report.commission_flags],
        ))

    # Omission flags
    if report.omission_flags:
        fig.add_trace(go.Scatter(
            x=[f.turn for f in report.omission_flags],
            y=[f.severity for f in report.omission_flags],
            mode="markers+text",
            marker=dict(size=12, color="#f59e0b", symbol="square",
                        line=dict(width=1, color="rgba(245,158,11,0.4)")),
            text=["O"] * len(report.omission_flags),
            textposition="middle center",
            textfont=dict(size=8, color="white", family="JetBrains Mono"),
            name="Omission",
            hovertemplate="<b>Omission</b><br>Turn %{x}<br>Severity %{y}<br>%{customdata}<extra></extra>",
            customdata=[f.description[:60] for f in report.omission_flags],
        ))

    # Barometer signals (RED only to avoid clutter)
    red_signals = [s for s in report.barometer_signals if s.classification == "RED"]
    if red_signals:
        fig.add_trace(go.Scatter(
            x=[s.turn for s in red_signals],
            y=[s.severity for s in red_signals],
            mode="markers+text",
            marker=dict(size=12, color="#f43f5e", symbol="triangle-up",
                        line=dict(width=1, color="rgba(244,63,94,0.4)")),
            text=["B"] * len(red_signals),
            textposition="middle center",
            textfont=dict(size=8, color="white", family="JetBrains Mono"),
            name="Barometer RED",
            hovertemplate="<b>Barometer RED</b><br>Turn %{x}<br>Severity %{y}<br>%{customdata}<extra></extra>",
            customdata=[s.description[:60] for s in red_signals],
        ))

    # Correction failures
    failed = [e for e in report.correction_events if not e.held]
    if failed:
        fig.add_trace(go.Scatter(
            x=[e.failure_turn for e in failed if e.failure_turn],
            y=[8] * len([e for e in failed if e.failure_turn]),
            mode="markers+text",
            marker=dict(size=14, color="#c084fc", symbol="x",
                        line=dict(width=2, color="rgba(192,132,252,0.4)")),
            text=["P"] * len([e for e in failed if e.failure_turn]),
            textposition="middle center",
            textfont=dict(size=8, color="white", family="JetBrains Mono"),
            name="Persistence Fail",
            hovertemplate="<b>Correction Failure</b><br>Turn %{x}<extra></extra>",
        ))

    # Background bands for severity zones
    fig.add_hrect(y0=0, y1=3, fillcolor="rgba(16,185,129,0.04)", line_width=0)
    fig.add_hrect(y0=3, y1=6, fillcolor="rgba(245,158,11,0.04)", line_width=0)
    fig.add_hrect(y0=6, y1=10, fillcolor="rgba(239,68,68,0.04)", line_width=0)

    max_turn = max((t["turn"] for t in turns), default=10)
    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=320,
        xaxis=dict(
            title="Turn", range=[-0.5, max_turn + 0.5],
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.04)",
        ),
        yaxis=dict(
            title="Severity", range=[0, 10.5],
            gridcolor="rgba(255,255,255,0.04)",
            zerolinecolor="rgba(255,255,255,0.04)",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def make_barometer_bar(report: AuditReport) -> go.Figure:
    """Stacked horizontal bar showing GREEN/YELLOW/RED distribution."""
    total = len(report.barometer_signals) or 1
    g = sum(1 for s in report.barometer_signals if s.classification == "GREEN")
    y = sum(1 for s in report.barometer_signals if s.classification == "YELLOW")
    r = sum(1 for s in report.barometer_signals if s.classification == "RED")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Barometer"], x=[g], name="Green",
        orientation="h", marker_color="#10b981",
        text=[f"{g}"], textposition="inside",
        textfont=dict(size=13, family="JetBrains Mono", color="white"),
        hovertemplate="Green: %{x}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=["Barometer"], x=[y], name="Yellow",
        orientation="h", marker_color="#f59e0b",
        text=[f"{y}"], textposition="inside",
        textfont=dict(size=13, family="JetBrains Mono", color="white"),
        hovertemplate="Yellow: %{x}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=["Barometer"], x=[r], name="Red",
        orientation="h", marker_color="#ef4444",
        text=[f"{r}"], textposition="inside",
        textfont=dict(size=13, family="JetBrains Mono", color="white"),
        hovertemplate="Red: %{x}<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        **PLOTLY_LAYOUT_DEFAULTS,
        height=100,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.15,
                    xanchor="center", x=0.5, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def make_correction_timeline(report: AuditReport) -> go.Figure:
    """Gantt-style chart showing correction events and whether they held."""
    if not report.correction_events:
        return None

    fig = go.Figure()
    for i, ev in enumerate(report.correction_events):
        color = "#10b981" if ev.held else "#ef4444"
        end_turn = ev.failure_turn if ev.failure_turn else (report.total_turns - 1)
        label = "HELD" if ev.held else f"FAILED @ Turn {ev.failure_turn}"

        fig.add_trace(go.Scatter(
            x=[ev.correction_turn, ev.acknowledgment_turn, end_turn],
            y=[i, i, i],
            mode="lines+markers+text",
            line=dict(color=color, width=3),
            marker=dict(
                size=[10, 8, 12],
                color=[color, color, color],
                symbol=["circle", "diamond", "circle" if ev.held else "x"],
                line=dict(width=1, color="rgba(255,255,255,0.2)"),
            ),
            text=["Correction", "Ack", label],
            textposition=["top center", "top center", "top center"],
            textfont=dict(size=9, color="#94a3b8"),
            name=f"Event {i+1}",
            showlegend=False,
            hovertemplate=(
                f"Correction Turn: {ev.correction_turn}<br>"
                f"Ack Turn: {ev.acknowledgment_turn}<br>"
                f"Status: {label}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT_DEFAULTS,
        height=max(120, 60 * len(report.correction_events) + 60),
        xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=40),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# Render: metric card
# ──────────────────────────────────────────────────────────────────────

def render_metric_card(label: str, value, sub: str = "", accent: str = "blue"):
    color_map = {
        "blue": "#3b82f6", "red": "#ef4444", "amber": "#f59e0b",
        "green": "#10b981", "purple": "#8b5cf6", "cyan": "#06b6d4",
        "rose": "#f43f5e",
    }
    c = color_map.get(accent, "#3b82f6")
    st.markdown(f"""
    <div class="metric-card {accent}">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{c}">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Render: conversation replay
# ──────────────────────────────────────────────────────────────────────

def render_conversation_replay(turns: list[dict], report: AuditReport):
    """Show the original conversation with inline drift badges."""
    # Pre-compute flagged turns
    flagged_turns: dict[int, list[str]] = {}
    for f in report.commission_flags:
        flagged_turns.setdefault(f.turn, []).append(
            f'<span class="drift-badge commission">COMMISSION sev {f.severity}</span>'
        )
    for f in report.omission_flags:
        flagged_turns.setdefault(f.turn, []).append(
            f'<span class="drift-badge omission">OMISSION sev {f.severity}</span>'
        )
    for s in report.barometer_signals:
        if s.classification == "RED":
            flagged_turns.setdefault(s.turn, []).append(
                f'<span class="drift-badge barometer-red">BAROMETER RED</span>'
            )
        elif s.classification == "YELLOW":
            flagged_turns.setdefault(s.turn, []).append(
                f'<span class="drift-badge barometer-yellow">BAROMETER YELLOW</span>'
            )
    for ev in report.correction_events:
        if not ev.held and ev.failure_turn is not None:
            flagged_turns.setdefault(ev.failure_turn, []).append(
                f'<span class="drift-badge correction-fail">PERSISTENCE FAIL</span>'
            )

    for turn in turns:
        role = turn["role"]
        if role not in ("user", "assistant"):
            continue
        turn_num = turn["turn"]
        badges = " ".join(flagged_turns.get(turn_num, []))
        is_flagged = role == "assistant" and turn_num in flagged_turns
        flagged_class = " flagged" if is_flagged else ""
        content_escaped = turn["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        # Truncate long content for readability
        if len(content_escaped) > 1200:
            content_escaped = content_escaped[:1200] + "<br><em style='color:var(--text-muted)'>[truncated]</em>"
        st.markdown(f"""
        <div class="chat-turn {role}{flagged_class}">
            <div class="chat-role {role}">Turn {turn_num} &mdash; {role.upper()} {badges}</div>
            {content_escaped}
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Render: flag detail card
# ──────────────────────────────────────────────────────────────────────

def render_flag_card(flag, layer_color: str):
    """Render a single drift flag as an expandable card."""
    sev_class = severity_css_class(flag.severity)
    with st.expander(
        f"Turn {flag.turn}  |  Severity {flag.severity}/10  —  {flag.description[:70]}",
        expanded=False,
    ):
        st.markdown(f"""
        <div style="margin-bottom:0.5rem">
            <span class="{sev_class}" style="font-weight:700;font-size:1.1rem">
                Severity {flag.severity}/10
            </span>
            <span style="color:var(--text-muted);margin-left:1rem">
                {severity_label(flag.severity)}
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"**Description:** {flag.description}")
        if flag.instruction_ref:
            st.markdown(f"**Instruction:** {flag.instruction_ref}")
        if flag.evidence:
            st.markdown(f'<div class="evidence-block">{str(flag.evidence)[:500]}</div>',
                        unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem">
        <div style="font-size:2.5rem;margin-bottom:0.3rem">🔬</div>
        <div style="font-size:1.2rem;font-weight:700;
             background:linear-gradient(135deg,#3b82f6,#8b5cf6);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             background-clip:text;">Drift Auditor</div>
        <div style="font-size:0.75rem;color:#64748b;margin-top:0.2rem">
            Multi-Turn Drift Diagnostic
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Upload Chat")
    uploaded_chat = st.file_uploader(
        "Drop a Claude/ChatGPT export or plain text transcript",
        type=["txt", "json"],
        key="chat_upload",
        label_visibility="collapsed",
    )

    st.markdown("##### Optional Context")
    uploaded_system_prompt = st.file_uploader(
        "System prompt (optional)",
        type=["txt"],
        key="sys_prompt",
    )
    uploaded_preferences = st.file_uploader(
        "User preferences (optional)",
        type=["txt"],
        key="prefs",
    )

    st.markdown("---")
    st.markdown("##### Audit Parameters")
    window_size = st.slider("Window size (turns)", 10, 100, 50, 5)
    window_overlap = st.slider("Window overlap", 0, 30, 10, 2)
    conversation_id = st.text_input("Conversation ID", value="uploaded_chat")

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;font-size:0.7rem;color:#475569">'
        'Built for Claude Opus 4.6 Hackathon<br>'
        'Author: George Abrahamyan'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div style="margin-bottom:0.5rem">
    <div class="hero-title">Drift Auditor</div>
    <div class="hero-subtitle">
        Multi-turn drift diagnostics for LLM conversations &mdash;
        commission, omission, correction persistence &amp; structural barometer analysis.
    </div>
</div>
""", unsafe_allow_html=True)

if not uploaded_chat:
    # ---------- Landing state ----------
    st.markdown("""
    <div class="glass-card" style="text-align:center;padding:3rem 2rem">
        <div style="font-size:3.5rem;margin-bottom:1rem">📂</div>
        <div style="font-size:1.25rem;font-weight:600;color:var(--text-primary);margin-bottom:0.5rem">
            Upload a conversation to begin
        </div>
        <div style="color:var(--text-secondary);max-width:500px;margin:0 auto;line-height:1.7">
            Drop a <strong>Claude.ai JSON export</strong>, <strong>ChatGPT export</strong>,
            or <strong>plain-text transcript</strong> (Human: / Assistant:) into the sidebar.
            <br><br>
            Optionally attach a system prompt and user preferences for deeper instruction-adherence analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show architecture overview
    st.markdown('<div class="section-header"><span class="icon">🏗️</span> Detection Layers</div>',
                unsafe_allow_html=True)
    layer_cols = st.columns(4)
    with layer_cols[0]:
        st.markdown("""
        <div class="glass-card" style="min-height:200px">
            <div style="font-size:1.5rem;margin-bottom:0.5rem">🚩</div>
            <div style="font-weight:700;color:#ef4444;margin-bottom:0.4rem">Layer 1 — Commission</div>
            <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.6">
                Detects sycophancy markers, reality distortion, and unwarranted confidence.
                Context gates suppress false positives in correction acknowledgments.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with layer_cols[1]:
        st.markdown("""
        <div class="glass-card" style="min-height:200px">
            <div style="font-size:1.5rem;margin-bottom:0.5rem">📜</div>
            <div style="font-weight:700;color:#f59e0b;margin-bottom:0.4rem">Layer 2 — Omission</div>
            <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.6">
                Checks instruction adherence via keyword heuristics. Flags prohibition
                violations and absent required behaviours. Enhanced by barometer signals.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with layer_cols[2]:
        st.markdown("""
        <div class="glass-card" style="min-height:200px">
            <div style="font-size:1.5rem;margin-bottom:0.5rem">🔄</div>
            <div style="font-weight:700;color:#c084fc;margin-bottom:0.4rem">Layer 3 — Persistence</div>
            <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.6">
                Tracks correction events. Did the model acknowledge a fix and then
                regress? Topic signatures prevent false matches.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with layer_cols[3]:
        st.markdown("""
        <div class="glass-card" style="min-height:200px">
            <div style="font-size:1.5rem;margin-bottom:0.5rem">🧠</div>
            <div style="font-weight:700;color:#06b6d4;margin-bottom:0.4rem">Layer 4 — Barometer</div>
            <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.6">
                Structural drift barometer. Classifies epistemic posture per assistant
                turn: GREEN (healthy), YELLOW (weak), RED (drifted).
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ──────────────────────────────────────────────────────────────────────
# Run the audit
# ──────────────────────────────────────────────────────────────────────

raw_text = uploaded_chat.read().decode("utf-8", errors="replace")
system_prompt = ""
user_preferences = ""
if uploaded_system_prompt:
    system_prompt = uploaded_system_prompt.read().decode("utf-8", errors="replace")
if uploaded_preferences:
    user_preferences = uploaded_preferences.read().decode("utf-8", errors="replace")

with st.spinner("Running multi-layer drift audit..."):
    report = audit_conversation(
        raw_text=raw_text,
        system_prompt=system_prompt,
        user_preferences=user_preferences,
        window_size=window_size,
        overlap=window_overlap,
        conversation_id=conversation_id,
    )
    turns = parse_chat_log(raw_text)

scores = report.summary_scores

# ──────────────────────────────────────────────────────────────────────
# Executive Summary — Metric Cards
# ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><span class="icon">📊</span> Executive Summary</div>',
            unsafe_allow_html=True)

# Stats chip row
total_flags = (scores.get("commission_flag_count", 0) +
               scores.get("omission_flag_count", 0))
st.markdown(f"""
<div class="stats-row">
    <div class="stat-chip"><strong>{report.total_turns}</strong> turns parsed</div>
    <div class="stat-chip"><strong>{report.instructions_extracted}</strong> instructions extracted</div>
    <div class="stat-chip"><strong>{total_flags}</strong> drift flags raised</div>
    <div class="stat-chip"><strong>{scores.get("correction_events_total", 0)}</strong> correction events</div>
    <div class="stat-chip"><strong>{scores.get("corrections_failed", 0)}</strong> corrections failed</div>
</div>
""", unsafe_allow_html=True)

# Top-level metric cards
mc = st.columns(5)
with mc[0]:
    overall = scores.get("overall_drift_score", 1)
    accent = "green" if overall <= 3 else ("amber" if overall <= 6 else "red")
    render_metric_card("Overall Drift", f"{overall}/10", severity_label(overall), accent)
with mc[1]:
    v = scores.get("commission_score", 1)
    render_metric_card("Commission", f"{v}/10",
                       f"{scores.get('commission_flag_count', 0)} flags", "red")
with mc[2]:
    v = scores.get("omission_score", 1)
    render_metric_card("Omission", f"{v}/10",
                       f"{scores.get('omission_flag_count', 0)} flags", "amber")
with mc[3]:
    v = scores.get("correction_persistence_score", 1)
    render_metric_card("Persistence", f"{v}/10",
                       f"{scores.get('corrections_failed', 0)}/{scores.get('correction_events_total', 0)} failed",
                       "purple")
with mc[4]:
    v = scores.get("barometer_score", 1)
    render_metric_card("Barometer", f"{v}/10",
                       f"{scores.get('barometer_red_count', 0)} red signals", "cyan")

# ──────────────────────────────────────────────────────────────────────
# Gauges + Radar
# ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><span class="icon">🎯</span> Drift Profile</div>',
            unsafe_allow_html=True)

gcol1, gcol2 = st.columns([3, 2])
with gcol1:
    gauge_cols = st.columns(4)
    layers = [
        ("Commission", "commission_score", "#ef4444"),
        ("Omission", "omission_score", "#f59e0b"),
        ("Persistence", "correction_persistence_score", "#c084fc"),
        ("Barometer", "barometer_score", "#06b6d4"),
    ]
    for col, (name, key, color) in zip(gauge_cols, layers):
        with col:
            st.plotly_chart(
                make_gauge(scores.get(key, 1), name, color),
                use_container_width=True,
                config={"displayModeBar": False},
            )
with gcol2:
    st.plotly_chart(
        make_radar(scores),
        use_container_width=True,
        config={"displayModeBar": False},
    )


# ──────────────────────────────────────────────────────────────────────
# Drift Timeline
# ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><span class="icon">📈</span> Drift Timeline</div>',
            unsafe_allow_html=True)
st.plotly_chart(
    make_timeline(report, turns),
    use_container_width=True,
    config={"displayModeBar": False},
)


# ──────────────────────────────────────────────────────────────────────
# Detailed Tabs
# ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><span class="icon">🔍</span> Layer Details</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚩 Commission",
    "📜 Omission",
    "🔄 Persistence",
    "🧠 Barometer",
    "💬 Conversation Replay",
])

# --- Tab 1: Commission ---
with tab1:
    if report.commission_flags:
        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">
                <span class="pulse-dot red"></span>
                <span style="font-weight:700;font-size:1.05rem">
                    {len(report.commission_flags)} Commission Flags
                </span>
            </div>
            <div style="color:var(--text-secondary);font-size:0.85rem">
                Sycophantic agreement, reality distortion, and unwarranted confidence detected
                in assistant responses.
            </div>
        </div>
        """, unsafe_allow_html=True)
        for f in sorted(report.commission_flags, key=lambda x: x.turn):
            render_flag_card(f, "#ef4444")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2rem">
            <span class="pulse-dot green"></span>
            <span style="font-weight:600;color:var(--accent-green)">
                No commission drift detected
            </span>
        </div>
        """, unsafe_allow_html=True)

# --- Tab 2: Omission ---
with tab2:
    if report.omission_flags:
        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">
                <span class="pulse-dot amber"></span>
                <span style="font-weight:700;font-size:1.05rem">
                    {len(report.omission_flags)} Omission Flags
                </span>
            </div>
            <div style="color:var(--text-secondary);font-size:0.85rem">
                Instruction violations and absent required behaviours. Enhanced by barometer signal integration.
            </div>
        </div>
        """, unsafe_allow_html=True)
        for f in sorted(report.omission_flags, key=lambda x: x.turn):
            render_flag_card(f, "#f59e0b")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2rem">
            <span class="pulse-dot green"></span>
            <span style="font-weight:600;color:var(--accent-green)">
                No omission drift detected (local heuristics)
            </span>
            <div style="color:var(--text-muted);font-size:0.8rem;margin-top:0.3rem">
                Full semantic detection requires API-powered analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Tab 3: Persistence ---
with tab3:
    if report.correction_events:
        held = sum(1 for e in report.correction_events if e.held)
        failed = sum(1 for e in report.correction_events if not e.held)
        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem">
                <span class="pulse-dot {'red' if failed else 'green'}"></span>
                <span style="font-weight:700;font-size:1.05rem">
                    {len(report.correction_events)} Correction Events —
                    <span style="color:var(--accent-green)">{held} held</span>,
                    <span style="color:var(--accent-red)">{failed} failed</span>
                </span>
            </div>
            <div style="color:var(--text-secondary);font-size:0.85rem">
                Tracks whether user corrections were acknowledged and actually persisted.
                Failure means the model apologised but reverted to the same behaviour.
            </div>
        </div>
        """, unsafe_allow_html=True)

        corr_fig = make_correction_timeline(report)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True,
                            config={"displayModeBar": False})

        for ev in report.correction_events:
            status = "✅ HELD" if ev.held else f"❌ FAILED at turn {ev.failure_turn}"
            status_color = "var(--accent-green)" if ev.held else "var(--accent-red)"
            with st.expander(
                f"Turn {ev.correction_turn} → Ack {ev.acknowledgment_turn}  |  {status}",
                expanded=not ev.held,
            ):
                st.markdown(f"""
                <div style="margin-bottom:0.5rem">
                    <span style="color:{status_color};font-weight:700">{status}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Correction context:** {ev.instruction[:300]}")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2rem">
            <span style="font-size:1.5rem">🤷</span><br>
            <span style="font-weight:600;color:var(--text-secondary)">
                No correction events detected in this conversation.
            </span>
        </div>
        """, unsafe_allow_html=True)

# --- Tab 4: Barometer ---
with tab4:
    if report.barometer_signals:
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-weight:700;font-size:1.05rem;margin-bottom:0.5rem">
                Epistemic Posture Distribution
            </div>
            <div style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:0.8rem">
                Each assistant turn classified by structural drift barometer:
                <strong style="color:#10b981">GREEN</strong> (healthy),
                <strong style="color:#f59e0b">YELLOW</strong> (weak hedging),
                <strong style="color:#ef4444">RED</strong> (drifted).
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(
            make_barometer_bar(report),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Per-turn barometer heatmap
        baro_data = []
        for s in report.barometer_signals:
            baro_data.append({
                "Turn": s.turn,
                "Severity": s.severity,
                "Classification": s.classification,
                "Description": s.description[:80],
            })
        if baro_data:
            df = pd.DataFrame(baro_data)
            color_map = {"GREEN": "#10b981", "YELLOW": "#f59e0b", "RED": "#ef4444"}
            fig = go.Figure()
            for cls, color in color_map.items():
                subset = df[df["Classification"] == cls]
                if not subset.empty:
                    fig.add_trace(go.Bar(
                        x=subset["Turn"],
                        y=subset["Severity"],
                        marker_color=color,
                        name=cls,
                        hovertemplate=(
                            "<b>%{customdata}</b><br>Turn %{x}<br>Severity %{y}<extra></extra>"
                        ),
                        customdata=subset["Description"],
                    ))
            fig.update_layout(
                **PLOTLY_LAYOUT_DEFAULTS,
                height=260,
                barmode="overlay",
                xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Severity", range=[0, 10.5],
                           gridcolor="rgba(255,255,255,0.04)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5, font=dict(size=11),
                            bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=50, r=20, t=40, b=50),
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

        # Expandable detail for RED signals
        red_signals = [s for s in report.barometer_signals if s.classification == "RED"]
        if red_signals:
            st.markdown(f"""
            <div style="font-weight:600;color:var(--accent-red);margin:1rem 0 0.5rem">
                🚨 {len(red_signals)} RED Structural Drift Signals
            </div>
            """, unsafe_allow_html=True)
            for s in red_signals:
                with st.expander(f"Turn {s.turn}  |  Severity {s.severity}/10  —  {s.description[:60]}"):
                    st.markdown(f"**Classification:** RED")
                    st.markdown(f"**Description:** {s.description}")
                    if s.evidence:
                        st.markdown(f'<div class="evidence-block">{s.evidence[:500]}</div>',
                                    unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2rem">
            <span style="font-weight:600;color:var(--text-secondary)">
                No barometer signals (no assistant turns detected).
            </span>
        </div>
        """, unsafe_allow_html=True)

# --- Tab 5: Conversation Replay ---
with tab5:
    st.markdown("""
    <div class="glass-card">
        <div style="font-weight:700;font-size:1.05rem;margin-bottom:0.5rem">
            Annotated Conversation
        </div>
        <div style="color:var(--text-secondary);font-size:0.85rem">
            The original conversation with inline drift annotations.
            Flagged assistant turns are highlighted with a red border.
        </div>
    </div>
    """, unsafe_allow_html=True)
    render_conversation_replay(turns, report)


# ──────────────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"><span class="icon">💾</span> Export</div>',
            unsafe_allow_html=True)

exp_col1, exp_col2, exp_col3 = st.columns(3)
with exp_col1:
    json_str = report_to_json(report)
    st.download_button(
        "Download JSON Report",
        data=json_str,
        file_name=f"drift_audit_{conversation_id}.json",
        mime="application/json",
        use_container_width=True,
    )
with exp_col2:
    text_str = format_report(report)
    st.download_button(
        "Download Text Report",
        data=text_str,
        file_name=f"drift_audit_{conversation_id}.txt",
        mime="text/plain",
        use_container_width=True,
    )
with exp_col3:
    # Summary data as CSV
    summary_rows = []
    for k, v in scores.items():
        summary_rows.append({"metric": k, "value": v})
    csv_str = pd.DataFrame(summary_rows).to_csv(index=False)
    st.download_button(
        "Download Scores CSV",
        data=csv_str,
        file_name=f"drift_scores_{conversation_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )
