"""
Drift Auditor - Competition-Grade Streamlit Dashboard
=====================================================
A visually stunning, interactive interface for multi-turn drift analysis.

Features:
  - Glassmorphism dark theme with animated neon accents
  - Animated radial gauge scores
  - Interactive Plotly timeline, heatmap, radar, and bar charts
  - Conversation replay with inline drift highlights
  - Collapsible evidence panels with severity badges
  - One-click sample data loading for instant demo
  - JSON/text report export
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import math
import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))
from drift_auditor import (
    audit_conversation,
    format_report,
    report_to_json,
    AuditReport,
    parse_chat_log,
)

# ---------------------------------------------------------------------------
# Page config  (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drift Auditor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS: dark glassmorphism theme with neon accents + animations
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---------- global dark bg ---------- */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 40%, #111827 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.92);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}

/* ---------- glass card ---------- */
.glass-card {
    background: rgba(30, 41, 59, 0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35),
                inset 0 1px 0 rgba(255, 255, 255, 0.04);
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.18),
                inset 0 1px 0 rgba(255, 255, 255, 0.06);
}

/* ---------- metric cards ---------- */
.metric-card {
    background: rgba(30, 41, 59, 0.60);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: 14px;
    padding: 1.1rem 1.2rem 0.9rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent, #6366f1), transparent);
    border-radius: 14px 14px 0 0;
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94a3b8;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent, #6366f1), var(--accent2, #a78bfa));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-sub {
    font-size: 0.68rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* accent colour variants */
.mc-indigo  { --accent: #6366f1; --accent2: #a78bfa; }
.mc-rose    { --accent: #f43f5e; --accent2: #fb7185; }
.mc-amber   { --accent: #f59e0b; --accent2: #fbbf24; }
.mc-emerald { --accent: #10b981; --accent2: #34d399; }
.mc-cyan    { --accent: #06b6d4; --accent2: #22d3ee; }

/* ---------- severity badges ---------- */
.badge {font-size:0.68rem;padding:0.2rem 0.55rem;border-radius:99px;font-weight:700;display:inline-block;letter-spacing:0.04em;}
.badge-red    {background:rgba(239,68,68,0.18);color:#f87171;border:1px solid rgba(239,68,68,0.35);}
.badge-yellow {background:rgba(234,179,8,0.15);color:#facc15;border:1px solid rgba(234,179,8,0.30);}
.badge-green  {background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.30);}
.badge-blue   {background:rgba(59,130,246,0.15);color:#60a5fa;border:1px solid rgba(59,130,246,0.30);}

/* ---------- conversation bubbles ---------- */
.chat-turn {
    padding: 0.85rem 1.1rem;
    border-radius: 12px;
    margin-bottom: 0.55rem;
    font-size: 0.88rem;
    line-height: 1.55;
    position: relative;
    max-width: 92%;
}
.chat-user {
    background: rgba(99, 102, 241, 0.10);
    border: 1px solid rgba(99, 102, 241, 0.20);
    margin-left: auto;
    border-bottom-right-radius: 3px;
}
.chat-assistant {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid rgba(71, 85, 105, 0.25);
    border-bottom-left-radius: 3px;
}
.chat-system {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.18);
    text-align: center;
    font-style: italic;
    max-width: 100%;
}
.chat-role {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 0.25rem;
}
.drift-marker {
    position: absolute;
    top: 6px; right: 8px;
    width: 10px; height: 10px;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
    70%  { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}

/* ---------- section headers ---------- */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    margin: 1.5rem 0 0.7rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.18);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header .icon {
    font-size: 1.25rem;
}

/* ---------- misc overrides ---------- */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.35);
    border: 1px dashed rgba(99, 102, 241, 0.30);
    border-radius: 14px;
    padding: 0.8rem;
}
div[data-testid="stExpander"] {
    background: rgba(30, 41, 59, 0.40);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 12px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(30, 41, 59, 0.50);
    border-radius: 8px 8px 0 0;
    border: 1px solid rgba(99, 102, 241, 0.12);
    color: #94a3b8;
    padding: 0.5rem 1.1rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.15) !important;
    color: #e2e8f0 !important;
    border-color: rgba(99, 102, 241, 0.35) !important;
}

/* plotly bg transparent */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* hide default streamlit metric delta */
[data-testid="stMetric"] { display: none; }

/* hero heading */
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #c7d2fe 0%, #6366f1 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.15rem;
}
.hero-sub {
    color: #64748b;
    font-size: 0.92rem;
    margin-bottom: 1.2rem;
}

/* animate fade-in */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeInUp 0.5s ease both; }
.fade-in-d1 { animation: fadeInUp 0.5s ease 0.08s both; }
.fade-in-d2 { animation: fadeInUp 0.5s ease 0.16s both; }
.fade-in-d3 { animation: fadeInUp 0.5s ease 0.24s both; }
.fade-in-d4 { animation: fadeInUp 0.5s ease 0.32s both; }

/* scrollable conversation container */
.conv-container {
    max-height: 550px;
    overflow-y: auto;
    padding-right: 6px;
}
.conv-container::-webkit-scrollbar { width: 5px; }
.conv-container::-webkit-scrollbar-track { background: transparent; }
.conv-container::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.25); border-radius: 4px; }

/* empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #475569;
}
.empty-state .icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-state h3 { color: #94a3b8; font-weight: 600; margin-bottom: 0.5rem; }
.empty-state p { font-size: 0.88rem; max-width: 440px; margin: 0 auto; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly shared layout defaults
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#cbd5e1", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(
        bgcolor="rgba(30,41,59,0.92)",
        bordercolor="rgba(99,102,241,0.35)",
        font_color="#e2e8f0",
    ),
)

SEVERITY_COLORS = {
    1: "#22c55e", 2: "#4ade80", 3: "#a3e635",
    4: "#facc15", 5: "#fbbf24", 6: "#f59e0b",
    7: "#f97316", 8: "#ef4444", 9: "#dc2626", 10: "#b91c1c",
}

def severity_color(s: int) -> str:
    return SEVERITY_COLORS.get(max(1, min(10, s)), "#6366f1")


# ---------------------------------------------------------------------------
# Helper: build gauge chart
# ---------------------------------------------------------------------------
def make_gauge(value: int, title: str, color: str) -> go.Figure:
    """Create an animated-look radial gauge 1-10."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(font=dict(size=38, color=color, family="Inter, system-ui")),
        title=dict(text=title, font=dict(size=12, color="#94a3b8")),
        gauge=dict(
            axis=dict(range=[1, 10], tickwidth=1, tickcolor="rgba(148,163,184,0.2)",
                      dtick=1, tickfont=dict(size=9, color="#64748b")),
            bar=dict(color=color, thickness=0.3),
            bgcolor="rgba(30,41,59,0.4)",
            borderwidth=0,
            steps=[
                dict(range=[1, 3], color="rgba(34,197,94,0.08)"),
                dict(range=[3, 6], color="rgba(250,204,21,0.06)"),
                dict(range=[6, 10], color="rgba(239,68,68,0.06)"),
            ],
            threshold=dict(
                line=dict(color="#f87171", width=2),
                thickness=0.8,
                value=value,
            ),
        ),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=195,
        margin=dict(l=20, r=20, t=45, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: radar chart
# ---------------------------------------------------------------------------
def make_radar(scores: dict) -> go.Figure:
    categories = ["Commission", "Omission", "Persistence", "Barometer"]
    values = [
        scores.get("commission_score", 1),
        scores.get("omission_score", 1),
        scores.get("correction_persistence_score", 1),
        scores.get("barometer_score", 1),
    ]
    values_closed = values + [values[0]]
    cats_closed = categories + [categories[0]]

    fig = go.Figure()
    # filled area
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=cats_closed,
        fill="toself",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=7, color="#a78bfa"),
        name="Drift Profile",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 10],
                gridcolor="rgba(148,163,184,0.08)",
                linecolor="rgba(148,163,184,0.08)",
                tickfont=dict(size=9, color="#64748b"),
            ),
            angularaxis=dict(
                gridcolor="rgba(148,163,184,0.10)",
                linecolor="rgba(148,163,184,0.10)",
                tickfont=dict(size=11, color="#94a3b8"),
            ),
        ),
        showlegend=False,
        height=340,
        margin=dict(l=60, r=60, t=30, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: drift timeline
# ---------------------------------------------------------------------------
def make_timeline(report: AuditReport) -> go.Figure:
    events = []
    for f in report.commission_flags:
        events.append(dict(turn=f.turn, severity=f.severity, layer="Commission",
                           desc=f.description, color="#f43f5e"))
    for f in report.omission_flags:
        events.append(dict(turn=f.turn, severity=f.severity, layer="Omission",
                           desc=f.description, color="#f59e0b"))
    for e in report.correction_events:
        sev = 8 if not e.held else 3
        events.append(dict(turn=e.correction_turn, severity=sev,
                           layer="Correction",
                           desc=f"{'FAILED' if not e.held else 'HELD'}: {e.instruction[:60]}",
                           color="#06b6d4"))
    for s in report.barometer_signals:
        if s.classification in ("RED", "YELLOW"):
            events.append(dict(turn=s.turn, severity=s.severity,
                               layer="Barometer", desc=s.description,
                               color="#a78bfa" if s.classification == "YELLOW" else "#ef4444"))

    if not events:
        fig = go.Figure()
        fig.add_annotation(text="No drift events detected", showarrow=False,
                           font=dict(size=14, color="#64748b"), xref="paper", yref="paper",
                           x=0.5, y=0.5)
        fig.update_layout(**PLOTLY_LAYOUT, height=280)
        return fig

    df = pd.DataFrame(events)

    fig = go.Figure()
    layer_order = ["Commission", "Omission", "Correction", "Barometer"]
    layer_colors = {"Commission": "#f43f5e", "Omission": "#f59e0b",
                    "Correction": "#06b6d4", "Barometer": "#a78bfa"}

    for layer in layer_order:
        sub = df[df["layer"] == layer]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["turn"], y=sub["severity"],
            mode="markers+lines",
            name=layer,
            marker=dict(
                size=sub["severity"] * 1.5 + 4,
                color=layer_colors[layer],
                opacity=0.85,
                line=dict(width=1, color="rgba(255,255,255,0.15)"),
            ),
            line=dict(color=layer_colors[layer], width=1.5, dash="dot"),
            text=sub["desc"],
            hovertemplate="<b>Turn %{x}</b><br>Severity: %{y}<br>%{text}<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        xaxis=dict(title="Turn", gridcolor="rgba(148,163,184,0.06)",
                   zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(title="Severity", range=[0, 11],
                   gridcolor="rgba(148,163,184,0.06)", zeroline=False,
                   tickfont=dict(size=10)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(size=11)),
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: barometer stacked bar
# ---------------------------------------------------------------------------
def make_barometer_chart(report: AuditReport) -> go.Figure:
    signals = report.barometer_signals
    if not signals:
        fig = go.Figure()
        fig.add_annotation(text="No barometer signals", showarrow=False,
                           font=dict(size=14, color="#64748b"), xref="paper", yref="paper",
                           x=0.5, y=0.5)
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        return fig

    turns = [s.turn for s in signals]
    severities = [s.severity for s in signals]
    classifications = [s.classification for s in signals]
    colors = [{"RED": "#ef4444", "YELLOW": "#facc15", "GREEN": "#22c55e"}[c] for c in classifications]

    fig = go.Figure(go.Bar(
        x=turns, y=severities,
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(width=0.5, color="rgba(255,255,255,0.08)"),
        ),
        text=classifications,
        hovertemplate="<b>Turn %{x}</b><br>Classification: %{text}<br>Severity: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        xaxis=dict(title="Assistant Turn", gridcolor="rgba(148,163,184,0.06)",
                   tickfont=dict(size=10)),
        yaxis=dict(title="Severity", range=[0, 11],
                   gridcolor="rgba(148,163,184,0.06)", tickfont=dict(size=10)),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: heatmap (turns x layers)
# ---------------------------------------------------------------------------
def make_heatmap(report: AuditReport) -> go.Figure:
    max_turn = report.total_turns
    if max_turn == 0:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=220)
        return fig

    layers = ["Commission", "Omission", "Persistence", "Barometer"]
    matrix = [[0]*max_turn for _ in range(4)]

    for f in report.commission_flags:
        if 0 <= f.turn < max_turn:
            matrix[0][f.turn] = max(matrix[0][f.turn], f.severity)
    for f in report.omission_flags:
        if 0 <= f.turn < max_turn:
            matrix[1][f.turn] = max(matrix[1][f.turn], f.severity)
    for e in report.correction_events:
        t = e.failure_turn if (not e.held and e.failure_turn is not None) else e.correction_turn
        if 0 <= t < max_turn:
            matrix[2][t] = max(matrix[2][t], 8 if not e.held else 3)
    for s in report.barometer_signals:
        if 0 <= s.turn < max_turn:
            matrix[3][s.turn] = max(matrix[3][s.turn], s.severity)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=list(range(max_turn)),
        y=layers,
        colorscale=[
            [0, "rgba(30,41,59,0.5)"],
            [0.1, "#164e63"],
            [0.3, "#22c55e"],
            [0.5, "#facc15"],
            [0.7, "#f97316"],
            [1.0, "#ef4444"],
        ],
        zmin=0, zmax=10,
        colorbar=dict(
            title="Sev",
            tickfont=dict(color="#94a3b8", size=10),
            titlefont=dict(color="#94a3b8", size=10),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        hovertemplate="Turn %{x}<br>Layer: %{y}<br>Severity: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=240,
        xaxis=dict(title="Turn", tickfont=dict(size=10),
                   gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(tickfont=dict(size=11),
                   gridcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: correction persistence Gantt-like chart
# ---------------------------------------------------------------------------
def make_persistence_chart(report: AuditReport) -> go.Figure:
    events = report.correction_events
    if not events:
        fig = go.Figure()
        fig.add_annotation(text="No correction events to display", showarrow=False,
                           font=dict(size=14, color="#64748b"), xref="paper", yref="paper",
                           x=0.5, y=0.5)
        fig.update_layout(**PLOTLY_LAYOUT, height=260)
        return fig

    fig = go.Figure()
    for i, ev in enumerate(events):
        color = "#ef4444" if not ev.held else "#22c55e"
        end_turn = ev.failure_turn if (not ev.held and ev.failure_turn is not None) else report.total_turns
        label = ev.instruction[:50] + ("..." if len(ev.instruction) > 50 else "")
        # line from correction to end/failure
        fig.add_trace(go.Scatter(
            x=[ev.correction_turn, ev.acknowledgment_turn, end_turn],
            y=[i, i, i],
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=[10, 8, 12],
                        color=[color, "#60a5fa", color],
                        symbol=["circle", "diamond", "x" if not ev.held else "circle"],
                        line=dict(width=1, color="rgba(255,255,255,0.2)")),
            name=f"{'FAIL' if not ev.held else 'HELD'}: {label}",
            hovertemplate=(
                f"<b>{'FAILED' if not ev.held else 'HELD'}</b><br>"
                f"Correction: Turn {ev.correction_turn}<br>"
                f"Ack: Turn {ev.acknowledgment_turn}<br>"
                f"{'Failed at: Turn ' + str(ev.failure_turn) if not ev.held else 'Persisted'}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(180, 60 * len(events) + 80),
        xaxis=dict(title="Turn", gridcolor="rgba(148,163,184,0.06)",
                   tickfont=dict(size=10)),
        yaxis=dict(
            tickvals=list(range(len(events))),
            ticktext=[f"Event {i+1}" for i in range(len(events))],
            tickfont=dict(size=10),
            gridcolor="rgba(148,163,184,0.04)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(size=10)),
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: donut chart for barometer distribution
# ---------------------------------------------------------------------------
def make_barometer_donut(report: AuditReport) -> go.Figure:
    scores = report.summary_scores
    r = scores.get("barometer_red_count", 0)
    y = scores.get("barometer_yellow_count", 0)
    g = scores.get("barometer_green_count", 0)
    total = r + y + g
    if total == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False,
                           font=dict(size=14, color="#64748b"),
                           xref="paper", yref="paper", x=0.5, y=0.5)
        fig.update_layout(**PLOTLY_LAYOUT, height=260)
        return fig

    fig = go.Figure(go.Pie(
        values=[r, y, g],
        labels=["RED", "YELLOW", "GREEN"],
        marker=dict(colors=["#ef4444", "#facc15", "#22c55e"],
                    line=dict(color="rgba(0,0,0,0.3)", width=2)),
        hole=0.55,
        textinfo="label+percent",
        textfont=dict(size=12, color="#e2e8f0"),
        hovertemplate="%{label}: %{value} signals (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=260,
        showlegend=False,
        annotations=[dict(
            text=f"<b>{total}</b><br><span style='font-size:10px;color:#94a3b8'>signals</span>",
            x=0.5, y=0.5, font=dict(size=22, color="#e2e8f0"),
            showarrow=False,
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# Helper: render a metric card (HTML)
# ---------------------------------------------------------------------------
def metric_card_html(label: str, value, sub: str = "", accent_class: str = "mc-indigo") -> str:
    return f"""
    <div class="metric-card {accent_class} fade-in">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Helper: render conversation with drift highlights
# ---------------------------------------------------------------------------
def render_conversation(report: AuditReport, turns: list[dict]):
    # build lookup of drift turns
    drift_turns: dict[int, list[str]] = {}
    for f in report.commission_flags:
        drift_turns.setdefault(f.turn, []).append(
            f'<span class="badge badge-red">Commission sev {f.severity}</span>')
    for f in report.omission_flags:
        drift_turns.setdefault(f.turn, []).append(
            f'<span class="badge badge-yellow">Omission sev {f.severity}</span>')
    for s in report.barometer_signals:
        if s.classification == "RED":
            drift_turns.setdefault(s.turn, []).append(
                f'<span class="badge badge-red">Barometer RED</span>')
        elif s.classification == "YELLOW":
            drift_turns.setdefault(s.turn, []).append(
                f'<span class="badge badge-yellow">Barometer YELLOW</span>')
    for e in report.correction_events:
        if not e.held and e.failure_turn is not None:
            drift_turns.setdefault(e.failure_turn, []).append(
                f'<span class="badge badge-red">Correction FAILED</span>')

    html_parts = ['<div class="conv-container">']
    for t in turns:
        role = t["role"]
        content = t["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        css_class = f"chat-{role}" if role in ("user", "assistant", "system") else "chat-assistant"
        role_display = role.upper()
        badges = drift_turns.get(t["turn"], [])
        marker = ""
        if any("badge-red" in b for b in badges):
            marker = '<div class="drift-marker" style="background:#ef4444;"></div>'
        elif any("badge-yellow" in b for b in badges):
            marker = '<div class="drift-marker" style="background:#facc15;animation:none;"></div>'

        badge_html = " ".join(badges)
        html_parts.append(f"""
        <div class="chat-turn {css_class}">
            {marker}
            <div class="chat-role">{role_display} &middot; Turn {t['turn']} {badge_html}</div>
            {content}
        </div>
        """)
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load sample data helper
# ---------------------------------------------------------------------------
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")

def load_sample() -> tuple[str, str]:
    conv_path = os.path.join(SAMPLE_DIR, "sample_conversation.txt")
    sp_path = os.path.join(SAMPLE_DIR, "sample_system_prompt.txt")
    conv = ""
    sp = ""
    if os.path.exists(conv_path):
        with open(conv_path, "r", encoding="utf-8") as f:
            conv = f.read()
    if os.path.exists(sp_path):
        with open(sp_path, "r", encoding="utf-8") as f:
            sp = f.read()
    return conv, sp


# ===================================================================
#  MAIN APP
# ===================================================================

# --- Hero header ---
st.markdown("""
<div class="fade-in">
    <div class="hero-title">Drift Auditor</div>
    <div class="hero-sub">Multi-turn drift diagnostic &mdash; correction persistence &amp; structural barometer analysis</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.8rem 0 0.4rem;">
        <span style="font-size:2.2rem;">🔬</span>
        <div style="font-size:1.05rem;font-weight:700;color:#c7d2fe;margin-top:0.3rem;">Drift Auditor</div>
        <div style="font-size:0.72rem;color:#64748b;margin-top:0.15rem;">v2.0 &mdash; Hackathon Edition</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("##### Upload Conversation")
    uploaded = st.file_uploader("Drop a chat export file", type=["txt", "json"],
                                label_visibility="collapsed")

    st.markdown("##### System Prompt (optional)")
    system_prompt_text = st.text_area("System prompt", height=90, label_visibility="collapsed",
                                      placeholder="Paste the system prompt used in the conversation...")

    st.markdown("---")
    st.markdown("##### Audit Parameters")
    window_size = st.slider("Window size (turns)", 10, 100, 50, step=5)
    overlap = st.slider("Window overlap", 0, 30, 10, step=2)

    st.markdown("---")
    use_sample = st.button("Load Sample Conversation", use_container_width=True, type="primary")

# --- State management ---
if "report" not in st.session_state:
    st.session_state.report = None
    st.session_state.raw_text = None
    st.session_state.turns = None
    st.session_state.system_prompt = ""

# Handle sample loading
if use_sample:
    sample_conv, sample_sp = load_sample()
    if sample_conv:
        st.session_state.raw_text = sample_conv
        st.session_state.system_prompt = sample_sp
        st.session_state.report = audit_conversation(
            raw_text=sample_conv,
            system_prompt=sample_sp,
            window_size=window_size,
            overlap=overlap,
            conversation_id="sample_conversation",
        )
        st.session_state.turns = parse_chat_log(sample_conv)
    else:
        st.warning("Sample files not found in examples/ directory.")

# Handle file upload
if uploaded is not None:
    raw = uploaded.read().decode("utf-8", errors="replace")
    sp = system_prompt_text.strip()
    # Only re-run if data changed
    cache_key = hash(raw + sp + str(window_size) + str(overlap))
    if st.session_state.get("_cache_key") != cache_key:
        st.session_state._cache_key = cache_key
        st.session_state.raw_text = raw
        st.session_state.system_prompt = sp
        st.session_state.report = audit_conversation(
            raw_text=raw,
            system_prompt=sp,
            window_size=window_size,
            overlap=overlap,
            conversation_id=uploaded.name,
        )
        st.session_state.turns = parse_chat_log(raw)

# --- EMPTY STATE ---
if st.session_state.report is None:
    st.markdown("""
    <div class="empty-state fade-in">
        <div class="icon">🔬</div>
        <h3>No conversation loaded</h3>
        <p>Upload a Claude chat export (JSON or plain text) using the sidebar, or click
        <strong>Load Sample Conversation</strong> to see the auditor in action.</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlight cards
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="glass-card fade-in" style="text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.4rem;">🚩</div>
            <div style="font-weight:700;color:#f43f5e;font-size:0.9rem;">Commission</div>
            <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.3rem;">Sycophancy & reality distortion detection</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="glass-card fade-in-d1" style="text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.4rem;">📜</div>
            <div style="font-weight:700;color:#f59e0b;font-size:0.9rem;">Omission</div>
            <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.3rem;">Instruction adherence & prohibition checks</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="glass-card fade-in-d2" style="text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.4rem;">🔄</div>
            <div style="font-weight:700;color:#06b6d4;font-size:0.9rem;">Persistence</div>
            <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.3rem;">Do acknowledged corrections actually hold?</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="glass-card fade-in-d3" style="text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.4rem;">🧠</div>
            <div style="font-weight:700;color:#a78bfa;font-size:0.9rem;">Barometer</div>
            <div style="font-size:0.78rem;color:#94a3b8;margin-top:0.3rem;">Epistemic posture & structural drift signals</div>
        </div>""", unsafe_allow_html=True)

    st.stop()

# ===================================================================
#  DASHBOARD (report loaded)
# ===================================================================

report: AuditReport = st.session_state.report
scores = report.summary_scores
turns = st.session_state.turns or []

# ---------------------------------------------------------------------------
# Row 1: Summary metric cards
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header fade-in"><span class="icon">📊</span> Audit Summary</div>',
            unsafe_allow_html=True)

cols = st.columns(5)
with cols[0]:
    overall = scores.get("overall_drift_score", 1)
    color_class = "mc-emerald" if overall <= 3 else ("mc-amber" if overall <= 6 else "mc-rose")
    st.markdown(metric_card_html("Overall Drift", f"{overall}/10",
                                  f"{report.total_turns} turns analyzed", color_class),
                unsafe_allow_html=True)
with cols[1]:
    v = scores.get("commission_score", 1)
    st.markdown(metric_card_html("Commission", f"{v}/10",
                                  f"{scores.get('commission_flag_count',0)} flags", "mc-rose"),
                unsafe_allow_html=True)
with cols[2]:
    v = scores.get("omission_score", 1)
    st.markdown(metric_card_html("Omission", f"{v}/10",
                                  f"{scores.get('omission_flag_count',0)} flags", "mc-amber"),
                unsafe_allow_html=True)
with cols[3]:
    v = scores.get("correction_persistence_score", 1)
    failed = scores.get("corrections_failed", 0)
    total_corr = scores.get("correction_events_total", 0)
    st.markdown(metric_card_html("Persistence", f"{v}/10",
                                  f"{failed}/{total_corr} corrections failed", "mc-cyan"),
                unsafe_allow_html=True)
with cols[4]:
    v = scores.get("barometer_score", 1)
    st.markdown(metric_card_html("Barometer", f"{v}/10",
                                  f"{scores.get('barometer_red_count',0)} RED signals", "mc-indigo"),
                unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Row 2: Gauges + Radar
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header fade-in-d1"><span class="icon">🎯</span> Drift Profile</div>',
            unsafe_allow_html=True)

g1, g2, g3, g4, g_radar = st.columns([1, 1, 1, 1, 1.6])
with g1:
    st.plotly_chart(make_gauge(scores.get("commission_score", 1), "Commission", "#f43f5e"),
                    use_container_width=True, config={"displayModeBar": False})
with g2:
    st.plotly_chart(make_gauge(scores.get("omission_score", 1), "Omission", "#f59e0b"),
                    use_container_width=True, config={"displayModeBar": False})
with g3:
    st.plotly_chart(make_gauge(scores.get("correction_persistence_score", 1), "Persistence", "#06b6d4"),
                    use_container_width=True, config={"displayModeBar": False})
with g4:
    st.plotly_chart(make_gauge(scores.get("barometer_score", 1), "Barometer", "#a78bfa"),
                    use_container_width=True, config={"displayModeBar": False})
with g_radar:
    st.plotly_chart(make_radar(scores), use_container_width=True,
                    config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Row 3: Timeline + Heatmap
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header fade-in-d2"><span class="icon">📈</span> Drift Timeline</div>',
            unsafe_allow_html=True)
st.plotly_chart(make_timeline(report), use_container_width=True,
                config={"displayModeBar": False})

st.markdown('<div class="section-header fade-in-d3"><span class="icon">🗺️</span> Layer &times; Turn Heatmap</div>',
            unsafe_allow_html=True)
st.plotly_chart(make_heatmap(report), use_container_width=True,
                config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Row 4: Detailed layer tabs
# ---------------------------------------------------------------------------
st.markdown('<div class="section-header fade-in-d4"><span class="icon">🔍</span> Detailed Analysis</div>',
            unsafe_allow_html=True)

tab_baro, tab_persist, tab_comm, tab_omit, tab_conv = st.tabs([
    "🧠 Barometer", "🔄 Persistence", "🚩 Commission", "📜 Omission", "💬 Conversation"
])

# --- TAB: Barometer ---
with tab_baro:
    bc1, bc2 = st.columns([2, 1])
    with bc1:
        st.plotly_chart(make_barometer_chart(report), use_container_width=True,
                        config={"displayModeBar": False})
    with bc2:
        st.plotly_chart(make_barometer_donut(report), use_container_width=True,
                        config={"displayModeBar": False})

    red_signals = [s for s in report.barometer_signals if s.classification == "RED"]
    yellow_signals = [s for s in report.barometer_signals if s.classification == "YELLOW"]

    if red_signals:
        st.markdown("**RED Signals — Structural Drift Detected:**")
        for sig in sorted(red_signals, key=lambda s: s.turn):
            with st.expander(f"Turn {sig.turn} — RED  (severity {sig.severity})"):
                st.markdown(f"**Description:** {sig.description}")
                if sig.evidence:
                    st.code(sig.evidence, language=None)
    if yellow_signals:
        st.markdown("**YELLOW Signals — Weak Epistemic Posture:**")
        for sig in sorted(yellow_signals, key=lambda s: s.turn):
            with st.expander(f"Turn {sig.turn} — YELLOW  (severity {sig.severity})"):
                st.markdown(f"**Description:** {sig.description}")
                if sig.evidence:
                    st.code(sig.evidence, language=None)
    if not red_signals and not yellow_signals:
        st.success("All assistant turns show healthy epistemic posture (GREEN).")

# --- TAB: Persistence ---
with tab_persist:
    st.plotly_chart(make_persistence_chart(report), use_container_width=True,
                    config={"displayModeBar": False})

    if report.correction_events:
        for ev in report.correction_events:
            status_badge = ('<span class="badge badge-red">FAILED</span>'
                            if not ev.held else '<span class="badge badge-green">HELD</span>')
            with st.expander(
                f"Correction at Turn {ev.correction_turn} → Ack Turn {ev.acknowledgment_turn}"
                f"  {'❌ FAILED at Turn ' + str(ev.failure_turn) if not ev.held else '✅ HELD'}"
            ):
                st.markdown(f"**Status:** {status_badge}", unsafe_allow_html=True)
                st.markdown(f"**User said:** {ev.instruction[:300]}")
                if not ev.held:
                    st.warning(f"Correction regressed at turn {ev.failure_turn}. "
                               "The model acknowledged the error but repeated the same behavior.")
    else:
        st.info("No user corrections were detected in this conversation.")

# --- TAB: Commission ---
with tab_comm:
    if report.commission_flags:
        for flag in sorted(report.commission_flags, key=lambda f: f.turn):
            sev_badge_cls = "badge-red" if flag.severity >= 6 else ("badge-yellow" if flag.severity >= 4 else "badge-green")
            with st.expander(f"Turn {flag.turn} — severity {flag.severity}  |  {flag.description[:70]}"):
                st.markdown(
                    f'Severity: <span class="badge {sev_badge_cls}">{flag.severity}/10</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Pattern:** {flag.description}")
                if flag.evidence:
                    st.code(str(flag.evidence), language=None)
                if flag.instruction_ref:
                    st.markdown(f"**Instruction violated:** {flag.instruction_ref}")
    else:
        st.success("No commission drift (sycophancy/reality distortion) detected.")

# --- TAB: Omission ---
with tab_omit:
    if report.omission_flags:
        for flag in sorted(report.omission_flags, key=lambda f: f.turn):
            sev_badge_cls = "badge-red" if flag.severity >= 6 else ("badge-yellow" if flag.severity >= 4 else "badge-green")
            with st.expander(f"Turn {flag.turn} — severity {flag.severity}  |  {flag.description[:70]}"):
                st.markdown(
                    f'Severity: <span class="badge {sev_badge_cls}">{flag.severity}/10</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Flagged:** {flag.description}")
                if flag.instruction_ref:
                    st.markdown(f"**Instruction:** {flag.instruction_ref}")
                if flag.evidence:
                    st.code(str(flag.evidence), language=None)
    else:
        st.success("No omission drift detected via local heuristics.")
        st.caption("Note: Full semantic omission detection requires the API-powered mode.")

# --- TAB: Conversation Replay ---
with tab_conv:
    if turns:
        st.markdown("""
        <div style="font-size:0.78rem;color:#64748b;margin-bottom:0.6rem;">
            Drift flags are shown as inline badges. Pulsing red dots indicate critical drift turns.
        </div>
        """, unsafe_allow_html=True)
        render_conversation(report, turns)
    else:
        st.info("No conversation turns to display.")

# ---------------------------------------------------------------------------
# Row 5: Export
# ---------------------------------------------------------------------------
st.markdown("---")
exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 4])
with exp_col1:
    st.download_button(
        "📥 Download JSON Report",
        data=report_to_json(report),
        file_name="drift_audit_report.json",
        mime="application/json",
        use_container_width=True,
    )
with exp_col2:
    st.download_button(
        "📄 Download Text Report",
        data=format_report(report),
        file_name="drift_audit_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
with exp_col3:
    st.markdown(
        '<div style="font-size:0.72rem;color:#475569;padding-top:0.6rem;">'
        'Built for the Claude Opus 4.6 Hackathon &mdash; '
        'Author: George Abrahamyan &mdash; Enhanced with Grok (xAI)</div>',
        unsafe_allow_html=True,
    )
