"""
Drift Auditor — Interactive Dashboard
======================================
Streamlit UI for the multi-turn drift diagnostic tool.
Built for Claude Opus 4.6 Hackathon.

Author: George Abrahamyan
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import os
import sys
from pathlib import Path
from dataclasses import asdict

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from drift_auditor import (
    audit_conversation,
    parse_chat_log,
    extract_instructions,
    format_report,
    report_to_json,
    AuditReport,
)

# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drift Auditor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ---------- global overrides ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-card: rgba(17, 25, 40, 0.75);
    --bg-card-hover: rgba(17, 25, 40, 0.90);
    --border-glass: rgba(255,255,255,0.08);
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-green: #10b981;
    --accent-yellow: #f59e0b;
    --accent-red: #ef4444;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
}

/* hide default streamlit decorations */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1rem; padding-bottom: 1rem;}

/* dark background */
.stApp {
    background: linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0f172a 100%);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid var(--border-glass);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
}

/* glass card helper */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: background 0.2s ease;
}
.glass-card:hover {
    background: var(--bg-card-hover);
}

/* metric card */
.metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card .metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 0.25rem;
}
.metric-card .metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.metric-card .metric-sub {
    font-size: 0.7rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}
.metric-card .glow-bar {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
}

/* severity colors */
.sev-low .metric-value  { color: var(--accent-green); }
.sev-low .glow-bar      { background: var(--accent-green); box-shadow: 0 0 12px var(--accent-green); }
.sev-med .metric-value  { color: var(--accent-yellow); }
.sev-med .glow-bar      { background: var(--accent-yellow); box-shadow: 0 0 12px var(--accent-yellow); }
.sev-high .metric-value { color: var(--accent-red); }
.sev-high .glow-bar     { background: var(--accent-red); box-shadow: 0 0 12px var(--accent-red); }
.sev-info .metric-value { color: var(--accent-blue); }
.sev-info .glow-bar     { background: var(--accent-blue); box-shadow: 0 0 12px var(--accent-blue); }

/* hero */
.hero-container {
    text-align: center;
    padding: 2.5rem 0 1rem;
}
.hero-container h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.hero-container .hero-sub {
    color: var(--text-secondary);
    font-size: 1rem;
    max-width: 640px;
    margin: 0 auto;
    line-height: 1.6;
}
.hero-container .hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(139,92,246,0.15));
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 1rem;
    letter-spacing: 0.04em;
}

/* tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-card);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid var(--border-glass);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.25rem;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.15) !important;
    color: var(--accent-blue) !important;
}

/* expander */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-weight: 600;
}

/* code blocks */
.stCodeBlock, code {
    font-family: 'JetBrains Mono', monospace !important;
}

/* chat bubbles */
.chat-turn {
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    line-height: 1.65;
    position: relative;
}
.chat-user {
    background: rgba(59,130,246,0.10);
    border: 1px solid rgba(59,130,246,0.20);
    margin-left: 2rem;
}
.chat-assistant {
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.15);
    margin-right: 2rem;
}
.chat-role {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.35rem;
}
.chat-user .chat-role { color: var(--accent-blue); }
.chat-assistant .chat-role { color: var(--accent-purple); }
.drift-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
    margin-left: 6px;
    vertical-align: middle;
}
.drift-badge-red {
    background: rgba(239,68,68,0.2);
    color: #fca5a5;
    border: 1px solid rgba(239,68,68,0.35);
}
.drift-badge-yellow {
    background: rgba(245,158,11,0.2);
    color: #fcd34d;
    border: 1px solid rgba(245,158,11,0.35);
}
.drift-badge-green {
    background: rgba(16,185,129,0.2);
    color: #6ee7b7;
    border: 1px solid rgba(16,185,129,0.35);
}
.drift-badge-blue {
    background: rgba(59,130,246,0.2);
    color: #93c5fd;
    border: 1px solid rgba(59,130,246,0.35);
}

/* section header */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-glass);
    color: var(--text-primary);
}

/* plotly backgrounds */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: severity CSS class
# ---------------------------------------------------------------------------

def _sev_class(score: int) -> str:
    if score <= 3:
        return "sev-low"
    if score <= 6:
        return "sev-med"
    return "sev-high"


def _color_for_score(score: int) -> str:
    if score <= 3:
        return "#10b981"
    if score <= 6:
        return "#f59e0b"
    return "#ef4444"


def _barom_color(classification: str) -> str:
    return {"GREEN": "#10b981", "YELLOW": "#f59e0b", "RED": "#ef4444"}.get(
        classification, "#64748b"
    )


# ---------------------------------------------------------------------------
# Metric card HTML
# ---------------------------------------------------------------------------

def metric_card(label: str, value, sub: str = "", severity_class: str = "sev-info"):
    return f"""
    <div class="metric-card {severity_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
        <div class="glow-bar"></div>
    </div>
    """


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(
        bgcolor="rgba(17,25,40,0.6)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(size=11),
    ),
)

def _plotly_layout(**overrides) -> dict:
    """Return a copy of the base Plotly layout with overrides merged."""
    base = {**_PLOTLY_BASE}
    base.update(overrides)
    return base


def build_drift_timeline(report: AuditReport) -> go.Figure:
    """Multi-layer drift timeline — the hero chart."""
    fig = go.Figure()

    # Barometer background strip
    for sig in report.barometer_signals:
        fig.add_vrect(
            x0=sig.turn - 0.45,
            x1=sig.turn + 0.45,
            fillcolor=_barom_color(sig.classification),
            opacity=0.07,
            layer="below",
            line_width=0,
        )

    # Commission flags
    if report.commission_flags:
        cf = report.commission_flags
        fig.add_trace(
            go.Scatter(
                x=[f.turn for f in cf],
                y=[f.severity for f in cf],
                mode="markers+lines",
                name="Commission",
                marker=dict(size=10, color="#ef4444", symbol="diamond"),
                line=dict(color="#ef4444", width=1, dash="dot"),
                hovertemplate="<b>Turn %{x}</b><br>Severity: %{y}<br>%{text}<extra>Commission</extra>",
                text=[f.description[:60] for f in cf],
            )
        )

    # Omission flags
    if report.omission_flags:
        of = report.omission_flags
        fig.add_trace(
            go.Scatter(
                x=[f.turn for f in of],
                y=[f.severity for f in of],
                mode="markers+lines",
                name="Omission",
                marker=dict(size=10, color="#f59e0b", symbol="triangle-up"),
                line=dict(color="#f59e0b", width=1, dash="dot"),
                hovertemplate="<b>Turn %{x}</b><br>Severity: %{y}<br>%{text}<extra>Omission</extra>",
                text=[f.description[:60] for f in of],
            )
        )

    # Barometer severity line
    if report.barometer_signals:
        bs = sorted(report.barometer_signals, key=lambda s: s.turn)
        fig.add_trace(
            go.Scatter(
                x=[s.turn for s in bs],
                y=[s.severity for s in bs],
                mode="lines+markers",
                name="Barometer",
                marker=dict(
                    size=8,
                    color=[_barom_color(s.classification) for s in bs],
                    line=dict(width=1, color="rgba(255,255,255,0.3)"),
                ),
                line=dict(color="#8b5cf6", width=2),
                hovertemplate="<b>Turn %{x}</b><br>Severity: %{y}<br>%{text}<extra>Barometer</extra>",
                text=[f"{s.classification}: {s.description[:50]}" for s in bs],
            )
        )

    # Correction events — vertical lines
    for ev in report.correction_events:
        color = "#ef4444" if not ev.held else "#10b981"
        label = "FAILED" if not ev.held else "HELD"
        fig.add_vline(
            x=ev.correction_turn,
            line_dash="dash",
            line_color=color,
            opacity=0.5,
            annotation_text=f"Correction ({label})",
            annotation_position="top",
            annotation_font_size=9,
            annotation_font_color=color,
        )
        if not ev.held and ev.failure_turn:
            fig.add_vline(
                x=ev.failure_turn,
                line_dash="dot",
                line_color="#ef4444",
                opacity=0.3,
                annotation_text="Regression",
                annotation_position="bottom",
                annotation_font_size=9,
                annotation_font_color="#ef4444",
            )

    fig.update_layout(
        **_plotly_layout(),
        title=dict(text="Drift Timeline — All Layers", font=dict(size=16, color="#f1f5f9")),
        xaxis=dict(
            title="Turn",
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Severity (1-10)",
            range=[0, 11],
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False,
        ),
        height=420,
        hovermode="x unified",
    )
    return fig


def build_radar_chart(scores: dict) -> go.Figure:
    """Radar chart of the four layer scores."""
    categories = ["Commission", "Omission", "Persistence", "Barometer"]
    values = [
        scores.get("commission_score", 1),
        scores.get("omission_score", 1),
        scores.get("correction_persistence_score", 1),
        scores.get("barometer_score", 1),
    ]
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(139,92,246,0.15)",
            line=dict(color="#8b5cf6", width=2),
            marker=dict(size=7, color="#8b5cf6"),
            name="Score",
        )
    )
    fig.update_layout(
        **_plotly_layout(margin=dict(l=60, r=60, t=30, b=30)),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=10, color="#64748b"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=11, color="#cbd5e1"),
            ),
        ),
        showlegend=False,
        height=340,
    )
    return fig


def build_barometer_heatmap(report: AuditReport) -> go.Figure:
    """Horizontal heatmap strip of barometer classifications per turn."""
    if not report.barometer_signals:
        return go.Figure()

    signals = sorted(report.barometer_signals, key=lambda s: s.turn)
    turns = [s.turn for s in signals]
    colors = [_barom_color(s.classification) for s in signals]
    classifications = [s.classification for s in signals]
    descriptions = [s.description[:60] for s in signals]

    # Map to numeric for heatmap
    color_map = {"GREEN": 1, "YELLOW": 2, "RED": 3}
    z = [[color_map.get(s.classification, 0) for s in signals]]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=turns,
            y=["Posture"],
            colorscale=[
                [0.0, "#1e293b"],
                [0.33, "#10b981"],
                [0.66, "#f59e0b"],
                [1.0, "#ef4444"],
            ],
            showscale=False,
            hovertemplate="Turn %{x}<br>%{text}<extra></extra>",
            text=[descriptions],
            xgap=2,
        )
    )
    fig.update_layout(
        **_plotly_layout(margin=dict(l=60, r=20, t=10, b=30)),
        height=90,
        xaxis=dict(title="Turn", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(showticklabels=True, gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_correction_flow(report: AuditReport) -> go.Figure:
    """Sankey-style flow of correction outcomes."""
    events = report.correction_events
    if not events:
        return go.Figure()

    held = sum(1 for e in events if e.held)
    failed = sum(1 for e in events if not e.held)

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20,
                thickness=30,
                line=dict(color="rgba(255,255,255,0.1)", width=1),
                label=[
                    f"Corrections ({len(events)})",
                    f"Acknowledged ({len(events)})",
                    f"Held ({held})",
                    f"Regressed ({failed})",
                ],
                color=[
                    "#3b82f6",
                    "#8b5cf6",
                    "#10b981",
                    "#ef4444",
                ],
            ),
            link=dict(
                source=[0, 1, 1],
                target=[1, 2, 3],
                value=[len(events), held, max(failed, 0.1)],
                color=[
                    "rgba(139,92,246,0.25)",
                    "rgba(16,185,129,0.25)",
                    "rgba(239,68,68,0.25)",
                ],
            ),
        )
    )
    fig.update_layout(
        **_plotly_layout(),
        title=dict(text="Correction Persistence Flow", font=dict(size=14, color="#f1f5f9")),
        height=300,
    )
    return fig


def build_score_gauge(score: int, label: str = "Overall Drift") -> go.Figure:
    """Semi-circular gauge for the overall drift score."""
    color = _color_for_score(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(font=dict(size=52, color=color, family="JetBrains Mono, monospace")),
            gauge=dict(
                axis=dict(range=[1, 10], tickwidth=1, tickcolor="rgba(255,255,255,0.1)"),
                bar=dict(color=color, thickness=0.75),
                bgcolor="rgba(17,25,40,0.5)",
                borderwidth=1,
                bordercolor="rgba(255,255,255,0.08)",
                steps=[
                    dict(range=[1, 3], color="rgba(16,185,129,0.12)"),
                    dict(range=[3, 6], color="rgba(245,158,11,0.12)"),
                    dict(range=[6, 10], color="rgba(239,68,68,0.12)"),
                ],
                threshold=dict(
                    line=dict(color="white", width=2),
                    thickness=0.8,
                    value=score,
                ),
            ),
            title=dict(text=label, font=dict(size=13, color="#94a3b8")),
        )
    )
    fig.update_layout(
        **_plotly_layout(margin=dict(l=30, r=30, t=50, b=10)),
        height=240,
    )
    return fig


# ---------------------------------------------------------------------------
# Conversation replay with annotations
# ---------------------------------------------------------------------------

def render_conversation_replay(turns: list[dict], report: AuditReport):
    """Render the conversation as chat bubbles with inline drift badges."""
    # Build lookup: turn -> list of flags/signals
    turn_flags = {}
    for f in report.commission_flags:
        turn_flags.setdefault(f.turn, []).append(("Commission", f.severity, f.description))
    for f in report.omission_flags:
        turn_flags.setdefault(f.turn, []).append(("Omission", f.severity, f.description))

    turn_barometer = {}
    for s in report.barometer_signals:
        turn_barometer[s.turn] = s

    correction_turns = set()
    for ev in report.correction_events:
        correction_turns.add(ev.correction_turn)

    for turn in turns:
        t = turn["turn"]
        role = turn["role"]
        content = turn["content"]

        if role == "system":
            continue

        css_class = "chat-user" if role == "user" else "chat-assistant"
        role_label = "USER" if role == "user" else "ASSISTANT"

        # Build badges
        badges_html = ""
        if t in correction_turns and role == "user":
            badges_html += '<span class="drift-badge drift-badge-blue">CORRECTION</span>'
        for layer, sev, desc in turn_flags.get(t, []):
            badge_cls = "drift-badge-red" if sev >= 6 else "drift-badge-yellow"
            badges_html += f'<span class="drift-badge {badge_cls}" title="{desc}">{layer} sev:{sev}</span>'
        baro = turn_barometer.get(t)
        if baro and role == "assistant":
            badge_cls = {
                "GREEN": "drift-badge-green",
                "YELLOW": "drift-badge-yellow",
                "RED": "drift-badge-red",
            }.get(baro.classification, "")
            badges_html += f'<span class="drift-badge {badge_cls}">{baro.classification}</span>'

        # Escape HTML in content
        safe_content = (
            content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

        st.markdown(
            f"""
            <div class="chat-turn {css_class}">
                <div class="chat-role">{role_label} &mdash; Turn {t} {badges_html}</div>
                <div>{safe_content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Load sample data
# ---------------------------------------------------------------------------

def load_sample() -> tuple[str, str]:
    """Return (sample_conversation, sample_system_prompt) from examples/."""
    base = Path(__file__).parent.parent / "examples"
    conv_path = base / "sample_conversation.txt"
    prompt_path = base / "sample_system_prompt.txt"
    conv = conv_path.read_text(encoding="utf-8") if conv_path.exists() else ""
    prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
    return conv, prompt


# ===================================================================
# MAIN APP
# ===================================================================

def main():
    # ---- Hero ----
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-badge">CLAUDE OPUS 4.6 HACKATHON</div>
            <h1>Drift Auditor</h1>
            <p class="hero-sub">
                Multi-turn drift diagnostic tool.
                Detects sycophancy, omission drift, correction persistence failures,
                and structural epistemic decay that single-turn evaluations miss.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("### Configuration")

        analysis_mode = st.radio(
            "Mode",
            ["Upload conversation", "Try demo"],
            index=1,
            help="Upload your own Claude chat export or try the built-in sample.",
        )

        st.markdown("---")
        st.markdown("#### Analysis Parameters")
        window_size = st.slider("Sliding window (turns)", 10, 100, 50, step=5)
        overlap = st.slider("Window overlap", 0, 20, 10, step=2)

        st.markdown("---")
        st.markdown("#### Optional Context")
        system_prompt_input = st.text_area(
            "System prompt",
            height=100,
            placeholder="Paste the system prompt used in this conversation...",
        )
        user_prefs_input = st.text_area(
            "User preferences",
            height=80,
            placeholder="Any known user preferences...",
        )

        st.markdown("---")
        st.markdown(
            "<small style='color:#64748b'>Built by George Abrahamyan<br>"
            "Powered by Claude Opus 4.6</small>",
            unsafe_allow_html=True,
        )

    # ---- Input handling ----
    raw_text = None
    system_prompt = system_prompt_input.strip()
    user_prefs = user_prefs_input.strip()

    if analysis_mode == "Try demo":
        sample_conv, sample_prompt = load_sample()
        raw_text = sample_conv
        if not system_prompt:
            system_prompt = sample_prompt
        st.info(
            "Running on the built-in sample conversation (bakery marketing advisor). "
            "Switch to **Upload conversation** in the sidebar to analyze your own chats.",
            icon="💡",
        )
    else:
        uploaded = st.file_uploader(
            "Drop a Claude chat export here",
            type=["txt", "json"],
            help="Supports Claude.ai JSON exports, plain-text transcripts with Human:/Assistant: markers, "
            "and custom JSON with role/content fields.",
        )
        if uploaded:
            raw_text = uploaded.read().decode("utf-8", errors="replace")

    if raw_text is None:
        # Empty state
        st.markdown(
            """
            <div class="glass-card" style="text-align:center; padding:3rem;">
                <p style="font-size:2.5rem; margin-bottom:0.5rem;">📂</p>
                <p style="color:var(--text-secondary); font-size:1rem;">
                    Upload a conversation or try the demo to begin analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ---- Run audit ----
    with st.spinner("Analyzing conversation for drift patterns..."):
        report = audit_conversation(
            raw_text=raw_text,
            system_prompt=system_prompt,
            user_preferences=user_prefs,
            window_size=window_size,
            overlap=overlap,
            conversation_id="dashboard_analysis",
        )
        turns = parse_chat_log(raw_text)

    scores = report.summary_scores

    # ================================================================
    # RESULTS
    # ================================================================

    # ---- Overall gauge + layer scores ----
    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(
            build_score_gauge(scores["overall_drift_score"]),
            use_container_width=True,
            key="gauge",
        )
    with g2:
        cols = st.columns(4)
        layer_data = [
            ("Commission", scores["commission_score"], f"{scores['commission_flag_count']} flags"),
            ("Omission", scores["omission_score"], f"{scores['omission_flag_count']} flags"),
            ("Persistence", scores["correction_persistence_score"],
             f"{scores['corrections_failed']}/{scores['correction_events_total']} failed"),
            ("Barometer", scores["barometer_score"],
             f"{scores['barometer_red_count']}R / {scores['barometer_yellow_count']}Y / {scores['barometer_green_count']}G"),
        ]
        for col, (label, score, sub) in zip(cols, layer_data):
            col.markdown(metric_card(label, score, sub, _sev_class(score)), unsafe_allow_html=True)

        # Summary stats row
        st.markdown("")
        scols = st.columns(3)
        scols[0].markdown(
            metric_card("Total Turns", report.total_turns, "", "sev-info"),
            unsafe_allow_html=True,
        )
        scols[1].markdown(
            metric_card("Instructions", report.instructions_extracted, "extracted", "sev-info"),
            unsafe_allow_html=True,
        )
        scols[2].markdown(
            metric_card(
                "Corrections",
                len(report.correction_events),
                f"{scores['corrections_failed']} regressed",
                "sev-info",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ---- Barometer heatmap strip ----
    if report.barometer_signals:
        st.markdown('<div class="section-header">Epistemic Posture Heatmap</div>', unsafe_allow_html=True)
        st.plotly_chart(build_barometer_heatmap(report), use_container_width=True, key="heatmap")

    # ---- Main timeline ----
    st.markdown('<div class="section-header">Drift Timeline</div>', unsafe_allow_html=True)
    st.plotly_chart(build_drift_timeline(report), use_container_width=True, key="timeline")

    # ---- Tabs ----
    tab_labels = [
        "🧠 Barometer",
        "🚩 Commission",
        "📜 Omission",
        "🔄 Persistence",
        "💬 Conversation",
        "📊 Radar",
        "📥 Export",
    ]
    tabs = st.tabs(tab_labels)

    # -- Barometer tab --
    with tabs[0]:
        if report.barometer_signals:
            red_sigs = [s for s in report.barometer_signals if s.classification == "RED"]
            yellow_sigs = [s for s in report.barometer_signals if s.classification == "YELLOW"]
            green_sigs = [s for s in report.barometer_signals if s.classification == "GREEN"]

            bc1, bc2, bc3 = st.columns(3)
            bc1.markdown(metric_card("RED", len(red_sigs), "structural drift", "sev-high"), unsafe_allow_html=True)
            bc2.markdown(metric_card("YELLOW", len(yellow_sigs), "generic hedging", "sev-med"), unsafe_allow_html=True)
            bc3.markdown(metric_card("GREEN", len(green_sigs), "healthy posture", "sev-low"), unsafe_allow_html=True)

            st.markdown("")
            for sig in sorted(report.barometer_signals, key=lambda s: s.turn):
                color = _barom_color(sig.classification)
                icon = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}.get(sig.classification, "⚪")
                with st.expander(f"{icon} Turn {sig.turn} — {sig.classification} (severity {sig.severity})"):
                    st.markdown(f"**Description:** {sig.description}")
                    if sig.evidence:
                        st.code(sig.evidence, language=None)
        else:
            st.info("No barometer signals detected.", icon="✅")

    # -- Commission tab --
    with tabs[1]:
        if report.commission_flags:
            st.markdown(f"**{len(report.commission_flags)} commission drift flags detected.**")
            for f in sorted(report.commission_flags, key=lambda x: x.turn):
                sev_icon = "🔴" if f.severity >= 6 else ("🟡" if f.severity >= 4 else "🟢")
                with st.expander(f"{sev_icon} Turn {f.turn} — severity {f.severity}: {f.description[:70]}"):
                    st.markdown(f"**Layer:** {f.layer}")
                    st.markdown(f"**Severity:** {f.severity}/10")
                    st.markdown(f"**Description:** {f.description}")
                    if f.evidence:
                        st.code(str(f.evidence), language=None)
        else:
            st.success("No commission drift detected.", icon="✅")

    # -- Omission tab --
    with tabs[2]:
        if report.omission_flags:
            st.markdown(f"**{len(report.omission_flags)} omission drift flags detected.**")
            for f in sorted(report.omission_flags, key=lambda x: x.turn):
                sev_icon = "🔴" if f.severity >= 6 else ("🟡" if f.severity >= 4 else "🟢")
                with st.expander(f"{sev_icon} Turn {f.turn} — severity {f.severity}"):
                    st.markdown(f"**Description:** {f.description}")
                    if f.instruction_ref:
                        st.markdown(f"**Violated instruction:** `{f.instruction_ref[:120]}`")
                    if f.evidence:
                        st.code(str(f.evidence), language=None)
        else:
            st.info(
                "No omission drift detected by local heuristics. "
                "For semantic omission detection, enable API mode with an Anthropic API key.",
                icon="ℹ️",
            )

    # -- Persistence tab --
    with tabs[3]:
        if report.correction_events:
            st.plotly_chart(build_correction_flow(report), use_container_width=True, key="sankey")
            st.markdown("")
            for ev in report.correction_events:
                status = "HELD ✅" if ev.held else f"REGRESSED ❌ at turn {ev.failure_turn}"
                color = "green" if ev.held else "red"
                with st.expander(
                    f"{'✅' if ev.held else '❌'} Turn {ev.correction_turn} → Ack {ev.acknowledgment_turn}: {status}"
                ):
                    st.markdown(f"**Correction at turn:** {ev.correction_turn}")
                    st.markdown(f"**Acknowledged at turn:** {ev.acknowledgment_turn}")
                    st.markdown(f"**Status:** :{color}[{status}]")
                    st.markdown(f"**Context:** {ev.instruction[:200]}")
                    if not ev.held:
                        st.warning(
                            f"The model acknowledged the correction but regressed to the same behavior "
                            f"at turn {ev.failure_turn}. This is the core correction persistence failure pattern."
                        )
        else:
            st.info("No correction events detected in this conversation.", icon="ℹ️")

    # -- Conversation replay tab --
    with tabs[4]:
        st.markdown(
            '<div class="section-header">Annotated Conversation Replay</div>',
            unsafe_allow_html=True,
        )
        st.caption("Badges show detected drift signals inline with each turn.")
        render_conversation_replay(turns, report)

    # -- Radar tab --
    with tabs[5]:
        rc1, rc2 = st.columns([1, 1])
        with rc1:
            st.plotly_chart(build_radar_chart(scores), use_container_width=True, key="radar")
        with rc2:
            st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
            score_desc = {
                "commission_score": ("Commission Drift", "Sycophancy, reality distortion, unwarranted confidence"),
                "omission_score": ("Omission Drift", "Instruction adherence failures via keyword heuristics"),
                "correction_persistence_score": ("Correction Persistence", "Ratio of corrections that regressed after acknowledgment"),
                "barometer_score": ("Structural Barometer", "Epistemic posture health — RED signal density"),
            }
            for key, (name, desc) in score_desc.items():
                val = scores.get(key, 1)
                color = _color_for_score(val)
                st.markdown(
                    f"**{name}:** <span style='color:{color}; font-weight:700; "
                    f"font-family:JetBrains Mono,monospace'>{val}/10</span>  \n"
                    f"<small style='color:#64748b'>{desc}</small>",
                    unsafe_allow_html=True,
                )
                st.markdown("")

            st.markdown(
                f"**Overall Drift Score:** "
                f"<span style='color:{_color_for_score(scores['overall_drift_score'])}; "
                f"font-weight:800; font-size:1.5rem; font-family:JetBrains Mono,monospace'>"
                f"{scores['overall_drift_score']}/10</span>",
                unsafe_allow_html=True,
            )
            st.caption("Weighted: Commission 20% · Omission 35% · Persistence 25% · Barometer 20%")

    # -- Export tab --
    with tabs[6]:
        st.markdown('<div class="section-header">Export Report</div>', unsafe_allow_html=True)

        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button(
                label="Download JSON Report",
                data=report_to_json(report),
                file_name="drift_audit_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with ec2:
            st.download_button(
                label="Download Text Report",
                data=format_report(report),
                file_name="drift_audit_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown("")
        with st.expander("Preview text report"):
            st.code(format_report(report), language=None)


if __name__ == "__main__":
    main()
