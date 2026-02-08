"""
Drift Auditor — Interactive Dashboard
======================================
Streamlit-based visual interface for the multi-turn drift diagnostic tool.

Run:  streamlit run app.py
"""

import sys, os, json, math
from pathlib import Path
from dataclasses import asdict

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── make src/ importable ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from drift_auditor import (
    audit_conversation,
    AuditReport,
    format_report,
    report_to_json,
)

# ═══════════════════════════════════════════════════════════════════════════
# Page config & custom CSS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Drift Auditor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS — dark glassmorphism theme
st.markdown(
    """
<style>
/* ── Base overrides ────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-card: rgba(15, 23, 42, 0.65);
    --border-glass: rgba(100, 120, 180, 0.18);
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-teal: #14b8a6;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --accent-green: #22c55e;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --glow-blue: 0 0 20px rgba(59,130,246,0.25);
    --glow-red: 0 0 20px rgba(239,68,68,0.25);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid var(--border-glass);
}

.stApp {
    background: linear-gradient(135deg, #0a0e17 0%, #0f172a 40%, #1a1035 100%);
    font-family: 'Inter', sans-serif;
}

/* ── Glass card containers ─────────────────────────────────────────────── */
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
    box-shadow: var(--glow-blue);
}

/* ── Score gauge ───────────────────────────────────────────────────────── */
.score-gauge {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.2rem 0.5rem;
}
.score-number {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -1px;
}
.score-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-secondary);
    margin-top: 0.4rem;
}
.score-sublabel {
    font-size: 0.68rem;
    color: var(--text-secondary);
    margin-top: 0.15rem;
}
.sev-low   { color: var(--accent-green); text-shadow: 0 0 18px rgba(34,197,94,0.5); }
.sev-med   { color: var(--accent-amber); text-shadow: 0 0 18px rgba(245,158,11,0.5); }
.sev-high  { color: var(--accent-red);   text-shadow: 0 0 18px rgba(239,68,68,0.5); }

/* ── Hero banner ───────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, rgba(59,130,246,0.12) 0%, rgba(139,92,246,0.12) 50%, rgba(20,184,166,0.08) 100%);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(59,130,246,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e2e8f0, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
}
.hero p {
    color: var(--text-secondary);
    font-size: 0.92rem;
    margin: 0;
    max-width: 750px;
}

/* ── Badge chips ───────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin: 0 0.3rem 0.3rem 0;
}
.badge-red    { background: rgba(239,68,68,0.18); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.badge-yellow { background: rgba(245,158,11,0.18); color: #fde68a; border: 1px solid rgba(245,158,11,0.3); }
.badge-green  { background: rgba(34,197,94,0.18); color: #86efac; border: 1px solid rgba(34,197,94,0.3); }
.badge-blue   { background: rgba(59,130,246,0.18); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3); }
.badge-purple { background: rgba(139,92,246,0.18); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.3); }

/* ── Turn evidence box ─────────────────────────────────────────────────── */
.evidence-box {
    background: rgba(15, 23, 42, 0.5);
    border-left: 3px solid var(--accent-purple);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin: 0.5rem 0;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Layer section headers ─────────────────────────────────────────────── */
.layer-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
}
.layer-header .layer-icon {
    font-size: 1.3rem;
}
.layer-header h3 {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}
.layer-header .layer-count {
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ── Correction timeline ───────────────────────────────────────────────── */
.correction-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(100,120,180,0.1);
}
.correction-row:last-child { border-bottom: none; }
.correction-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.correction-dot.held  { background: var(--accent-green); box-shadow: 0 0 8px rgba(34,197,94,0.6); }
.correction-dot.failed { background: var(--accent-red); box-shadow: 0 0 8px rgba(239,68,68,0.6); }

/* ── Streamlit widget tweaks ───────────────────────────────────────────── */
.stExpander {
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    background: rgba(15, 23, 42, 0.4) !important;
}

div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(12px);
}

div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

div[data-testid="stTabs"] button {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stDownloadButton > button:hover {
    opacity: 0.85 !important;
}

/* ── Stat separator ────────────────────────────────────────────────────── */
.stat-divider {
    width: 1px;
    height: 50px;
    background: var(--border-glass);
    margin: 0 0.5rem;
}

/* ── Empty state ───────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--text-secondary);
}
.empty-state .empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}
.empty-state h3 {
    color: var(--text-primary);
    font-weight: 600;
    margin-bottom: 0.4rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sev_class(score: int) -> str:
    """CSS class for a severity score (1-10)."""
    if score <= 3:
        return "sev-low"
    elif score <= 6:
        return "sev-med"
    return "sev-high"


def _sev_word(score: int) -> str:
    if score <= 2:
        return "Clean"
    elif score <= 4:
        return "Low"
    elif score <= 6:
        return "Moderate"
    elif score <= 8:
        return "High"
    return "Severe"


def _sev_color(score: int) -> str:
    if score <= 3:
        return "#22c55e"
    elif score <= 6:
        return "#f59e0b"
    return "#ef4444"


def _baro_color(cls: str) -> str:
    return {"RED": "#ef4444", "YELLOW": "#f59e0b", "GREEN": "#22c55e"}.get(cls, "#64748b")


# ═══════════════════════════════════════════════════════════════════════════
# Charts (Plotly)
# ═══════════════════════════════════════════════════════════════════════════

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(
        bgcolor="rgba(15,23,42,0.6)",
        bordercolor="rgba(100,120,180,0.2)",
        borderwidth=1,
        font=dict(size=11),
    ),
)


def build_drift_timeline(report: AuditReport) -> go.Figure:
    """
    Multi-layer drift timeline — scatter plot showing severity per turn
    for each detection layer, plus a barometer colour band at the bottom.
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.78, 0.22],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Drift Severity by Turn", "Barometer Status"),
    )

    # ── Commission flags ────────────────────────────────────────────────
    if report.commission_flags:
        turns = [f.turn for f in report.commission_flags]
        sevs = [f.severity for f in report.commission_flags]
        texts = [f.description for f in report.commission_flags]
        fig.add_trace(
            go.Scatter(
                x=turns, y=sevs, mode="markers+lines",
                name="Commission",
                marker=dict(size=10, color="#f43f5e", symbol="circle",
                            line=dict(width=1, color="#fff")),
                line=dict(width=1, dash="dot", color="rgba(244,63,94,0.3)"),
                text=texts, hovertemplate="Turn %{x}<br>Severity: %{y}<br>%{text}<extra>Commission</extra>",
            ),
            row=1, col=1,
        )

    # ── Omission flags ──────────────────────────────────────────────────
    if report.omission_flags:
        turns = [f.turn for f in report.omission_flags]
        sevs = [f.severity for f in report.omission_flags]
        texts = [f.description for f in report.omission_flags]
        fig.add_trace(
            go.Scatter(
                x=turns, y=sevs, mode="markers+lines",
                name="Omission",
                marker=dict(size=10, color="#a78bfa", symbol="diamond",
                            line=dict(width=1, color="#fff")),
                line=dict(width=1, dash="dot", color="rgba(167,139,250,0.3)"),
                text=texts, hovertemplate="Turn %{x}<br>Severity: %{y}<br>%{text}<extra>Omission</extra>",
            ),
            row=1, col=1,
        )

    # ── Correction persistence failures ─────────────────────────────────
    failed = [e for e in report.correction_events if not e.held]
    if failed:
        fig.add_trace(
            go.Scatter(
                x=[e.failure_turn for e in failed],
                y=[8] * len(failed),
                mode="markers",
                name="Correction Failed",
                marker=dict(size=14, color="#fb923c", symbol="x-thin-open",
                            line=dict(width=3, color="#fb923c")),
                text=[f"Correction from T{e.correction_turn} regressed" for e in failed],
                hovertemplate="Turn %{x}<br>%{text}<extra>Correction Failure</extra>",
            ),
            row=1, col=1,
        )

    # ── Barometer band (row 2) ──────────────────────────────────────────
    if report.barometer_signals:
        for sig in report.barometer_signals:
            color = _baro_color(sig.classification)
            fig.add_trace(
                go.Bar(
                    x=[sig.turn], y=[1],
                    marker_color=color,
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate=(
                        f"Turn {sig.turn}<br>{sig.classification}<br>"
                        f"{sig.description[:80]}<extra></extra>"
                    ),
                ),
                row=2, col=1,
            )

    # ── Layout ──────────────────────────────────────────────────────────
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=420,
        barmode="stack",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Severity", range=[0, 11], dtick=2, row=1, col=1,
                     gridcolor="rgba(100,120,180,0.08)")
    fig.update_yaxes(showticklabels=False, row=2, col=1,
                     gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_text="Turn", row=2, col=1,
                     gridcolor="rgba(100,120,180,0.08)")
    fig.update_xaxes(gridcolor="rgba(100,120,180,0.08)", row=1, col=1)

    return fig


def build_score_radar(scores: dict) -> go.Figure:
    """Radar chart of the four layer scores."""
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
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=cats_closed,
        fill="toself",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=7, color="#3b82f6"),
        hovertemplate="%{theta}: %{r}/10<extra></extra>",
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=320,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 10], showticklabels=True, tickfont=dict(size=10, color="#64748b"),
                            gridcolor="rgba(100,120,180,0.12)"),
            angularaxis=dict(gridcolor="rgba(100,120,180,0.12)",
                             tickfont=dict(size=12, color="#e2e8f0")),
        ),
        showlegend=False,
    )
    return fig


def build_barometer_donut(report: AuditReport) -> go.Figure:
    """Donut chart — GREEN / YELLOW / RED distribution."""
    red = sum(1 for s in report.barometer_signals if s.classification == "RED")
    yellow = sum(1 for s in report.barometer_signals if s.classification == "YELLOW")
    green = sum(1 for s in report.barometer_signals if s.classification == "GREEN")

    fig = go.Figure(go.Pie(
        labels=["RED", "YELLOW", "GREEN"],
        values=[red, yellow, green],
        marker=dict(colors=["#ef4444", "#f59e0b", "#22c55e"],
                    line=dict(color="#0f172a", width=2)),
        hole=0.6,
        textinfo="label+value",
        textfont=dict(size=12, family="Inter"),
        hovertemplate="%{label}: %{value} turns<extra></extra>",
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=280,
        showlegend=False,
        annotations=[dict(text="Posture", font=dict(size=14, color="#e2e8f0"), showarrow=False)],
    )
    return fig


def build_correction_waterfall(report: AuditReport) -> go.Figure:
    """Waterfall showing correction events — held vs failed."""
    if not report.correction_events:
        fig = go.Figure()
        fig.update_layout(**_PLOTLY_LAYOUT, height=250,
                          annotations=[dict(text="No correction events", font=dict(size=14, color="#64748b"),
                                            showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])
        return fig

    labels, colors, values = [], [], []
    for i, ev in enumerate(report.correction_events):
        label = f"T{ev.correction_turn}→T{ev.acknowledgment_turn}"
        held = ev.held
        labels.append(label)
        colors.append("#22c55e" if held else "#ef4444")
        values.append(1 if held else -1)

    fig = go.Figure(go.Bar(
        x=labels, y=[1] * len(labels),
        marker_color=colors,
        text=["Held" if v > 0 else "Failed" for v in values],
        textposition="inside",
        hovertemplate="%{x}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=250,
        yaxis=dict(showticklabels=False, gridcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Correction (Turn→Ack)", gridcolor="rgba(100,120,180,0.08)"),
    )
    return fig


def build_severity_heatmap(report: AuditReport) -> go.Figure:
    """
    Turn-by-turn heatmap: each column is a turn, each row is a layer.
    Cell intensity = max severity at that (layer, turn).
    """
    max_turn = report.total_turns
    if max_turn == 0:
        fig = go.Figure()
        fig.update_layout(**_PLOTLY_LAYOUT, height=200)
        return fig

    layers = ["Commission", "Omission", "Persistence", "Barometer"]
    grid = [[0] * max_turn for _ in range(4)]

    for f in report.commission_flags:
        if 0 <= f.turn < max_turn:
            grid[0][f.turn] = max(grid[0][f.turn], f.severity)
    for f in report.omission_flags:
        if 0 <= f.turn < max_turn:
            grid[1][f.turn] = max(grid[1][f.turn], f.severity)
    for e in report.correction_events:
        if not e.held and e.failure_turn is not None and 0 <= e.failure_turn < max_turn:
            grid[2][e.failure_turn] = max(grid[2][e.failure_turn], 8)
    for s in report.barometer_signals:
        if 0 <= s.turn < max_turn:
            grid[3][s.turn] = max(grid[3][s.turn], s.severity)

    fig = go.Figure(go.Heatmap(
        z=grid,
        x=list(range(max_turn)),
        y=layers,
        colorscale=[
            [0.0, "rgba(15,23,42,0.8)"],
            [0.3, "#1e40af"],
            [0.5, "#f59e0b"],
            [0.7, "#ea580c"],
            [1.0, "#dc2626"],
        ],
        zmin=0, zmax=10,
        hovertemplate="Turn %{x}<br>%{y}<br>Severity: %{z}<extra></extra>",
        colorbar=dict(title="Sev", titlefont=dict(size=11), tickfont=dict(size=10)),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        height=240,
        xaxis=dict(title="Turn", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Renderers — card builders
# ═══════════════════════════════════════════════════════════════════════════

def render_score_card(label: str, score: int, sublabel: str = ""):
    cls = _sev_class(score)
    sub_html = f'<div class="score-sublabel">{sublabel}</div>' if sublabel else ""
    st.markdown(
        f"""<div class="glass-card score-gauge">
            <div class="score-number {cls}">{score}</div>
            <div class="score-label">{label}</div>
            {sub_html}
        </div>""",
        unsafe_allow_html=True,
    )


def render_flag_list(flags, layer_name: str, icon: str, color_class: str):
    """Render a list of DriftFlags in expandable cards."""
    if not flags:
        st.markdown(
            f"""<div class="empty-state">
                <div class="empty-icon">✅</div>
                <h3>No {layer_name} flags</h3>
                <p>No issues detected in this layer.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    sorted_flags = sorted(flags, key=lambda f: (-f.severity, f.turn))
    for f in sorted_flags:
        sev_color = _sev_color(f.severity)
        with st.expander(
            f"Turn {f.turn}  ·  Severity {f.severity}/10  —  {f.description[:90]}",
            expanded=f.severity >= 7,
        ):
            cols = st.columns([1, 4])
            with cols[0]:
                st.markdown(
                    f'<div style="text-align:center;padding-top:0.5rem;">'
                    f'<span style="font-size:2rem;font-weight:800;color:{sev_color}">{f.severity}</span>'
                    f'<br><span style="font-size:0.7rem;color:#94a3b8">SEVERITY</span></div>',
                    unsafe_allow_html=True,
                )
            with cols[1]:
                st.markdown(f"**{f.description}**")
                if f.instruction_ref:
                    st.markdown(f"📌 **Instruction:** {f.instruction_ref[:200]}")
                if f.evidence:
                    st.markdown(
                        f'<div class="evidence-box">{str(f.evidence)[:500]}</div>',
                        unsafe_allow_html=True,
                    )


def render_corrections(events):
    """Render correction persistence events."""
    if not events:
        st.markdown(
            """<div class="empty-state">
                <div class="empty-icon">🔄</div>
                <h3>No correction events</h3>
                <p>No user corrections detected in this conversation.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    for ev in events:
        status_cls = "held" if ev.held else "failed"
        status_text = "HELD" if ev.held else f"FAILED at turn {ev.failure_turn}"
        status_badge = "badge-green" if ev.held else "badge-red"

        with st.expander(
            f"Correction at turn {ev.correction_turn}  →  Ack turn {ev.acknowledgment_turn}  ·  {status_text}",
            expanded=not ev.held,
        ):
            st.markdown(
                f'<span class="badge {status_badge}">{status_text}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Context:** {ev.instruction[:300]}")
            if not ev.held:
                st.warning(
                    f"The model acknowledged the correction at turn {ev.acknowledgment_turn} "
                    f"but reverted to the same behaviour at turn {ev.failure_turn}."
                )


def render_barometer_signals(signals):
    """Render barometer signals grouped by classification."""
    if not signals:
        st.info("No barometer signals available.")
        return

    for classification, emoji, badge_cls in [("RED", "🔴", "badge-red"),
                                               ("YELLOW", "🟡", "badge-yellow"),
                                               ("GREEN", "🟢", "badge-green")]:
        group = [s for s in signals if s.classification == classification]
        if not group:
            continue
        st.markdown(
            f'<div class="layer-header">'
            f'<span class="layer-icon">{emoji}</span>'
            f'<h3>{classification}</h3>'
            f'<span class="layer-count">{len(group)} signal{"s" if len(group) != 1 else ""}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for s in sorted(group, key=lambda x: x.turn):
            with st.expander(f"Turn {s.turn}  ·  Severity {s.severity}  —  {s.description[:80]}",
                             expanded=(classification == "RED")):
                st.markdown(f"**{s.description}**")
                if s.evidence:
                    st.markdown(
                        f'<div class="evidence-box">{str(s.evidence)[:500]}</div>',
                        unsafe_allow_html=True,
                    )


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — inputs
# ═══════════════════════════════════════════════════════════════════════════

def sidebar_inputs():
    with st.sidebar:
        st.markdown("### Configuration")

        uploaded = st.file_uploader(
            "Chat transcript",
            type=["txt", "json"],
            help="Claude.ai JSON export, or plain text with Human:/Assistant: markers.",
        )

        use_demo = st.checkbox("Use demo conversation", value=(uploaded is None))

        system_prompt = st.text_area(
            "System prompt (optional)",
            height=100,
            placeholder="Paste the system prompt used in the conversation...",
        )

        user_prefs = st.text_area(
            "User preferences (optional)",
            height=80,
            placeholder="Any standing user preferences...",
        )

        st.markdown("---")
        st.markdown("#### Advanced")
        window = st.slider("Window size (turns)", 10, 200, 50, step=5)
        overlap = st.slider("Window overlap", 0, 50, 10, step=5)

        return uploaded, use_demo, system_prompt, user_prefs, window, overlap


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    uploaded, use_demo, system_prompt, user_prefs, window, overlap = sidebar_inputs()

    # ── Hero ────────────────────────────────────────────────────────────
    st.markdown(
        """<div class="hero">
            <h1>Drift Auditor</h1>
            <p>Multi-turn drift diagnostic for AI conversations. Detects sycophancy, instruction omission,
            correction regression, and structural epistemic drift — the failure modes single-turn evals miss.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Load data ───────────────────────────────────────────────────────
    raw_text = None
    demo_system_prompt = ""
    if uploaded is not None:
        raw_text = uploaded.read().decode("utf-8", errors="replace")
    elif use_demo:
        sample_path = Path(__file__).resolve().parent / "examples" / "sample_conversation.txt"
        prompt_path = Path(__file__).resolve().parent / "examples" / "sample_system_prompt.txt"
        if sample_path.exists():
            raw_text = sample_path.read_text()
        if prompt_path.exists():
            demo_system_prompt = prompt_path.read_text()

    if raw_text is None:
        st.markdown(
            """<div class="empty-state" style="margin-top:4rem;">
                <div class="empty-icon">📄</div>
                <h3>Upload a conversation to begin</h3>
                <p>Drop a Claude chat export (.json or .txt) in the sidebar, or enable the demo conversation.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        return

    # Merge system prompt: user-provided takes priority, fall back to demo
    effective_prompt = system_prompt if system_prompt.strip() else demo_system_prompt

    # ── Run audit ───────────────────────────────────────────────────────
    with st.spinner("Auditing conversation..."):
        report = audit_conversation(
            raw_text=raw_text,
            system_prompt=effective_prompt,
            user_preferences=user_prefs,
            window_size=window,
            overlap=overlap,
            conversation_id="dashboard-upload",
        )

    scores = report.summary_scores

    # ═══════════════════════════════════════════════════════════════════
    # Dashboard — Score cards
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("### Overall Assessment")

    cols = st.columns([1.5, 1, 1, 1, 1])
    with cols[0]:
        render_score_card("Overall Drift", scores["overall_drift_score"],
                          _sev_word(scores["overall_drift_score"]))
    with cols[1]:
        render_score_card("Commission", scores["commission_score"],
                          f'{scores["commission_flag_count"]} flags')
    with cols[2]:
        render_score_card("Omission", scores["omission_score"],
                          f'{scores["omission_flag_count"]} flags')
    with cols[3]:
        render_score_card("Persistence", scores["correction_persistence_score"],
                          f'{scores["corrections_failed"]}/{scores["correction_events_total"]} failed')
    with cols[4]:
        render_score_card("Barometer", scores["barometer_score"],
                          f'{scores["barometer_red_count"]} red')

    # ── Quick stats row ─────────────────────────────────────────────────
    stat_cols = st.columns(4)
    stat_cols[0].metric("Total Turns", report.total_turns)
    stat_cols[1].metric("Instructions Found", report.instructions_extracted)
    stat_cols[2].metric("Total Flags",
                        scores["commission_flag_count"] + scores["omission_flag_count"])
    stat_cols[3].metric("Correction Events", scores["correction_events_total"])

    # ═══════════════════════════════════════════════════════════════════
    # Visualisations row
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Drift Timeline")
    st.plotly_chart(build_drift_timeline(report), use_container_width=True)

    vcols = st.columns([1, 1])
    with vcols[0]:
        st.markdown("#### Layer Severity Heatmap")
        st.plotly_chart(build_severity_heatmap(report), use_container_width=True)
    with vcols[1]:
        radar_col, donut_col = st.columns(2)
        with radar_col:
            st.markdown("#### Score Radar")
            st.plotly_chart(build_score_radar(scores), use_container_width=True)
        with donut_col:
            st.markdown("#### Barometer Distribution")
            st.plotly_chart(build_barometer_donut(report), use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # Detailed layer tabs
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### Detailed Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚩 Commission",
        "📜 Omission",
        "🔄 Correction Persistence",
        "🧠 Barometer",
        "📊 Raw Data",
    ])

    with tab1:
        st.markdown(
            '<div class="layer-header"><span class="layer-icon">🚩</span>'
            '<h3>Layer 1 — Commission Drift</h3>'
            f'<span class="layer-count">{len(report.commission_flags)} flags</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Sycophancy, reality distortion, and unwarranted confidence markers.")
        render_flag_list(report.commission_flags, "commission", "🚩", "badge-red")

    with tab2:
        st.markdown(
            '<div class="layer-header"><span class="layer-icon">📜</span>'
            '<h3>Layer 2 — Omission Drift</h3>'
            f'<span class="layer-count">{len(report.omission_flags)} flags</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Instruction violations and missing required behaviours (local heuristics).")
        render_flag_list(report.omission_flags, "omission", "📜", "badge-purple")

    with tab3:
        st.markdown(
            '<div class="layer-header"><span class="layer-icon">🔄</span>'
            '<h3>Layer 3 — Correction Persistence</h3>'
            f'<span class="layer-count">{len(report.correction_events)} events</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Did acknowledged corrections actually hold in subsequent turns?")

        corr_cols = st.columns([2, 1])
        with corr_cols[0]:
            render_corrections(report.correction_events)
        with corr_cols[1]:
            st.markdown("#### Correction Outcomes")
            st.plotly_chart(build_correction_waterfall(report), use_container_width=True)

    with tab4:
        st.markdown(
            '<div class="layer-header"><span class="layer-icon">🧠</span>'
            '<h3>Layer 4 — Structural Drift Barometer</h3>'
            f'<span class="layer-count">{len(report.barometer_signals)} signals</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Epistemic posture analysis — GREEN (healthy), YELLOW (weak), RED (drifted).")
        render_barometer_signals(report.barometer_signals)

    with tab5:
        st.markdown("#### Export")
        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.download_button(
                "Download JSON Report",
                data=report_to_json(report),
                file_name="drift_audit_report.json",
                mime="application/json",
            )
        with dl_cols[1]:
            st.download_button(
                "Download Text Report",
                data=format_report(report),
                file_name="drift_audit_report.txt",
                mime="text/plain",
            )

        st.markdown("#### Summary Scores")
        st.json(scores)

        st.markdown("#### Metadata")
        st.json(report.metadata)


if __name__ == "__main__":
    main()
