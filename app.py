"""
Drift Auditor — Streamlit Dashboard
Interactive multi-turn drift analysis for Claude conversations.
"""
import sys
import os
import json
from collections import defaultdict

# Make src/ importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
import plotly.graph_objects as go

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

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drift Auditor",
    page_icon="\U0001f4e1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* --- Base theme: warm dark with amber accents (Bakers Agent palette) --- */
.stApp {
    background: #0a0807;
    font-family: 'DM Sans', sans-serif;
    color: #e8dfd0;
}

/* Glass card — warm dark */
.glass-card {
    background: #12100f;
    border: 1px solid #2a2623;
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 8px;
}

.glass-card-hero {
    background: #12100f;
    border: 1px solid #3a3530;
    border-radius: 4px;
    padding: 28px 20px;
    text-align: center;
    box-shadow: 0 0 30px rgba(245, 158, 11, 0.08);
}

.metric-value {
    font-size: 2.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    text-align: center;
}

.metric-value-hero {
    font-size: 4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    text-align: center;
}

.metric-label {
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #8b7d6b;
    text-align: center;
    margin-bottom: 6px;
}

.metric-sub {
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: #8b7d6b;
    text-align: center;
    margin-top: 4px;
}

.status-held {
    color: #22c55e;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.status-failed {
    color: #ef4444;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.event-card {
    background: #12100f;
    border: 1px solid #2a2623;
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 10px;
}

/* Sidebar — warm dark */
section[data-testid="stSidebar"] {
    background: #0a0807;
    border-right: 1px solid #2a2623;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #2a2623;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 4px 4px 0 0;
    padding: 8px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8b7d6b;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #12100f;
    color: #f59e0b;
    border-bottom: 2px solid #f59e0b;
}

/* Headers */
h1, h2, h3 {
    font-family: 'JetBrains Mono', monospace;
    color: #e8dfd0;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #c4b8a3;
}

/* Metric overrides */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    color: #f59e0b;
}

[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8b7d6b;
}

/* Button styling */
.stButton > button {
    background: #12100f;
    border: 1px solid #2a2623;
    color: #e8dfd0;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.78rem;
    border-radius: 4px;
    transition: all 0.2s;
}

.stButton > button:hover {
    border-color: #f59e0b;
    color: #f59e0b;
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.15);
}

/* Download button */
.stDownloadButton > button {
    background: #12100f;
    border: 1px solid #f59e0b;
    color: #f59e0b;
    font-family: 'JetBrains Mono', monospace;
    border-radius: 4px;
}

/* Text area / inputs */
.stTextArea textarea, .stTextInput input {
    background: #12100f;
    border: 1px solid #2a2623;
    color: #e8dfd0;
    font-family: 'DM Sans', sans-serif;
    border-radius: 4px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #2a2623;
    border-radius: 4px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0807; }
::-webkit-scrollbar-thumb { background: #2a2623; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #f59e0b; }

/* Info/Success/Warning/Error boxes */
.stAlert {
    border-radius: 4px;
    font-family: 'DM Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper functions — all typed, all pure
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


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c4b8a3", family="JetBrains Mono, DM Sans, sans-serif", size=12),
    margin=dict(l=50, r=30, t=40, b=40),
)


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
            marker=dict(size=11, color="#ef4444", symbol="diamond", line=dict(width=1, color="#ef444480")),
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
            marker=dict(size=11, color="#f59e0b", symbol="triangle-up", line=dict(width=1, color="#f59e0b80")),
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
# Cached audit function
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Auditing conversation\u2026")
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
    st.markdown("## \U0001f4e1 Drift Auditor")
    st.caption("Multi-turn drift diagnostic tool")

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


# ---------------------------------------------------------------------------
# Determine input source
# ---------------------------------------------------------------------------

raw_bytes: bytes | None = None
conv_id = "unknown"

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    conv_id = uploaded.name
elif st.session_state.sample_bytes is not None:
    raw_bytes = st.session_state.sample_bytes
    conv_id = "sample_conversation.txt"

if raw_bytes is None:
    st.markdown("## \U0001f4e1 Drift Auditor")
    st.markdown(
        "Upload a Claude conversation export or **load the sample** from the sidebar "
        "to analyze multi-turn drift patterns."
    )
    st.info("Supports Claude.ai JSON exports and plain text transcripts with role markers.")
    st.stop()


# ---------------------------------------------------------------------------
# Run audit
# ---------------------------------------------------------------------------

try:
    report: AuditReport = run_audit(
        raw_bytes, system_prompt, preferences, window_size, overlap, conv_id
    )
except Exception as exc:
    st.error(f"Unable to audit this conversation: {exc}")
    st.stop()

if report.total_turns == 0:
    st.error("Could not parse any turns from the uploaded file. Supported formats: Claude.ai JSON export, plain text with Human:/Assistant: markers.")
    st.stop()

scores = report.summary_scores


# ---------------------------------------------------------------------------
# Dashboard — Summary metrics
# ---------------------------------------------------------------------------

st.markdown("## \U0001f4e1 Drift Audit Results")

# Hero metric row
hero_col, c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1, 1])

with hero_col:
    overall = scores.get("overall_drift_score", 1)
    render_metric_card(
        "Overall Drift", overall,
        score_label(overall),
        hero=True,
    )

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

# Structural score row
st.markdown("")
struct_col, void_col, conflict_col, shadow_col, op_col = st.columns(5)
with struct_col:
    render_metric_card("Structural", scores.get("structural_score", 1),
                       f"composite of new detectors")
with void_col:
    st.metric("Void Events", scores.get("void_events_count", 0))
with conflict_col:
    st.metric("Conflict Pairs", scores.get("conflict_pairs_count", 0))
with shadow_col:
    st.metric("Shadow Patterns", scores.get("shadow_patterns_count", 0))
with op_col:
    effective = scores.get("op_moves_effective", 0)
    total_moves = scores.get("op_moves_total", 0)
    st.metric("Operator Moves", f"{effective}/{total_moves} effective")

# Stats row
st.markdown("")
s1, s2, s3, s4, s5, s6 = st.columns(6)
s1.metric("Total Turns", report.total_turns)
s2.metric("Instructions", report.instructions_extracted)
s3.metric("Total Flags", scores.get("commission_flag_count", 0) + scores.get("omission_flag_count", 0))
s4.metric("Corrections", scores.get("correction_events_total", 0))
s5.metric("Instrs Active", scores.get("instructions_active", 0))
s6.metric("Instrs Dropped", scores.get("instructions_omitted", 0))


# ---------------------------------------------------------------------------
# Drift Timeline
# ---------------------------------------------------------------------------

st.subheader("Drift Timeline")
timeline_fig = build_timeline_fig(report)
if report.commission_flags or report.omission_flags or report.correction_events:
    st.plotly_chart(timeline_fig, use_container_width=True, config={"displaylogo": False})
else:
    st.success("No drift flags detected across the conversation.")

# Barometer strip
if report.barometer_signals:
    strip_fig = build_barometer_strip(report)
    st.plotly_chart(strip_fig, use_container_width=True, config={"displayModeBar": False})


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
import glob as glob_mod

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
    import pandas as pd
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
