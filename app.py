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
/* --- Base theme overrides --- */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #121228 50%, #0d0d20 100%);
}

/* Glass card */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 8px;
}

.glass-card-hero {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 28px 20px;
    text-align: center;
}

.metric-value {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    text-align: center;
}

.metric-value-hero {
    font-size: 4rem;
    font-weight: 800;
    line-height: 1.1;
    text-align: center;
}

.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: rgba(255, 255, 255, 0.5);
    text-align: center;
    margin-bottom: 6px;
}

.metric-sub {
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.4);
    text-align: center;
    margin-top: 4px;
}

.status-held {
    color: #00d4ff;
    font-weight: 700;
}

.status-failed {
    color: #ff3366;
    font-weight: 700;
}

.event-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
}

/* Sidebar tweaks */
section[data-testid="stSidebar"] {
    background: rgba(10, 10, 30, 0.95);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    padding: 8px 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper functions — all typed, all pure
# ---------------------------------------------------------------------------

def score_color(score: int) -> str:
    """Map a 1-10 severity score to a hex color."""
    if score <= 3:
        return "#00d4ff"
    elif score <= 5:
        return "#ffb300"
    elif score <= 7:
        return "#ff6b35"
    else:
        return "#ff3366"


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
    font=dict(color="rgba(255,255,255,0.8)", family="Inter, sans-serif", size=12),
    margin=dict(l=50, r=30, t=40, b=40),
)


def render_metric_card(label: str, score: int, subtitle: str, hero: bool = False) -> None:
    """Render a glass metric card with a colored score value."""
    color = score_color(score)
    card_cls = "glass-card-hero" if hero else "glass-card"
    val_cls = "metric-value-hero" if hero else "metric-value"
    shadow = f"0 0 30px {color}40" if hero else "none"
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
            marker=dict(size=11, color="#ff3366", symbol="diamond", line=dict(width=1, color="#ff336680")),
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
            marker=dict(size=11, color="#ffb300", symbol="triangle-up", line=dict(width=1, color="#ffb30080")),
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
            marker=dict(size=9, color="#ff3366", symbol="circle", opacity=0.4),
            hovertext=[f"T{s.turn}: {s.description[:60]}" for s in red_signals],
            hoverinfo="text",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(title="Severity", range=[0, 11], gridcolor="rgba(255,255,255,0.06)", zeroline=False),
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
    colors = {"GREEN": "#00d4ff", "YELLOW": "#ffb300", "RED": "#ff3366"}

    fig = go.Figure()
    for s in report.barometer_signals:
        fig.add_trace(go.Bar(
            x=[1], y=["Epistemic Posture"],
            orientation="h",
            marker_color=colors.get(s.classification, "#ffb300"),
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

    colors = {"GREEN": "#00d4ff", "YELLOW": "#ffb300", "RED": "#ff3366"}
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
            marker=dict(size=14, color="#ff6b35", symbol="circle"),
            showlegend=False, hoverinfo="text",
            hovertext=f"User corrected at T{event.correction_turn}",
        ))

        # Acknowledgment point
        fig.add_trace(go.Scatter(
            x=[event.acknowledgment_turn], y=[y_pos],
            mode="markers+text", text=["Ack"],
            textposition="top center", textfont=dict(size=10, color="rgba(255,255,255,0.6)"),
            marker=dict(size=14, color="#00d4ff", symbol="circle"),
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
                textposition="top center", textfont=dict(size=10, color="#ff3366"),
                marker=dict(size=16, color="#ff3366", symbol="x"),
                showlegend=False, hoverinfo="text",
                hovertext=f"Correction failed at T{event.failure_turn}",
            ))
            fig.add_trace(go.Scatter(
                x=[event.acknowledgment_turn, event.failure_turn],
                y=[y_pos, y_pos],
                mode="lines",
                line=dict(color="#ff3366", width=2, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
        else:
            # Held indicator
            fig.add_trace(go.Scatter(
                x=[event.acknowledgment_turn + 2], y=[y_pos],
                mode="markers+text", text=["Held \u2713"],
                textposition="middle right", textfont=dict(size=11, color="#00d4ff"),
                marker=dict(size=12, color="#00d4ff", symbol="circle"),
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
        line=dict(color="#ffb300", width=2),
        marker=dict(size=8, color="#ffb300"),
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

# Stats row
st.markdown("")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Total Turns", report.total_turns)
s2.metric("Instructions Extracted", report.instructions_extracted)
s3.metric("Total Flags", scores.get("commission_flag_count", 0) + scores.get("omission_flag_count", 0))
s4.metric("Corrections Tracked", scores.get("correction_events_total", 0))


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

tab_baro, tab_persist, tab_comm, tab_omit = st.tabs(
    ["\U0001f9e0 Barometer", "\U0001f504 Persistence", "\U0001f6a9 Commission", "\U0001f4dc Omission"]
)

# --- Barometer Tab ---
with tab_baro:
    st.caption("Structural epistemic posture analysis per assistant turn.")

    red_signals = [s for s in report.barometer_signals if s.classification == "RED"]
    yellow_signals = [s for s in report.barometer_signals if s.classification == "YELLOW"]
    green_signals = [s for s in report.barometer_signals if s.classification == "GREEN"]

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 3px solid #ff3366">
            <div class="metric-label">RED</div>
            <div class="metric-value" style="color: #ff3366">{len(red_signals)}</div>
            <div class="metric-sub">Active structural drift</div>
        </div>""", unsafe_allow_html=True)
    with bc2:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 3px solid #ffb300">
            <div class="metric-label">YELLOW</div>
            <div class="metric-value" style="color: #ffb300">{len(yellow_signals)}</div>
            <div class="metric-sub">Passive drift / hedging</div>
        </div>""", unsafe_allow_html=True)
    with bc3:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 3px solid #00d4ff">
            <div class="metric-label">GREEN</div>
            <div class="metric-value" style="color: #00d4ff">{len(green_signals)}</div>
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
            <div class="event-card" style="border-left: 3px solid {'#00d4ff' if event.held else '#ff3366'}">
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
