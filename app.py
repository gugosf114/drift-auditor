"""
Drift Auditor — Professional Dashboard
=======================================
Multi-turn drift analysis for AI conversations.
Built for compliance officers, AI safety teams, and enterprise IT.

Author: George Abrahamyants + Claude (Opus 4.6)
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

from drift_auditor import (
    audit_conversation, AuditReport, DriftFlag, CorrectionEvent,
    BarometerSignal, InstructionLifecycle, ConflictPair, ShadowPattern,
    OpMove, VoidEvent, DriftTag, OperatorRule, format_report, report_to_json,
)
from operator_load import compute_operator_load, OperatorLoadMetrics
from visualizations import (
    score_color, score_label, coupling_label, coupling_color,
    build_barometer_timeline, build_drift_timeline, build_flag_summary_chart,
    build_lifecycle_timeline, build_decay_curves,
    build_operator_move_timeline, build_rule_frequency,
    build_operator_load_comparison, build_persistence_fig,
    CHART_CONFIG, RED, AMBER, GREEN, PURPLE, BLUE, SLATE, LIGHT, MUTED,
    BG, CARD_BG, BORDER,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drift Auditor",
    page_icon="https://raw.githubusercontent.com/gugosf114/drift-auditor/main/docs/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

VERSION = "2.0"

# ---------------------------------------------------------------------------
# CSS — Professional dark theme
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Base */
.stApp {{
    background: {BG};
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}

/* Header bar */
.header-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 24px;
}}
.header-brand {{
    font-size: 1.3rem;
    font-weight: 700;
    color: {LIGHT};
    letter-spacing: -0.5px;
}}
.header-brand span {{
    color: {PURPLE};
}}
.header-version {{
    font-size: 0.75rem;
    color: {MUTED};
    font-family: 'JetBrains Mono', monospace;
    background: {CARD_BG};
    padding: 3px 10px;
    border-radius: 12px;
    border: 1px solid {BORDER};
}}

/* Metric cards */
.metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}}
.metric-card-hero {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 24px 20px;
    text-align: center;
}}
.metric-number {{
    font-size: 2.4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}}
.metric-number-hero {{
    font-size: 3.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.2;
}}
.metric-label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: {MUTED};
    margin-bottom: 6px;
}}
.metric-sub {{
    font-size: 0.75rem;
    color: {SLATE};
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}}

/* Section headers */
.section-header {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {LIGHT};
    padding-bottom: 8px;
    border-bottom: 1px solid {BORDER};
    margin: 24px 0 16px 0;
    letter-spacing: -0.3px;
}}

/* Badges */
.badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
}}
.badge-red {{ background: rgba(239,68,68,0.15); color: {RED}; border: 1px solid rgba(239,68,68,0.3); }}
.badge-amber {{ background: rgba(245,158,11,0.15); color: {AMBER}; border: 1px solid rgba(245,158,11,0.3); }}
.badge-green {{ background: rgba(16,185,129,0.15); color: {GREEN}; border: 1px solid rgba(16,185,129,0.3); }}
.badge-purple {{ background: rgba(124,58,237,0.15); color: {PURPLE}; border: 1px solid rgba(124,58,237,0.3); }}
.badge-blue {{ background: rgba(59,130,246,0.15); color: {BLUE}; border: 1px solid rgba(59,130,246,0.3); }}
.badge-slate {{ background: rgba(100,116,139,0.15); color: {SLATE}; border: 1px solid rgba(100,116,139,0.3); }}

/* Detail cards */
.detail-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
}}
.detail-card-left {{
    border-left: 3px solid;
}}

/* Empty state */
.empty-state {{
    text-align: center;
    padding: 60px 20px;
    color: {MUTED};
}}
.empty-state h3 {{
    color: {LIGHT};
    margin-bottom: 8px;
    font-weight: 600;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {BG};
    border-right: 1px solid {BORDER};
}}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 6px 6px 0 0;
    padding: 10px 18px;
    font-size: 0.82rem;
    font-weight: 500;
    color: {MUTED};
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: {CARD_BG};
    color: {PURPLE};
    border-bottom: 2px solid {PURPLE};
}}

/* Buttons */
.stButton > button {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    color: {LIGHT};
    font-weight: 500;
    border-radius: 6px;
    transition: all 0.15s;
}}
.stButton > button:hover {{
    border-color: {PURPLE};
    color: {PURPLE};
}}
.stDownloadButton > button {{
    background: rgba(124,58,237,0.1);
    border: 1px solid {PURPLE};
    color: {PURPLE};
    font-weight: 500;
    border-radius: 6px;
}}

/* Inputs */
.stTextArea textarea, .stTextInput input {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    color: {LIGHT};
    border-radius: 6px;
}}
[data-testid="stFileUploader"] {{
    border: 2px dashed {BORDER};
    border-radius: 8px;
    padding: 8px;
}}

/* Metric overrides */
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {PURPLE}; }}

/* Hide Streamlit branding */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: metric card HTML
# ---------------------------------------------------------------------------

def render_metric(label: str, value, subtitle: str = "", color: str = None,
                  hero: bool = False):
    """Render a professional metric card."""
    if color is None:
        color = LIGHT
    cls = "metric-card-hero" if hero else "metric-card"
    num_cls = "metric-number-hero" if hero else "metric-number"
    st.markdown(f"""
    <div class="{cls}">
        <div class="metric-label">{label}</div>
        <div class="{num_cls}" style="color: {color}">{value}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def severity_badge(score: int) -> str:
    """Return HTML badge for severity."""
    color = score_color(score)
    if score <= 3: cls = "badge-green"
    elif score <= 6: cls = "badge-amber"
    else: cls = "badge-red"
    return f'<span class="badge {cls}">{score}/10</span>'


def status_badge(status: str) -> str:
    """Return HTML badge for status."""
    badge_map = {
        "active": ("ACTIVE", "badge-green"),
        "omitted": ("OMITTED", "badge-red"),
        "degraded": ("DEGRADED", "badge-amber"),
        "superseded": ("SUPERSEDED", "badge-slate"),
    }
    text, cls = badge_map.get(status, (status.upper(), "badge-slate"))
    return f'<span class="badge {cls}">{text}</span>'


def coupling_badge(score: float) -> str:
    """Return HTML badge for coupling score."""
    label = coupling_label(score)
    if label == "HIGH": cls = "badge-red"
    elif label == "MEDIUM": cls = "badge-amber"
    else: cls = "badge-green"
    return f'<span class="badge {cls}">{label} ({score:.2f})</span>'


# ---------------------------------------------------------------------------
# Cached audit runner
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_audit(file_bytes: bytes, system_prompt: str, preferences: str,
              window_size: int, overlap: int, conversation_id: str) -> AuditReport:
    """Run the drift audit with caching."""
    raw = file_bytes.decode("utf-8", errors="ignore")
    return audit_conversation(
        raw_text=raw, system_prompt=system_prompt,
        user_preferences=preferences, window_size=window_size,
        overlap=overlap, conversation_id=conversation_id,
    )


# ---------------------------------------------------------------------------
# Sidebar — Upload & Config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-size: 1.15rem; font-weight: 700; color: {LIGHT}; letter-spacing: -0.3px;">
            Drift Auditor
        </div>
        <div style="font-size: 0.72rem; color: {MUTED}; margin-top: 2px;">
            AI Conversation Drift Analysis &nbsp;
            <span style="background: {CARD_BG}; padding: 1px 8px; border-radius: 10px;
                         border: 1px solid {BORDER}; font-family: 'JetBrains Mono', monospace;">
                v{VERSION}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-header">Upload</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload conversation",
        type=["txt", "json"],
        help="Supports Claude.ai JSON exports and plain text transcripts (.txt)",
        label_visibility="collapsed",
    )

    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
    sample_conv_path = os.path.join(sample_dir, "sample_conversation.txt")
    sample_prompt_path = os.path.join(sample_dir, "sample_system_prompt.txt")

    if "sample_bytes" not in st.session_state:
        st.session_state.sample_bytes = None
    if "sample_prompt" not in st.session_state:
        st.session_state.sample_prompt = ""

    if st.button("Load Sample Conversation", use_container_width=True):
        try:
            with open(sample_conv_path, "r", encoding="utf-8") as f:
                st.session_state.sample_bytes = f.read().encode("utf-8")
            with open(sample_prompt_path, "r", encoding="utf-8") as f:
                st.session_state.sample_prompt = f.read()
        except FileNotFoundError:
            st.error("Sample files not found in examples/ directory.")

    st.markdown(f'<div class="section-header">Configuration</div>', unsafe_allow_html=True)

    default_prompt = st.session_state.sample_prompt if st.session_state.sample_bytes and not uploaded else ""
    system_prompt = st.text_area("System Prompt", value=default_prompt, height=80,
                                  placeholder="Optional: paste system prompt...")
    preferences = st.text_area("User Preferences", height=60,
                                placeholder="Optional: user preferences...")

    st.markdown(f'<div class="section-header">Analysis Parameters</div>', unsafe_allow_html=True)
    window_size = st.slider("Window Size", 10, 100, 50, step=5)
    overlap = st.slider("Overlap", 0, 25, 10, step=1)


# ---------------------------------------------------------------------------
# Determine input & run audit
# ---------------------------------------------------------------------------

raw_bytes = None
conv_id = "unknown"

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    conv_id = uploaded.name
elif st.session_state.sample_bytes is not None:
    raw_bytes = st.session_state.sample_bytes
    conv_id = "sample_conversation.txt"

# Empty state
if raw_bytes is None:
    st.markdown(f"""
    <div class="header-bar">
        <div class="header-brand"><span>Drift</span> Auditor</div>
        <div class="header-version">v{VERSION}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="empty-state">
        <h3>Upload a conversation to begin</h3>
        <p>Drag and drop a Claude.ai JSON export or plain text transcript into the sidebar,<br>
        or click "Load Sample Conversation" to see the tool in action.</p>
        <p style="margin-top: 24px; font-size: 0.8rem;">
            Supported formats: <code>.json</code> (Claude.ai export) &nbsp;&bull;&nbsp;
            <code>.txt</code> (Human:/Assistant: markers)
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Run audit with progress
with st.spinner("Analyzing conversation..."):
    try:
        report = run_audit(raw_bytes, system_prompt, preferences, window_size, overlap, conv_id)
    except Exception as exc:
        st.error(f"Unable to audit this conversation: {exc}")
        st.stop()

if report.total_turns == 0:
    st.error("Could not parse any turns. Supported: Claude.ai JSON export, plain text with Human:/Assistant: markers.")
    st.stop()

scores = report.summary_scores


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="header-bar">
    <div class="header-brand"><span>Drift</span> Auditor</div>
    <div class="header-version">v{VERSION} &nbsp;&bull;&nbsp; {report.total_turns} turns &nbsp;&bull;&nbsp; {report.instructions_extracted} instructions</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------

tab_dash, tab_lifecycle, tab_operator, tab_flags, tab_export = st.tabs([
    "Dashboard", "Instruction Lifecycle", "Operator Load", "Detailed Flags", "Export"
])


# =========================================================================
# TAB 1: Dashboard Overview
# =========================================================================
with tab_dash:

    # --- Row 1: Hero metrics ---
    overall = scores.get("overall_drift_score", 1)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric("Overall Drift Score", overall, score_label(overall),
                      color=score_color(overall), hero=True)
    with c2:
        # Compute operator load for current conversation
        conv_result = {
            "message_count": report.total_turns,
            "corrections_total": len(report.correction_events),
            "corrections_failed": sum(1 for e in report.correction_events if not e.held),
            "op_moves": len(report.op_moves),
            "op_moves_effective": scores.get("op_moves_effective", 0),
            "instructions_extracted": report.instructions_extracted,
            "instructions_omitted": scores.get("instructions_omitted", 0),
            "void_events": len(report.void_events),
            "commission_flags": len(report.commission_flags),
            "omission_flags": len(report.omission_flags),
        }
        ol_metrics = compute_operator_load([conv_result], "current")
        oli = ol_metrics.operator_load_index
        oli_label = "LOW" if oli < 0.05 else "MEDIUM" if oli < 0.15 else "HIGH"
        oli_color = GREEN if oli < 0.05 else AMBER if oli < 0.15 else RED
        render_metric("Operator Load Index", f"{oli:.3f}", oli_label,
                      color=oli_color, hero=True)

    with c3:
        # Operator Fatigue: weighted cognitive cost
        fatigue = round(ol_metrics.alignment_tax * 100, 1)
        fatigue_color = GREEN if fatigue < 10 else AMBER if fatigue < 30 else RED
        render_metric("Alignment Tax", f"{fatigue}%", "of conversation spent steering",
                      color=fatigue_color, hero=True)

    with c4:
        survival = round(ol_metrics.instruction_survival_rate * 100, 1)
        surv_color = GREEN if survival > 80 else AMBER if survival > 50 else RED
        render_metric("Instruction Survival", f"{survival}%", "of instructions persisted",
                      color=surv_color, hero=True)

    st.markdown("")

    # --- Row 2: Signature charts ---
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown(f'<div class="section-header">Drift Barometer</div>', unsafe_allow_html=True)
        if report.barometer_signals:
            st.plotly_chart(build_barometer_timeline(report), use_container_width=True,
                            config=CHART_CONFIG)
        else:
            st.info("No barometer signals generated for this conversation.")

    with chart_right:
        st.markdown(f'<div class="section-header">Drift Timeline</div>', unsafe_allow_html=True)
        if report.commission_flags or report.omission_flags or report.correction_events:
            st.plotly_chart(build_drift_timeline(report), use_container_width=True,
                            config=CHART_CONFIG)
        else:
            st.success("No drift flags detected.")

    st.markdown("")

    # --- Row 3: Layer scores + Flag summary ---
    score_col, summary_col = st.columns([3, 2])

    with score_col:
        st.markdown(f'<div class="section-header">Layer Scores</div>', unsafe_allow_html=True)
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        with lc1:
            s = scores.get("commission_score", 1)
            render_metric("L1: Commission", s, f"{scores.get('commission_flag_count', 0)} flags",
                          color=score_color(s))
        with lc2:
            s = scores.get("omission_score", 1)
            render_metric("L2: Omission", s, f"{scores.get('omission_flag_count', 0)} flags",
                          color=score_color(s))
        with lc3:
            s = scores.get("correction_persistence_score", 1)
            failed = scores.get("corrections_failed", 0)
            total = scores.get("correction_events_total", 0)
            render_metric("L3: Persistence", s, f"{failed}/{total} failed",
                          color=score_color(s))
        with lc4:
            s = scores.get("barometer_score", 1)
            render_metric("L4: Barometer", s,
                          f"{scores.get('barometer_red_count', 0)} RED signals",
                          color=score_color(s))
        with lc5:
            s = scores.get("structural_score", 1)
            render_metric("Structural", s,
                          f"{scores.get('conflict_pairs_count', 0)} conflicts",
                          color=score_color(s))

    with summary_col:
        st.markdown(f'<div class="section-header">Flag Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(build_flag_summary_chart(report), use_container_width=True,
                        config=CHART_CONFIG)

    # --- Row 4: Quick stats ---
    st.markdown("")
    qs1, qs2, qs3, qs4, qs5, qs6 = st.columns(6)
    qs1.metric("Total Turns", report.total_turns)
    qs2.metric("Instructions", report.instructions_extracted)
    qs3.metric("Total Flags", scores.get("commission_flag_count", 0) + scores.get("omission_flag_count", 0))
    qs4.metric("Void Events", scores.get("void_events_count", 0))
    qs5.metric("Shadow Patterns", scores.get("shadow_patterns_count", 0))
    qs6.metric("Operator Moves", f"{scores.get('op_moves_effective', 0)}/{scores.get('op_moves_total', 0)}")


# =========================================================================
# TAB 2: Instruction Lifecycle
# =========================================================================
with tab_lifecycle:

    if report.instruction_lifecycles:
        # Lifecycle timeline chart
        st.markdown(f'<div class="section-header">Instruction Timeline</div>', unsafe_allow_html=True)
        st.plotly_chart(build_lifecycle_timeline(report), use_container_width=True,
                        config=CHART_CONFIG)

        # Instruction table
        st.markdown(f'<div class="section-header">Instruction Details</div>', unsafe_allow_html=True)

        for lc in report.instruction_lifecycles:
            sc = {"active": GREEN, "omitted": RED, "degraded": AMBER, "superseded": SLATE}.get(lc.status, SLATE)
            timeline_text = f"Given T{lc.turn_given}"
            if lc.turn_last_followed is not None:
                timeline_text += f" &rarr; Followed T{lc.turn_last_followed}"
            if lc.turn_first_omitted is not None:
                timeline_text += f' &rarr; <span style="color:{RED}">Omitted T{lc.turn_first_omitted}</span>'

            st.markdown(f"""
            <div class="detail-card detail-card-left" style="border-left-color: {sc}">
                <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 6px;">
                    {status_badge(lc.status)}
                    {coupling_badge(lc.coupling_score)}
                    <span class="badge badge-slate">{lc.source}</span>
                    <span class="badge badge-slate">{lc.position_in_conversation}</span>
                </div>
                <div style="color: {LIGHT}; font-size: 0.88rem; margin-bottom: 4px;">
                    {lc.instruction_text[:150]}{'...' if len(lc.instruction_text) > 150 else ''}
                </div>
                <div style="font-size: 0.75rem; color: {MUTED};">{timeline_text}</div>
            </div>
            """, unsafe_allow_html=True)

        # Decay curves
        st.markdown(f'<div class="section-header">Instruction Decay Curves</div>', unsafe_allow_html=True)
        st.plotly_chart(build_decay_curves(report), use_container_width=True,
                        config=CHART_CONFIG)

        # Positional analysis
        pa = report.positional_analysis
        if pa and any(isinstance(v, dict) and v.get("count", 0) > 0
                      for k, v in pa.items() if k != "hypothesis_supported"):
            st.markdown(f'<div class="section-header">Positional Analysis</div>', unsafe_allow_html=True)
            pos_cols = st.columns(3)
            for col, pos_name in zip(pos_cols, ["edge_start", "middle", "edge_end"]):
                with col:
                    data = pa.get(pos_name, {})
                    if isinstance(data, dict):
                        rate = data.get("rate", 0)
                        omitted = data.get("omitted", 0)
                        count = data.get("count", 0)
                        label_map = {"edge_start": "Start (first 20%)", "middle": "Middle (20-80%)", "edge_end": "End (last 20%)"}
                        rate_color = RED if rate > 0.5 else AMBER if rate > 0.2 else GREEN
                        render_metric(label_map.get(pos_name, pos_name),
                                      f"{rate:.0%}", f"{omitted}/{count} omitted",
                                      color=rate_color)

            hyp = pa.get("hypothesis_supported")
            if hyp is not None:
                if hyp:
                    st.warning("Hypothesis SUPPORTED: Middle instructions are omitted more than edge instructions.")
                else:
                    st.success("Hypothesis NOT SUPPORTED: Edge and middle instructions omitted at similar rates.")
    else:
        st.info("No instructions extracted to track.")


# =========================================================================
# TAB 3: Operator Load
# =========================================================================
with tab_operator:

    # Per-conversation operator load metrics
    st.markdown(f'<div class="section-header">Operator Load Metrics</div>', unsafe_allow_html=True)

    om1, om2, om3, om4, om5 = st.columns(5)
    with om1:
        render_metric("Load Index", f"{ol_metrics.operator_load_index:.3f}",
                      "interventions/msg", color=oli_color)
    with om2:
        render_metric("Alignment Tax", f"{ol_metrics.alignment_tax:.1%}",
                      "steering overhead", color=fatigue_color)
    with om3:
        dr_color = GREEN if ol_metrics.drift_resistance > 10 else AMBER if ol_metrics.drift_resistance > 5 else RED
        render_metric("Drift Resistance", f"{ol_metrics.drift_resistance:.1f}",
                      "turns before intervention", color=dr_color)
    with om4:
        ce = ol_metrics.correction_efficiency
        ce_color = GREEN if ce > 0.8 else AMBER if ce > 0.5 else RED
        render_metric("Correction Efficiency", f"{ce:.0%}",
                      "corrections that held", color=ce_color)
    with om5:
        ss = ol_metrics.self_sufficiency_score
        ss_color = GREEN if ss > 80 else AMBER if ss > 50 else RED
        render_metric("Self-Sufficiency", f"{ss:.0f}%",
                      "autonomy score", color=ss_color)

    st.markdown("")

    # Operator move timeline + rule frequency side by side
    op_left, op_right = st.columns(2)

    with op_left:
        st.markdown(f'<div class="section-header">Rule Frequency</div>', unsafe_allow_html=True)
        if report.op_moves:
            st.plotly_chart(build_rule_frequency(report), use_container_width=True,
                            config=CHART_CONFIG)
        else:
            st.info("No operator moves detected.")

    with op_right:
        st.markdown(f'<div class="section-header">Correction Persistence</div>', unsafe_allow_html=True)
        if report.correction_events:
            st.plotly_chart(build_persistence_fig(report), use_container_width=True,
                            config=CHART_CONFIG)
        else:
            st.info("No correction events detected.")

    # Operator move details
    if report.op_moves:
        st.markdown(f'<div class="section-header">Operator Move Details</div>', unsafe_allow_html=True)
        RULE_DESCRIPTIONS = {
            "R01_ANCHOR": "Set explicit instruction at start",
            "R02_ECHO_CHECK": "Ask model to restate instructions",
            "R03_BOUNDARY": "Enforce scope limits",
            "R04_CORRECTION": "Direct error correction",
            "R05_NOT_SHOT": "Misinterpretation from voice/typo",
            "R06_CONTRASTIVE": "What changed between X and Y?",
            "R07_RESET": "Start over / full context reset",
            "R08_DECOMPOSE": "Break instruction into steps",
            "R09_EVIDENCE": "Show me where / proof request",
            "R10_META_CALL": "Calling out the drift pattern",
            "R11_TIGER_TAMER": "Active reinforcement",
            "R12_KILL_SWITCH": "Abandon thread / hard stop",
        }
        for m in report.op_moves:
            eff_color = {"effective": GREEN, "partially_effective": AMBER,
                         "ineffective": RED, "unknown": SLATE}.get(m.effectiveness, SLATE)
            eff_badge = f"badge-green" if m.effectiveness == "effective" else \
                        f"badge-amber" if m.effectiveness == "partially_effective" else \
                        f"badge-red" if m.effectiveness == "ineffective" else "badge-slate"
            st.markdown(f"""
            <div class="detail-card detail-card-left" style="border-left-color: {eff_color}">
                <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 4px;">
                    <span class="badge badge-purple">{m.rule}</span>
                    <span class="badge {eff_badge}">{m.effectiveness.upper()}</span>
                    <span style="color: {MUTED}; font-size: 0.75rem;">Turn {m.turn}</span>
                </div>
                <div style="color: {SLATE}; font-size: 0.78rem;">{RULE_DESCRIPTIONS.get(m.rule, '')}</div>
                <div style="color: {LIGHT}; font-size: 0.82rem; margin-top: 4px;">{m.target_behavior[:150]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Cross-model comparison
    st.markdown(f'<div class="section-header">Cross-Model Comparison</div>', unsafe_allow_html=True)

    leaderboard_data = {}
    for batch_dir, model_key in [("batch_results", "claude"), ("batch_results_chatgpt", "chatgpt")]:
        batch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), batch_dir)
        json_files = glob_mod.glob(os.path.join(batch_path, "batch_results*.json"))
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    results = json.load(f)
                if results:
                    leaderboard_data[model_key] = results
            except Exception:
                pass

    if len(leaderboard_data) >= 2:
        claude_ol = compute_operator_load(leaderboard_data.get("claude", []), "Claude")
        chatgpt_ol = compute_operator_load(leaderboard_data.get("chatgpt", []), "ChatGPT")

        comp_stats_claude = {
            "operator_load_index": claude_ol.operator_load_index,
            "alignment_tax": claude_ol.alignment_tax,
            "self_sufficiency_score": claude_ol.self_sufficiency_score,
            "instruction_survival_rate": claude_ol.instruction_survival_rate,
            "correction_efficiency": claude_ol.correction_efficiency,
            "avg_drift_score": 0,
        }
        comp_stats_chatgpt = {
            "operator_load_index": chatgpt_ol.operator_load_index,
            "alignment_tax": chatgpt_ol.alignment_tax,
            "self_sufficiency_score": chatgpt_ol.self_sufficiency_score,
            "instruction_survival_rate": chatgpt_ol.instruction_survival_rate,
            "correction_efficiency": chatgpt_ol.correction_efficiency,
            "avg_drift_score": 0,
        }
        st.plotly_chart(build_operator_load_comparison(comp_stats_claude, comp_stats_chatgpt),
                        use_container_width=True, config=CHART_CONFIG)

        comp1, comp2 = st.columns(2)
        with comp1:
            st.markdown(f"""
            <div class="detail-card" style="text-align: center;">
                <div class="metric-label">Claude</div>
                <div style="font-size: 0.82rem; color: {MUTED};">
                    {claude_ol.conversations} conversations &bull;
                    {claude_ol.total_messages:,} messages &bull;
                    OLI: {claude_ol.operator_load_index:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with comp2:
            st.markdown(f"""
            <div class="detail-card" style="text-align: center;">
                <div class="metric-label">ChatGPT</div>
                <div style="font-size: 0.82rem; color: {MUTED};">
                    {chatgpt_ol.conversations} conversations &bull;
                    {chatgpt_ol.total_messages:,} messages &bull;
                    OLI: {chatgpt_ol.operator_load_index:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Cross-model comparison requires batch results from both Claude and ChatGPT. Run batch_audit.py and batch_audit_chatgpt.py to generate data.")


# =========================================================================
# TAB 4: Detailed Flags
# =========================================================================
with tab_flags:

    all_flags = []
    for f in report.commission_flags:
        all_flags.append(("Commission", f))
    for f in report.omission_flags:
        all_flags.append(("Omission", f))
    for e in report.correction_events:
        if not e.held:
            all_flags.append(("Persistence", DriftFlag(
                layer="correction_persistence", turn=e.failure_turn or e.acknowledgment_turn,
                severity=7, description=f"Correction failed: {e.instruction[:80]}",
                instruction_ref=e.instruction, evidence=None,
                tag=DriftTag.CORRECTION_DECAY.value
            )))

    # Structural items
    for cp in report.conflict_pairs:
        all_flags.append(("Structural", DriftFlag(
            layer="structural", turn=cp.turn_a, severity=cp.severity,
            description=f"Conflict: T{cp.turn_a} vs T{cp.turn_b} on {cp.topic}",
            instruction_ref=None,
            evidence=f"A: {cp.statement_a[:80]}\nB: {cp.statement_b[:80]}",
            tag=DriftTag.CONFLICT_PAIR.value
        )))
    for v in report.void_events:
        all_flags.append(("Structural", DriftFlag(
            layer="structural", turn=v.turn, severity=v.severity,
            description=f"Void: {v.instruction_text[:80]} (break at: {v.void_at})",
            instruction_ref=v.instruction_text, evidence=None,
            tag=DriftTag.VOID_DETECTED.value
        )))

    if all_flags:
        # Filters
        st.markdown(f'<div class="section-header">Filters</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            layer_filter = st.multiselect("Layer", ["Commission", "Omission", "Persistence", "Structural"],
                                           default=["Commission", "Omission", "Persistence", "Structural"])
        with fc2:
            min_sev, max_sev = st.slider("Severity Range", 1, 10, (1, 10))
        with fc3:
            max_turn = max(f.turn for _, f in all_flags) if all_flags else report.total_turns
            turn_range = st.slider("Turn Range", 0, max_turn, (0, max_turn))

        # Apply filters
        filtered = [
            (cat, f) for cat, f in all_flags
            if cat in layer_filter
            and min_sev <= f.severity <= max_sev
            and turn_range[0] <= f.turn <= turn_range[1]
        ]

        st.markdown(f'<div class="section-header">Flags ({len(filtered)} of {len(all_flags)})</div>',
                    unsafe_allow_html=True)

        # Render filtered flags
        for cat, f in sorted(filtered, key=lambda x: x[1].severity, reverse=True):
            cat_badge = {"Commission": "badge-red", "Omission": "badge-amber",
                         "Persistence": "badge-purple", "Structural": "badge-blue"}.get(cat, "badge-slate")
            tag_text = f.tag or ""

            evidence_html = ""
            if f.evidence:
                ev = str(f.evidence)[:200]
                evidence_html = f'<div style="background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 4px; margin-top: 6px; font-family: \'JetBrains Mono\', monospace; font-size: 0.72rem; color: {MUTED};">{ev}</div>'

            instr_html = ""
            if f.instruction_ref:
                instr_html = f'<div style="font-size: 0.75rem; color: {SLATE}; margin-top: 4px;">Instruction: {f.instruction_ref[:120]}</div>'

            sc = score_color(f.severity)
            st.markdown(f"""
            <div class="detail-card detail-card-left" style="border-left-color: {sc}">
                <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-bottom: 4px;">
                    <span class="badge {cat_badge}">{cat.upper()}</span>
                    {severity_badge(f.severity)}
                    <span class="badge badge-slate">{tag_text}</span>
                    <span style="color: {MUTED}; font-size: 0.75rem;">Turn {f.turn}</span>
                </div>
                <div style="color: {LIGHT}; font-size: 0.85rem;">{f.description}</div>
                {instr_html}
                {evidence_html}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No drift flags detected across the conversation.")


# =========================================================================
# TAB 5: Export
# =========================================================================
with tab_export:

    st.markdown(f'<div class="section-header">Export Audit Report</div>', unsafe_allow_html=True)

    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            label="Download JSON Report",
            data=report_to_json(report),
            file_name=f"drift_audit_{report.conversation_id}.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp2:
        st.download_button(
            label="Download Text Report",
            data=format_report(report),
            file_name=f"drift_audit_{report.conversation_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # Summary text
    st.markdown(f'<div class="section-header">Quick Summary (copy for email/Slack)</div>',
                unsafe_allow_html=True)

    summary_text = f"""Drift Audit Summary — {report.conversation_id}
Overall Drift Score: {scores.get('overall_drift_score', 1)}/10 ({score_label(scores.get('overall_drift_score', 1))})
Total Turns: {report.total_turns} | Instructions: {report.instructions_extracted}
Commission Flags: {scores.get('commission_flag_count', 0)} | Omission Flags: {scores.get('omission_flag_count', 0)}
Corrections: {scores.get('corrections_failed', 0)}/{scores.get('correction_events_total', 0)} failed
Operator Load Index: {ol_metrics.operator_load_index:.3f} | Alignment Tax: {ol_metrics.alignment_tax:.1%}
Instruction Survival Rate: {ol_metrics.instruction_survival_rate:.0%}
Barometer: {scores.get('barometer_red_count', 0)} RED / {scores.get('barometer_total_signals', 0)} total signals
Generated by Drift Auditor v{VERSION}"""

    st.code(summary_text, language=None)
