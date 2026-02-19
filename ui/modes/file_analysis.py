"""
Drift Auditor — File Analysis Mode
=====================================
Upload a conversation file and run a full drift audit.
This is the primary mode — 7 detail tabs, leaderboard, export.
"""
import os
import json
import glob as glob_mod
from collections import defaultdict

import streamlit as st
import plotly.graph_objects as go

from drift_auditor import (
    audit_conversation, AuditReport, DriftFlag,
    format_report, report_to_json,
)
from operator_load import compute_operator_load
from detectors.frustration import compute_frustration_index
from parsers.chat_parser import parse_chat_log as _parse_for_frustration
from ui.theme import THEMES, score_color, score_label
from ui.components import render_metric_card, _racing_stickmen_html, chart_export_png
from ui.charts import (
    build_timeline_fig, build_barometer_strip, build_barometer_detail,
    build_persistence_fig, build_commission_fig, build_omission_fig,
    build_cumulative_drift_fig, build_tag_breakdown_fig,
    build_frustration_gauge, build_frustration_line_fig,
)
from ui.spark import generate_spark_ideas


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


def render_file_analysis_mode(config: dict) -> None:
    """Render the File Analysis mode UI."""
    T = THEMES[st.session_state.get("theme_name", "Ember")]
    PLOTLY_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["chart_text"], family="JetBrains Mono, DM Sans, sans-serif", size=12),
        margin=dict(l=50, r=30, t=40, b=40),
    )

    uploaded = config.get("uploaded_file")
    system_prompt = config.get("system_prompt", "")
    preferences = config.get("preferences", "")
    window_size = config.get("window_size", 50)
    overlap = config.get("overlap", 10)

    raw_bytes: bytes | None = None
    conv_id = "unknown"

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        conv_id = uploaded.name
    elif st.session_state.get("sample_bytes") is not None:
        raw_bytes = st.session_state.sample_bytes
        conv_id = "sample_conversation.txt"

    if raw_bytes is None:
        st.markdown("## 🧪 Drift Auditor")
        st.markdown(
            "Upload a Claude conversation export or **load the sample** from the sidebar "
            "to analyze multi-turn drift patterns."
        )
        st.info("Supports Claude.ai JSON exports and plain text transcripts with role markers.")
        return

    # Run audit
    race_placeholder = st.empty()
    race_placeholder.markdown(_racing_stickmen_html(T), unsafe_allow_html=True)
    try:
        report: AuditReport = run_audit(raw_bytes, system_prompt, preferences, window_size, overlap, conv_id)
    except Exception as exc:
        race_placeholder.empty()
        st.error(f"Unable to audit this conversation: {exc}")
        return
    race_placeholder.empty()

    if report.total_turns == 0:
        st.error("Could not parse any turns from the uploaded file. "
                 "Supported formats: Claude.ai JSON export, plain text with Human:/Assistant: markers.")
        return

    scores = report.summary_scores

    # Dashboard — Summary metrics
    st.markdown("## 🧪 Drift Audit Results")
    overall = scores.get("overall_drift_score", 1)
    hero_col, c1, c2, c3, c4 = st.columns(5)
    with hero_col:
        render_metric_card("Overall Drift", overall, score_label(overall), T)
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
            render_metric_card(label, scores.get(key, 1), subtitle_fn(scores), T)

    st.markdown("")
    struct_col, void_col, conflict_col, shadow_col, op_col = st.columns(5)
    with struct_col:
        render_metric_card("Structural", scores.get("structural_score", 1), "composite of new detectors", T)
    for col, label, key in [
        (void_col, "Void Events", "void_events_count"),
        (conflict_col, "Conflict Pairs", "conflict_pairs_count"),
        (shadow_col, "Shadow Patterns", "shadow_patterns_count"),
    ]:
        with col:
            st.markdown(f"""<div class="glass-card" style="text-align:center;">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{T['accent']}">{scores.get(key, 0)}</div>
            </div>""", unsafe_allow_html=True)
    with op_col:
        effective = scores.get("op_moves_effective", 0)
        total_moves = scores.get("op_moves_total", 0)
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="metric-label">Operator Moves</div>
            <div class="metric-value" style="color:{T['accent']}">{effective}/{total_moves}</div>
            <div class="metric-sub">effective</div>
        </div>""", unsafe_allow_html=True)

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

    # Frustration Index
    raw_text_str = raw_bytes.decode("utf-8", errors="ignore")
    _transcript = _parse_for_frustration(raw_text_str)
    frustration = compute_frustration_index(_transcript)

    if frustration.per_turn:
        st.markdown("---")
        st.subheader("🧪 Frustration Index")
        st.caption("Experimental proxy — operator message sentiment (keyword + VADER).")
        frust_gauge_col, frust_chart_col = st.columns([1, 3])
        with frust_gauge_col:
            st.markdown(build_frustration_gauge(frustration, T), unsafe_allow_html=True)
        with frust_chart_col:
            frust_fig = build_frustration_line_fig(frustration, T)
            if frust_fig is not None:
                st.plotly_chart(frust_fig, use_container_width=True, config={"displaylogo": False})
                chart_export_png(frust_fig, f"frustration_{report.conversation_id}.png", "Download Frustration PNG")

    # Spark Ideas
    sparks = generate_spark_ideas(report, frustration if frustration.per_turn else None)
    if sparks:
        st.markdown("---")
        st.subheader("✨ Spark Ideas")
        st.caption("Notable moments and patterns from this conversation.")
        for spark in sparks:
            st.markdown(f"- {spark}")

    # Drift Timeline
    st.subheader("Drift Timeline")
    timeline_fig = build_timeline_fig(report, T)
    if report.commission_flags or report.omission_flags or report.correction_events:
        st.plotly_chart(timeline_fig, use_container_width=True, config={"displaylogo": False})
        chart_export_png(timeline_fig, f"drift_timeline_{report.conversation_id}.png", "Download Timeline PNG")
    else:
        st.success("No drift flags detected across the conversation.")

    if report.barometer_signals:
        strip_fig = build_barometer_strip(report, T)
        st.plotly_chart(strip_fig, use_container_width=True, config={"displayModeBar": False})

    cum_fig = build_cumulative_drift_fig(report, T)
    if cum_fig is not None:
        st.subheader("Drift Accumulation")
        st.plotly_chart(cum_fig, use_container_width=True, config={"displaylogo": False})
        chart_export_png(cum_fig, f"drift_accumulation_{report.conversation_id}.png", "Download Accumulation PNG")

    tag_fig = build_tag_breakdown_fig(report, T)
    if tag_fig is not None:
        st.subheader("Tag Breakdown")
        st.plotly_chart(tag_fig, use_container_width=True, config={"displaylogo": False})
        chart_export_png(tag_fig, f"tag_breakdown_{report.conversation_id}.png", "Download Tag Breakdown PNG")

    # Detail Tabs
    tab_lifecycle, tab_baro, tab_persist, tab_comm, tab_omit, tab_struct, tab_operator = st.tabs(
        ["📋 Lifecycle", "🧠 Barometer", "🔄 Persistence",
         "🚩 Commission", "📜 Omission", "🔍 Structural", "🎯 Operator"]
    )

    # Lifecycle Tab
    with tab_lifecycle:
        st.caption("Per-instruction tracking: when given, when followed, when dropped, coupling score.")
        if report.instruction_lifecycles:
            lc_fig = go.Figure()
            statuses = {"active": "#22c55e", "omitted": "#ef4444", "degraded": "#f59e0b", "superseded": "#666"}
            for i, lc in enumerate(report.instruction_lifecycles):
                y_pos = len(report.instruction_lifecycles) - i
                color = statuses.get(lc.status, "#888")
                label = lc.instruction_text[:40] + ("..." if len(lc.instruction_text) > 40 else "")
                lc_fig.add_trace(go.Scatter(
                    x=[lc.turn_given], y=[y_pos], mode="markers",
                    marker=dict(size=12, color="#22c55e", symbol="circle"),
                    showlegend=False, hoverinfo="text",
                    hovertext=f"Given at T{lc.turn_given}: {lc.instruction_text[:60]}",
                ))
                if lc.turn_last_followed is not None:
                    lc_fig.add_trace(go.Scatter(
                        x=[lc.turn_last_followed], y=[y_pos], mode="markers",
                        marker=dict(size=12, color="#22c55e", symbol="diamond"),
                        showlegend=False, hoverinfo="text",
                        hovertext=f"Last followed at T{lc.turn_last_followed}",
                    ))
                    lc_fig.add_trace(go.Scatter(
                        x=[lc.turn_given, lc.turn_last_followed], y=[y_pos, y_pos],
                        mode="lines", line=dict(color="#22c55e", width=3),
                        showlegend=False, hoverinfo="skip",
                    ))
                if lc.turn_first_omitted is not None:
                    lc_fig.add_trace(go.Scatter(
                        x=[lc.turn_first_omitted], y=[y_pos], mode="markers",
                        marker=dict(size=14, color="#ef4444", symbol="x"),
                        showlegend=False, hoverinfo="text",
                        hovertext=f"First omitted at T{lc.turn_first_omitted}",
                    ))
                    end_pt = lc.turn_last_followed if lc.turn_last_followed else lc.turn_given
                    lc_fig.add_trace(go.Scatter(
                        x=[end_pt, lc.turn_first_omitted], y=[y_pos, y_pos],
                        mode="lines", line=dict(color="#ef4444", width=2, dash="dash"),
                        showlegend=False, hoverinfo="skip",
                    ))
                lc_fig.add_annotation(x=-0.5, y=y_pos, text=label, showarrow=False,
                    xanchor="right", font=dict(size=10, color=color))
            lc_fig.update_layout(**PLOTLY_LAYOUT,
                height=max(300, len(report.instruction_lifecycles) * 55 + 80),
                xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
                yaxis=dict(visible=False), margin=dict(l=250, r=30, t=40, b=40),
            )
            st.plotly_chart(lc_fig, use_container_width=True, config={"displaylogo": False})

            st.markdown("**Instruction Details**")
            for lc in report.instruction_lifecycles:
                sc = {"active": "#22c55e", "omitted": "#ef4444",
                      "degraded": "#f59e0b", "superseded": "#666"}.get(lc.status, "#888")
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
                </div>""", unsafe_allow_html=True)

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
                                         "middle": "Middle (20-80%)", "edge_end": "End (last 20%)"}
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

    # Barometer Tab
    with tab_baro:
        st.caption("Structural epistemic posture analysis per assistant turn.")
        red_signals = [s for s in report.barometer_signals if s.classification == "RED"]
        yellow_signals = [s for s in report.barometer_signals if s.classification == "YELLOW"]
        green_signals = [s for s in report.barometer_signals if s.classification == "GREEN"]
        bc1, bc2, bc3 = st.columns(3)
        for col, label, signals, color in [
            (bc1, "RED", red_signals, "#ef4444"),
            (bc2, "YELLOW", yellow_signals, "#f59e0b"),
            (bc3, "GREEN", green_signals, "#22c55e"),
        ]:
            descs = {"RED": "Active structural drift", "YELLOW": "Passive drift / hedging", "GREEN": "Healthy posture"}
            with col:
                st.markdown(f"""
                <div class="glass-card" style="border-left: 3px solid {color}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color: {color}">{len(signals)}</div>
                    <div class="metric-sub">{descs[label]}</div>
                </div>""", unsafe_allow_html=True)
        if report.barometer_signals:
            st.plotly_chart(build_barometer_detail(report, T), use_container_width=True,
                            config={"displaylogo": False})
        if red_signals:
            st.markdown("**RED Signal Details**")
            for s in sorted(red_signals, key=lambda x: x.turn):
                with st.expander(f"Turn {s.turn} — Severity {s.severity}/10: {s.description[:60]}"):
                    st.markdown(f"**Classification**: :red[RED] — Active structural drift")
                    st.markdown(f"**Description**: {s.description}")
                    if s.evidence:
                        st.code(s.evidence, language=None)
        elif report.barometer_signals:
            st.success("No RED structural drift signals detected.")

    # Persistence Tab
    with tab_persist:
        st.caption("Tracks whether user corrections actually hold across subsequent turns.")
        if report.correction_events:
            st.plotly_chart(build_persistence_fig(report, T), use_container_width=True,
                            config={"displaylogo": False})
            st.markdown("**Event Details**")
            for event in report.correction_events:
                held_html = '<span class="status-held">HELD ✓</span>' if event.held else \
                            f'<span class="status-failed">FAILED at turn {event.failure_turn}</span>'
                st.markdown(f"""
                <div class="event-card" style="border-left: 3px solid {'#22c55e' if event.held else '#ef4444'}">
                    {held_html}
                    <br><span style="color: rgba(255,255,255,0.6)">
                        Turn {event.correction_turn} → Ack {event.acknowledgment_turn}
                    </span>
                    <br><small style="color: rgba(255,255,255,0.4)">{event.instruction[:150]}</small>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No correction events detected in this conversation.")

    # Commission Tab
    with tab_comm:
        st.caption("Sycophancy, reality distortion, and unwarranted confidence markers.")
        if report.commission_flags:
            st.plotly_chart(build_commission_fig(report, T), use_container_width=True,
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

    # Omission Tab
    with tab_omit:
        st.caption("Instruction violations: required behaviors absent or prohibitions broken.")
        if report.omission_flags:
            grouped: dict[str, list] = defaultdict(list)
            for f in report.omission_flags:
                grouped[f.instruction_ref or "General"].append(f)
            for instruction, flags in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
                max_sev = max(f.severity for f in flags)
                with st.expander(f"{instruction[:65]}... ({len(flags)} violations, max sev: {max_sev})"):
                    st.markdown(f"**Instruction**: {instruction}")
                    st.markdown(f"**Violations**: {len(flags)} | **Max Severity**: {max_sev}")
                    for f in sorted(flags, key=lambda x: x.turn):
                        st.markdown(f"- Turn {f.turn} [sev {f.severity}]: {f.description[:100]}")
            st.markdown("**Severity Over Time**")
            st.plotly_chart(build_omission_fig(report, T), use_container_width=True,
                            config={"displaylogo": False})
        else:
            st.info("No omission drift detected (local heuristics only). Full semantic detection requires API mode.")

    # Structural Tab
    with tab_struct:
        st.caption("Conflict pairs, void events, shadow patterns, pre-drift signals, and false equivalence.")
        struct_sub1, struct_sub2 = st.columns(2)
        with struct_sub1:
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

    # Operator Tab
    with tab_operator:
        st.caption("12-Rule Operator System: classifying the human's corrective actions.")
        if report.op_moves:
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
            rule_fig.add_trace(go.Bar(y=rule_names, x=rule_totals, orientation="h",
                name="Total", marker_color="rgba(255,255,255,0.15)"))
            rule_fig.add_trace(go.Bar(y=rule_names, x=rule_eff, orientation="h",
                name="Effective", marker_color="#22c55e"))
            rule_fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay",
                height=max(200, len(rule_names) * 40 + 60),
                xaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(rule_fig, use_container_width=True, config={"displaylogo": False})

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
                st.markdown(f"""
                <div class="event-card" style="border-left: 3px solid {eff_color}">
                    <span style="color: {eff_color}; font-weight: 700">{m.rule}</span>
                    <span style="color: rgba(255,255,255,0.4)"> | {RULE_DESCRIPTIONS.get(m.rule, '')}</span>
                    <br><span style="color: rgba(255,255,255,0.5)">Turn {m.turn} | {m.effectiveness}</span>
                    <br><small style="color: rgba(255,255,255,0.6)">{m.target_behavior[:120]}</small>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No operator steering moves detected in this conversation.")

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

    # Cross-Model Leaderboard
    st.divider()
    st.subheader("Cross-Model Drift Leaderboard")
    leaderboard_data = []
    repo_root = os.path.dirname(os.path.abspath(__file__))
    for _ in range(2):
        repo_root = os.path.dirname(repo_root)
    for batch_dir in ["batch_results", "batch_results_chatgpt"]:
        batch_path = os.path.join(repo_root, batch_dir)
        json_files = glob_mod.glob(os.path.join(batch_path, "batch_results*.json"))
        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    results = json.load(f)
                if results:
                    leaderboard_data.extend(results)
            except Exception:
                pass

    if leaderboard_data:
        model_stats: dict = {}
        for r in leaderboard_data:
            model = r.get("model", "claude" if "uuid" in r else "unknown")
            if model not in model_stats:
                model_stats[model] = {"conversations": 0, "total_score": 0, "total_msgs": 0,
                    "total_flags": 0, "corrections": 0, "corrections_failed": 0,
                    "voids": 0, "op_moves": 0}
            s = model_stats[model]
            s["conversations"] += 1
            s["total_score"] += r.get("overall_score", 0)
            s["total_msgs"] += r.get("message_count", 0)
            s["total_flags"] += r.get("commission_flags", 0) + r.get("omission_flags", 0)
            s["corrections"] += r.get("corrections_total", 0)
            s["corrections_failed"] += r.get("corrections_failed", 0)
            s["voids"] += r.get("void_events", 0)
            s["op_moves"] += r.get("op_moves", 0)

        lb_rows = []
        for model, s in sorted(model_stats.items(), key=lambda x: x[1]["total_score"] / max(x[1]["conversations"], 1)):
            avg_drift = s["total_score"] / max(s["conversations"], 1)
            corr_fail_rate = s["corrections_failed"] / max(s["corrections"], 1) * 100
            void_rate = s["voids"] / max(s["total_msgs"], 1)
            flags_per_msg = s["total_flags"] / max(s["total_msgs"], 1)
            lb_rows.append({
                "Model": model, "Conversations": s["conversations"], "Messages": s["total_msgs"],
                "Avg Drift Score": round(avg_drift, 1), "Flags/Message": round(flags_per_msg, 3),
                "Correction Fail %": round(corr_fail_rate, 1), "Void Rate": round(void_rate, 4),
                "Operator Moves": s["op_moves"],
            })

        import pandas as pd
        lb_df = pd.DataFrame(lb_rows)
        st.dataframe(lb_df, use_container_width=True, hide_index=True)

        lb_chart_scores = [r["Avg Drift Score"] for r in lb_rows]
        lb_fig = go.Figure(go.Bar(
            x=[r["Model"] for r in lb_rows], y=lb_chart_scores,
            marker_color=[score_color(int(s), T) for s in lb_chart_scores],
            text=[f"{s:.1f}" for s in lb_chart_scores], textposition="outside",
            textfont=dict(color="#e8dfd0"),
        ))
        lb_fig.update_layout(**PLOTLY_LAYOUT, height=350,
            xaxis=dict(title="Model", gridcolor="#2a2623"),
            yaxis=dict(title="Avg Drift Score", range=[0, 10], gridcolor="#2a2623"),
        )
        st.plotly_chart(lb_fig, use_container_width=True, config={"displaylogo": False})
        total_convs = sum(r["Conversations"] for r in lb_rows)
        total_msgs = sum(r["Messages"] for r in lb_rows)
        st.caption(f"Leaderboard based on {total_convs} conversations, {total_msgs:,} messages across {len(lb_rows)} models.")
    else:
        st.info("No batch results found. Run batch_audit.py or batch_audit_chatgpt.py to generate cross-model data.")

    # Export
    st.divider()
    st.subheader("Export Report")
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            label="📥 Download JSON Report",
            data=report_to_json(report),
            file_name=f"drift_audit_{report.conversation_id}.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp2:
        st.download_button(
            label="📄 Download Text Report",
            data=format_report(report),
            file_name=f"drift_audit_{report.conversation_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )
