"""
Drift Auditor — Live Analysis Mode
====================================
Paste-as-you-go mode. Re-analyzes on every update and tracks OLI over time.
"""
import streamlit as st
import plotly.graph_objects as go

from drift_auditor import audit_conversation, AuditReport
from operator_load import compute_operator_load
from ui.theme import THEMES, DEFAULT_THEME, score_color, score_label
from ui.components import render_metric_card, _audit_progress_html
from ui.charts import build_timeline_fig


def render_live_analysis_mode(config: dict) -> None:
    """Render the Live Analysis mode UI."""
    T = THEMES[DEFAULT_THEME]
    PLOTLY_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["chart_text"], family="JetBrains Mono, DM Sans, sans-serif", size=12),
        margin=dict(l=50, r=30, t=40, b=40),
    )

    st.markdown("## Live analysis")
    st.caption("Paste a growing conversation — re-analyzes on every update and tracks OLI over time.")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Live Config**")
        live_system_prompt = st.text_area(
            "System Prompt (Live)",
            height=80,
            placeholder="Optional: paste system prompt here…",
            key="live_sys_prompt",
        )
        live_preferences = st.text_area(
            "User Preferences (Live)",
            height=60,
            placeholder="Optional…",
            key="live_prefs",
        )

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
        analyze_clicked = st.button("🔍 Analyze Now", use_container_width=True, type="primary")
    with clear_col:
        if st.button("🗑️ Reset History", use_container_width=True):
            st.session_state.live_oli_history = []
            st.session_state.live_prev_text = ""
            st.rerun()

    if not (analyze_clicked and live_text.strip()):
        return

    _live_race = st.empty()
    _live_race.markdown(_audit_progress_html(T), unsafe_allow_html=True)
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

        live_scores = live_report.summary_scores
        overall = live_scores.get("overall_drift_score", 1)

        lm1, lm2, lm3, lm4, lm5 = st.columns(5)
        with lm1:
            render_metric_card("Overall Drift", overall, score_label(overall), T, hero=True)
        with lm2:
            oli_val = live_ol.operator_load_index
            oli_color = T["red"] if oli_val > 0.5 else T["accent"] if oli_val > 0.2 else T["green"]
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Operator Load Index</div>
                <div class="metric-value" style="color: {oli_color}">{oli_val:.3f}</div>
                <div class="metric-sub">human effort per turn</div>
            </div>""", unsafe_allow_html=True)
        with lm3:
            at = live_ol.alignment_tax
            at_color = T["red"] if at > 0.3 else T["accent"] if at > 0.15 else T["green"]
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Alignment Tax</div>
                <div class="metric-value" style="color: {at_color}">{at:.1%}</div>
                <div class="metric-sub">correction overhead</div>
            </div>""", unsafe_allow_html=True)
        with lm4:
            isr = live_ol.instruction_survival_rate
            isr_color = T["green"] if isr > 0.8 else T["accent"] if isr > 0.5 else T["red"]
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Instruction Survival</div>
                <div class="metric-value" style="color: {isr_color}">{isr:.0%}</div>
                <div class="metric-sub">instructions maintained</div>
            </div>""", unsafe_allow_html=True)
        with lm5:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">Turns Parsed</div>
                <div class="metric-value" style="color: {T["text"]}">{live_report.total_turns}</div>
                <div class="metric-sub">{live_report.instructions_extracted} instructions</div>
            </div>""", unsafe_allow_html=True)

        # OLI Trend Chart
        oli_hist = st.session_state.live_oli_history
        if len(oli_hist) >= 2:
            st.markdown("### Operator Load trend")
            st.caption("Operator Load Index over successive pastes — watch it climb as drift accumulates.")
            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(
                x=list(range(1, len(oli_hist) + 1)),
                y=[h["oli"] for h in oli_hist],
                mode="lines+markers", name="OLI",
                line=dict(color=T["accent"], width=3),
                marker=dict(size=10, color=T["accent"]),
                hovertext=[f"Snapshot {i+1}: OLI={h['oli']:.3f}, {h['turns']} turns"
                           for i, h in enumerate(oli_hist)],
                hoverinfo="text",
            ))
            trend_fig.add_trace(go.Scatter(
                x=list(range(1, len(oli_hist) + 1)),
                y=[h["drift_score"] / 10 for h in oli_hist],
                mode="lines+markers", name="Drift Score (÷10)",
                line=dict(color=T["red"], width=2, dash="dash"),
                marker=dict(size=7, color=T["red"]),
            ))
            trend_fig.add_trace(go.Scatter(
                x=list(range(1, len(oli_hist) + 1)),
                y=[h["alignment_tax"] for h in oli_hist],
                mode="lines+markers", name="Alignment Tax",
                line=dict(color=T["deep_red"], width=2, dash="dot"),
                marker=dict(size=7, color=T["deep_red"]),
            ))
            trend_fig.update_layout(
                **PLOTLY_LAYOUT, height=350,
                xaxis=dict(title="Paste Snapshot #", gridcolor="#2a2623", zeroline=False, dtick=1),
                yaxis=dict(title="Score", range=[0, 1.1], gridcolor="#2a2623", zeroline=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(trend_fig, use_container_width=True, config={"displaylogo": False})
        elif len(oli_hist) == 1:
            st.info("Paste more text and re-analyze to see the OLI trend chart build up over time.")

        # Drift Timeline
        if live_report.commission_flags or live_report.omission_flags:
            st.markdown("### Drift timeline")
            st.plotly_chart(build_timeline_fig(live_report, T), use_container_width=True,
                            config={"displaylogo": False})

        # Flag summary
        total_flags = (live_scores.get("commission_flag_count", 0) +
                       live_scores.get("omission_flag_count", 0))
        if total_flags > 0:
            st.markdown("### Flags detected")
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("Commission", live_scores.get("commission_flag_count", 0))
            fc2.metric("Omission", live_scores.get("omission_flag_count", 0))
            fc3.metric("Corrections Failed", live_scores.get("corrections_failed", 0))
            fc4.metric("Barometer RED", live_scores.get("barometer_red_count", 0))

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
