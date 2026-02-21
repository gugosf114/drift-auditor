"""
Drift Auditor — Chart Builders
================================
All Plotly figure builders and the frustration gauge HTML generator.
No Streamlit dependency — takes dataclasses, returns go.Figure or str.
Fully testable.
"""

from __future__ import annotations

from collections import defaultdict

import plotly.graph_objects as go

from ui.theme import get_plotly_layout, score_color


def build_timeline_fig(report, t: dict) -> go.Figure:
    """Scatter plot of all drift flags across conversation turns."""
    layout = get_plotly_layout(t)
    fig = go.Figure()

    if report.commission_flags:
        fig.add_trace(go.Scatter(
            x=[f.turn for f in report.commission_flags],
            y=[f.severity for f in report.commission_flags],
            mode="markers",
            name="Commission",
            marker=dict(size=11, color="#ef4444", symbol="diamond",
                        line=dict(width=1, color="rgba(239,68,68,0.5)")),
            hovertext=[f"T{f.turn}: {f.description[:60]}" for f in report.commission_flags],
            hoverinfo="text",
        ))

    if report.omission_flags:
        fig.add_trace(go.Scatter(
            x=[f.turn for f in report.omission_flags],
            y=[f.severity for f in report.omission_flags],
            mode="markers",
            name="Omission",
            marker=dict(size=11, color="#f59e0b", symbol="triangle-up",
                        line=dict(width=1, color="rgba(245,158,11,0.5)")),
            hovertext=[f"T{f.turn}: {f.description[:60]}" for f in report.omission_flags],
            hoverinfo="text",
        ))

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
        **layout,
        height=380,
        xaxis=dict(title="Turn", gridcolor="#2a2623", zeroline=False),
        yaxis=dict(title="Severity", range=[0, 11], gridcolor="#2a2623", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="closest",
    )
    return fig


def build_barometer_strip(report, t: dict) -> go.Figure:
    """Thin heatmap strip showing barometer classification per assistant turn."""
    layout = get_plotly_layout(t)
    if not report.barometer_signals:
        return go.Figure()

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

    fig.update_layout(**layout)
    fig.update_layout(
        barmode="stack",
        height=60,
        margin=dict(l=120, r=30, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def build_barometer_detail(report, t: dict) -> go.Figure:
    """Per-turn bar chart of barometer signals colored by classification."""
    layout = get_plotly_layout(t)
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
        **layout,
        barmode="stack",
        height=320,
        xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Severity", gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def build_persistence_fig(report, t: dict) -> go.Figure:
    """Correction -> acknowledgment -> failure/held flow visualization."""
    layout = get_plotly_layout(t)
    events = report.correction_events
    if not events:
        return go.Figure()

    fig = go.Figure()
    for i, event in enumerate(events):
        y_pos = len(events) - i
        held = event.held

        fig.add_trace(go.Scatter(
            x=[event.correction_turn], y=[y_pos],
            mode="markers+text", text=["Correction"],
            textposition="top center", textfont=dict(size=10, color="rgba(255,255,255,0.6)"),
            marker=dict(size=14, color="#d68910", symbol="circle"),
            showlegend=False, hoverinfo="text",
            hovertext=f"User corrected at T{event.correction_turn}",
        ))
        fig.add_trace(go.Scatter(
            x=[event.acknowledgment_turn], y=[y_pos],
            mode="markers+text", text=["Ack"],
            textposition="top center", textfont=dict(size=10, color="rgba(255,255,255,0.6)"),
            marker=dict(size=14, color="#22c55e", symbol="circle"),
            showlegend=False, hoverinfo="text",
            hovertext=f"Model acknowledged at T{event.acknowledgment_turn}",
        ))
        fig.add_trace(go.Scatter(
            x=[event.correction_turn, event.acknowledgment_turn],
            y=[y_pos, y_pos],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.2)", width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

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
            fig.add_trace(go.Scatter(
                x=[event.acknowledgment_turn + 2], y=[y_pos],
                mode="markers+text", text=["Held \u2713"],
                textposition="middle right", textfont=dict(size=11, color="#22c55e"),
                marker=dict(size=12, color="#22c55e", symbol="circle"),
                showlegend=False, hoverinfo="text",
                hovertext="Correction held across subsequent turns",
            ))

    fig.update_layout(
        **layout,
        height=max(180, len(events) * 80 + 60),
        xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(visible=False),
    )
    return fig


def build_commission_fig(report, t: dict) -> go.Figure:
    """Horizontal bar chart of commission flags sorted by severity."""
    layout = get_plotly_layout(t)
    flags = sorted(report.commission_flags, key=lambda f: f.severity, reverse=True)
    if not flags:
        return go.Figure()

    fig = go.Figure(go.Bar(
        y=[f"T{f.turn}: {f.description[:35]}" for f in flags],
        x=[f.severity for f in flags],
        orientation="h",
        marker_color=[score_color(f.severity, t) for f in flags],
        hovertext=[f"Turn {f.turn} | Sev {f.severity} | {f.description}" for f in flags],
        hoverinfo="text",
    ))
    fig.update_layout(
        **layout,
        height=max(200, len(flags) * 45 + 60),
        xaxis=dict(title="Severity", range=[0, 11], gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def build_omission_fig(report, t: dict) -> go.Figure:
    """Omission flag severity over time."""
    layout = get_plotly_layout(t)
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
        **layout,
        height=300,
        xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Severity", range=[0, 11], gridcolor="rgba(255,255,255,0.06)"),
    )
    return fig


def build_cumulative_drift_fig(report, t: dict) -> go.Figure | None:
    """Line chart: cumulative drift flag count per turn across the conversation."""
    layout = get_plotly_layout(t)
    all_flags = (
        [(f.turn, "Commission", f.severity) for f in report.commission_flags]
        + [(f.turn, "Omission", f.severity) for f in report.omission_flags]
    )
    if not all_flags:
        return None

    max_turn = report.total_turns or max(turn for turn, _, _ in all_flags) + 1
    cum_count = [0] * (max_turn + 1)
    cum_severity = [0.0] * (max_turn + 1)

    for turn, _, sev in all_flags:
        if 0 <= turn <= max_turn:
            cum_count[turn] += 1
            cum_severity[turn] += sev

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
        **layout,
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


def build_tag_breakdown_fig(report, t: dict) -> go.Figure | None:
    """Horizontal bar chart showing which drift tags dominate the conversation."""
    layout = get_plotly_layout(t)
    tag_counts: dict[str, int] = defaultdict(int)
    tag_severity: dict[str, float] = defaultdict(float)

    for f in report.commission_flags + report.omission_flags:
        tag = f.tag or "UNTAGGED"
        tag_counts[tag] += 1
        tag_severity[tag] += f.severity

    if not tag_counts:
        return None

    sorted_tags = sorted(tag_counts.keys(), key=lambda tag: tag_counts[tag], reverse=True)
    counts = [tag_counts[tag] for tag in sorted_tags]
    avg_sev = [round(tag_severity[tag] / tag_counts[tag], 1) for tag in sorted_tags]

    tag_colors_map = {
        "SYCOPHANCY": "#ef4444", "REALITY_DISTORT": "#dc2626",
        "CONF_INFLATE": "#f97316", "INSTR_DROP": "#f59e0b",
        "SEM_DILUTE": "#eab308", "CORR_DECAY": "#a855f7",
        "CONFLICT_PAIR": "#8b5cf6", "SHADOW_PATTERN": "#6366f1",
        "OP_MOVE": "#3b82f6", "VOID_DETECTED": "#64748b",
    }
    colors = [tag_colors_map.get(tag, "#94a3b8") for tag in sorted_tags]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_tags, x=counts,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertext=[f"{tag}: {c} flags, avg severity {s}"
                   for tag, c, s in zip(sorted_tags, counts, avg_sev)],
        hoverinfo="text",
    ))

    fig.update_layout(
        **layout,
        height=max(200, len(sorted_tags) * 40 + 60),
        xaxis=dict(title="Flag Count", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


def build_frustration_gauge(result, t: dict) -> str:
    """Return HTML/CSS for a vertical thermometer gauge."""
    avg = result.average
    pct = min(100, max(0, avg * 10))

    bar_color = score_color(avg, t)
    glow = bar_color

    trend_icon = {"rising": "\u2191", "falling": "\u2193", "spike": "\u26a1", "stable": "\u2014"}.get(
        result.trend, "\u2014"
    )
    trend_label = result.trend.capitalize()

    return f"""
    <div class="glass-card" style="text-align: center; padding: 24px 16px;">
        <div class="metric-label">Frustration Index</div>
        <div style="
            width: 36px; height: 160px; margin: 16px auto 8px;
            background: {t["surface"]}; border: 1px solid {t["border"]};
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
                    font-size: 0.75rem; color: {t["muted"]};">
            {trend_icon} {trend_label} &nbsp;&middot;&nbsp; Peak {result.peak:.1f} at T{result.peak_turn}
        </div>
        <div style="margin-top: 4px; font-family: 'JetBrains Mono', monospace;
                    font-size: 0.65rem; color: {t["muted"]};">
            via {result.backend}
        </div>
    </div>
    """


def build_frustration_line_fig(result, t: dict) -> go.Figure | None:
    """Line chart: frustration score per operator message over conversation turns."""
    layout = get_plotly_layout(t)
    if not result.per_turn:
        return None

    fig = go.Figure()

    fig.add_hrect(y0=0, y1=3, fillcolor=t["green"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=3, y1=5, fillcolor=t["accent"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=5, y1=7, fillcolor=t["red"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=7, y1=10, fillcolor=t["deep_red"], opacity=0.05, line_width=0)

    colors = [score_color(s, t) for s in result.per_turn]

    fig.add_trace(go.Scatter(
        x=result.turn_indices,
        y=result.per_turn,
        mode="lines+markers",
        name="Frustration",
        line=dict(color=t["accent"], width=2),
        marker=dict(size=7, color=colors, line=dict(width=1, color=t["border"])),
        hovertext=[
            f"Turn {turn}: {s:.1f}/10"
            for turn, s in zip(result.turn_indices, result.per_turn)
        ],
        hoverinfo="text",
        fill="tozeroy",
        fillcolor=(
            f"rgba({int(t['accent'][1:3],16)},"
            f"{int(t['accent'][3:5],16)},"
            f"{int(t['accent'][5:7],16)},0.06)"
        ),
    ))

    fig.add_hline(
        y=result.average, line_dash="dot",
        line_color=t["muted"], opacity=0.6,
        annotation_text=f"avg {result.average:.1f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color=t["muted"]),
    )

    fig.update_layout(
        **layout,
        height=280,
        xaxis=dict(title="Conversation Turn", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Frustration Score", range=[0, 10.5],
                   gridcolor="rgba(255,255,255,0.06)"),
        showlegend=False,
    )
    return fig
