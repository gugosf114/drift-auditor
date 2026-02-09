"""
Drift Auditor — Visualization Module
=====================================
Professional chart builders for the Streamlit dashboard.

All chart functions return Plotly Figure objects.
Layout code stays in app.py; chart generation lives here.

Author: George Abrahamyants + Claude (Opus 4.6)
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from drift_auditor import AuditReport

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Brand palette
RED = "#EF4444"
AMBER = "#F59E0B"
GREEN = "#10B981"
PURPLE = "#7C3AED"
BLUE = "#3B82F6"
SLATE = "#64748B"
LIGHT = "#E2E8F0"
MUTED = "#94A3B8"
BG = "#0F0F12"
CARD_BG = "#1A1A24"
BORDER = "#2D2D3D"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=LIGHT, family="Inter, system-ui, sans-serif", size=12),
    margin=dict(l=50, r=30, t=40, b=40),
)

CHART_CONFIG = {"displaylogo": False, "displayModeBar": False}


def score_color(score: int) -> str:
    """Map 1-10 severity to brand color."""
    if score <= 3:
        return GREEN
    elif score <= 6:
        return AMBER
    else:
        return RED


def score_label(score: int) -> str:
    """Human label for 1-10 score."""
    if score <= 2: return "Clean"
    elif score <= 4: return "Low Drift"
    elif score <= 6: return "Moderate"
    elif score <= 8: return "Elevated"
    else: return "Severe"


def coupling_label(score: float) -> str:
    if score >= 0.6: return "HIGH"
    elif score >= 0.3: return "MEDIUM"
    else: return "LOW"

def coupling_color(score: float) -> str:
    if score >= 0.6: return RED
    elif score >= 0.3: return AMBER
    else: return GREEN


# ---------------------------------------------------------------------------
# Dashboard Overview Charts
# ---------------------------------------------------------------------------

def build_barometer_timeline(report: AuditReport) -> go.Figure:
    """
    Signature visualization: RED/YELLOW/GREEN barometer over conversation turns.
    Full-width, prominent, the chart that defines Drift Auditor.
    """
    if not report.barometer_signals:
        return go.Figure()

    colors_map = {"GREEN": GREEN, "YELLOW": AMBER, "RED": RED}

    fig = go.Figure()

    # Background shading per signal
    for s in report.barometer_signals:
        fig.add_trace(go.Bar(
            x=[s.turn], y=[s.severity],
            marker_color=colors_map.get(s.classification, AMBER),
            marker_opacity=0.7,
            hovertext=f"Turn {s.turn}: {s.classification}<br>Severity: {s.severity}/10<br>{s.description[:80]}",
            hoverinfo="text",
            showlegend=False,
        ))

    # Add threshold lines
    fig.add_hline(y=7, line_dash="dot", line_color="rgba(239,68,68,0.3)",
                  annotation_text="Critical", annotation_position="top right",
                  annotation_font_color=MUTED, annotation_font_size=10)
    fig.add_hline(y=4, line_dash="dot", line_color="rgba(245,158,11,0.3)",
                  annotation_text="Warning", annotation_position="top right",
                  annotation_font_color=MUTED, annotation_font_size=10)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        xaxis=dict(title="Conversation Turn", gridcolor=BORDER, zeroline=False,
                   tickfont=dict(size=10)),
        yaxis=dict(title="Severity", range=[0, 11], gridcolor=BORDER, zeroline=False,
                   tickfont=dict(size=10)),
        bargap=0.15,
    )
    return fig


def build_drift_timeline(report: AuditReport) -> go.Figure:
    """Scatter plot of all drift flags overlaid on conversation."""
    fig = go.Figure()

    if report.commission_flags:
        fig.add_trace(go.Scatter(
            x=[f.turn for f in report.commission_flags],
            y=[f.severity for f in report.commission_flags],
            mode="markers", name="Commission",
            marker=dict(size=10, color=RED, symbol="diamond",
                        line=dict(width=1, color="rgba(239,68,68,0.4)")),
            hovertext=[f"T{f.turn}: {f.description[:60]}" for f in report.commission_flags],
            hoverinfo="text",
        ))

    if report.omission_flags:
        fig.add_trace(go.Scatter(
            x=[f.turn for f in report.omission_flags],
            y=[f.severity for f in report.omission_flags],
            mode="markers", name="Omission",
            marker=dict(size=10, color=AMBER, symbol="triangle-up",
                        line=dict(width=1, color="rgba(245,158,11,0.4)")),
            hovertext=[f"T{f.turn}: {f.description[:60]}" for f in report.omission_flags],
            hoverinfo="text",
        ))

    failed = [e for e in report.correction_events if not e.held and e.failure_turn is not None]
    if failed:
        fig.add_trace(go.Scatter(
            x=[e.failure_turn for e in failed],
            y=[8] * len(failed),
            mode="markers", name="Correction Failed",
            marker=dict(size=12, color=PURPLE, symbol="x", line=dict(width=2, color=PURPLE)),
            hovertext=[f"Correction failed at T{e.failure_turn}" for e in failed],
            hoverinfo="text",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        xaxis=dict(title="Turn", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Severity", range=[0, 11], gridcolor=BORDER, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="closest",
    )
    return fig


def build_flag_summary_chart(report: AuditReport) -> go.Figure:
    """Horizontal bar chart summarizing flag counts by category."""
    categories = ["Commission", "Omission", "Persistence", "Structural"]
    counts = [
        len(report.commission_flags),
        len(report.omission_flags),
        sum(1 for e in report.correction_events if not e.held),
        len(report.conflict_pairs) + len(report.void_events) + len(report.shadow_patterns),
    ]
    colors = [RED, AMBER, PURPLE, BLUE]

    fig = go.Figure(go.Bar(
        y=categories, x=counts, orientation="h",
        marker_color=colors,
        text=counts, textposition="outside",
        textfont=dict(color=LIGHT, size=12),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=200,
        xaxis=dict(gridcolor=BORDER, zeroline=False, title="Count"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=100, r=50, t=10, b=30),
    )
    return fig


# ---------------------------------------------------------------------------
# Instruction Lifecycle Charts
# ---------------------------------------------------------------------------

def build_lifecycle_timeline(report: AuditReport) -> go.Figure:
    """Gantt-style chart showing instruction lifecycle."""
    if not report.instruction_lifecycles:
        return go.Figure()

    status_colors = {"active": GREEN, "omitted": RED, "degraded": AMBER, "superseded": SLATE}
    fig = go.Figure()

    for i, lc in enumerate(report.instruction_lifecycles):
        y_pos = len(report.instruction_lifecycles) - i
        color = status_colors.get(lc.status, SLATE)
        label = lc.instruction_text[:45] + ("..." if len(lc.instruction_text) > 45 else "")

        # Given point
        fig.add_trace(go.Scatter(
            x=[lc.turn_given], y=[y_pos], mode="markers",
            marker=dict(size=10, color=GREEN, symbol="circle"),
            showlegend=False, hoverinfo="text",
            hovertext=f"Given at T{lc.turn_given}: {lc.instruction_text[:80]}",
        ))

        # Active span
        if lc.turn_last_followed is not None:
            fig.add_trace(go.Scatter(
                x=[lc.turn_given, lc.turn_last_followed], y=[y_pos, y_pos],
                mode="lines", line=dict(color=GREEN, width=4),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=[lc.turn_last_followed], y=[y_pos], mode="markers",
                marker=dict(size=10, color=GREEN, symbol="diamond"),
                showlegend=False, hoverinfo="text",
                hovertext=f"Last followed at T{lc.turn_last_followed}",
            ))

        # Omission point
        if lc.turn_first_omitted is not None:
            end_pt = lc.turn_last_followed if lc.turn_last_followed else lc.turn_given
            fig.add_trace(go.Scatter(
                x=[end_pt, lc.turn_first_omitted], y=[y_pos, y_pos],
                mode="lines", line=dict(color=RED, width=3, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=[lc.turn_first_omitted], y=[y_pos], mode="markers",
                marker=dict(size=12, color=RED, symbol="x"),
                showlegend=False, hoverinfo="text",
                hovertext=f"First omitted at T{lc.turn_first_omitted}",
            ))

        # Label
        fig.add_annotation(
            x=-0.5, y=y_pos, text=label, showarrow=False,
            xanchor="right", font=dict(size=10, color=color),
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(300, len(report.instruction_lifecycles) * 50 + 80),
        xaxis=dict(title="Turn", gridcolor=BORDER, zeroline=False),
        yaxis=dict(visible=False),
        margin=dict(l=280, r=30, t=20, b=40),
    )
    return fig


def build_decay_curves(report: AuditReport) -> go.Figure:
    """
    Instruction Decay Curves: compliance % over conversation turns.
    Derives compliance from lifecycle data — if an instruction was followed
    from turn_given to turn_last_followed, compliance is 100% in that range,
    then drops to 0 at turn_first_omitted.
    """
    if not report.instruction_lifecycles:
        return go.Figure()

    fig = go.Figure()
    status_colors = {"active": GREEN, "omitted": RED, "degraded": AMBER, "superseded": SLATE}
    max_turn = report.total_turns

    for lc in report.instruction_lifecycles:
        if lc.turn_given >= max_turn:
            continue

        color = status_colors.get(lc.status, SLATE)
        turns = []
        compliance = []

        start = lc.turn_given
        followed_until = lc.turn_last_followed if lc.turn_last_followed else max_turn
        omitted_at = lc.turn_first_omitted

        # Build the curve
        turns.append(start)
        compliance.append(100)

        if omitted_at is not None:
            # Was followed, then dropped
            if followed_until > start:
                turns.append(followed_until)
                compliance.append(100)
            turns.append(omitted_at)
            compliance.append(0)
            turns.append(max_turn)
            compliance.append(0)
        else:
            # Still active
            turns.append(max_turn)
            compliance.append(100)

        label = lc.instruction_text[:50] + ("..." if len(lc.instruction_text) > 50 else "")
        fig.add_trace(go.Scatter(
            x=turns, y=compliance, mode="lines",
            name=label, line=dict(color=color, width=2),
            hovertext=f"{lc.instruction_text[:60]} [{lc.status}]",
            hoverinfo="text",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        xaxis=dict(title="Turn", gridcolor=BORDER, zeroline=False),
        yaxis=dict(title="Compliance %", range=[-5, 105], gridcolor=BORDER, zeroline=False),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(l=50, r=30, t=20, b=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Operator Load Charts
# ---------------------------------------------------------------------------

def build_operator_move_timeline(report: AuditReport) -> go.Figure:
    """Vertical timeline of operator moves with rule types."""
    if not report.op_moves:
        return go.Figure()

    eff_colors = {
        "effective": GREEN, "partially_effective": AMBER,
        "ineffective": RED, "unknown": SLATE
    }

    fig = go.Figure()
    for i, m in enumerate(report.op_moves):
        y_pos = len(report.op_moves) - i
        color = eff_colors.get(m.effectiveness, SLATE)
        fig.add_trace(go.Scatter(
            x=[m.turn], y=[y_pos], mode="markers+text",
            marker=dict(size=12, color=color, symbol="circle"),
            text=[m.rule], textposition="middle right",
            textfont=dict(size=10, color=MUTED),
            showlegend=False, hoverinfo="text",
            hovertext=f"T{m.turn} | {m.rule} | {m.effectiveness}<br>{m.target_behavior[:80]}",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(200, len(report.op_moves) * 40 + 60),
        xaxis=dict(title="Turn", gridcolor=BORDER, zeroline=False),
        yaxis=dict(visible=False),
        margin=dict(l=30, r=200, t=20, b=40),
    )
    return fig


def build_rule_frequency(report: AuditReport) -> go.Figure:
    """Bar chart of operator rule usage frequency."""
    if not report.op_moves:
        return go.Figure()

    rule_counts = defaultdict(int)
    rule_effective = defaultdict(int)
    for m in report.op_moves:
        rule_counts[m.rule] += 1
        if m.effectiveness == "effective":
            rule_effective[m.rule] += 1

    rules_sorted = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
    rule_names = [r for r, _ in rules_sorted]
    totals = [c for _, c in rules_sorted]
    effective = [rule_effective.get(r, 0) for r in rule_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=rule_names, x=totals, orientation="h",
        name="Total", marker_color="rgba(100,116,139,0.3)",
    ))
    fig.add_trace(go.Bar(
        y=rule_names, x=effective, orientation="h",
        name="Effective", marker_color=GREEN,
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="overlay",
        height=max(200, len(rule_names) * 35 + 60),
        xaxis=dict(title="Count", gridcolor=BORDER),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=120, r=30, t=40, b=30),
    )
    return fig


def build_operator_load_comparison(claude_stats: dict, chatgpt_stats: dict) -> go.Figure:
    """Side-by-side operator load comparison."""
    metrics = [
        "Operator Load Index", "Alignment Tax", "Self-Sufficiency",
        "Instruction Survival", "Correction Efficiency", "Avg Drift Score",
    ]
    keys = [
        "operator_load_index", "alignment_tax", "self_sufficiency_score",
        "instruction_survival_rate", "correction_efficiency", "avg_drift_score",
    ]
    claude_vals = [claude_stats.get(k, 0) for k in keys]
    chatgpt_vals = [chatgpt_stats.get(k, 0) for k in keys]

    fig = go.Figure(data=[
        go.Bar(name="Claude", x=metrics, y=claude_vals, marker_color=PURPLE),
        go.Bar(name="ChatGPT", x=metrics, y=chatgpt_vals, marker_color=GREEN),
    ])
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        height=400,
        yaxis_title="Score",
        legend=dict(x=0.85, y=1, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Persistence Chart
# ---------------------------------------------------------------------------

def build_persistence_fig(report: AuditReport) -> go.Figure:
    """Correction -> acknowledgment -> outcome flow."""
    events = report.correction_events
    if not events:
        return go.Figure()

    fig = go.Figure()
    for i, event in enumerate(events):
        y_pos = len(events) - i

        # Correction point
        fig.add_trace(go.Scatter(
            x=[event.correction_turn], y=[y_pos],
            mode="markers", marker=dict(size=12, color=AMBER, symbol="circle"),
            showlegend=False, hoverinfo="text",
            hovertext=f"User corrected at T{event.correction_turn}",
        ))

        # Acknowledgment
        fig.add_trace(go.Scatter(
            x=[event.acknowledgment_turn], y=[y_pos],
            mode="markers", marker=dict(size=12, color=GREEN, symbol="circle"),
            showlegend=False, hoverinfo="text",
            hovertext=f"Model acknowledged at T{event.acknowledgment_turn}",
        ))

        # Connector
        fig.add_trace(go.Scatter(
            x=[event.correction_turn, event.acknowledgment_turn], y=[y_pos, y_pos],
            mode="lines", line=dict(color="rgba(148,163,184,0.3)", width=2, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

        # Outcome
        if not event.held and event.failure_turn is not None:
            fig.add_trace(go.Scatter(
                x=[event.failure_turn], y=[y_pos],
                mode="markers", marker=dict(size=14, color=RED, symbol="x"),
                showlegend=False, hoverinfo="text",
                hovertext=f"Correction failed at T{event.failure_turn}",
            ))
            fig.add_trace(go.Scatter(
                x=[event.acknowledgment_turn, event.failure_turn], y=[y_pos, y_pos],
                mode="lines", line=dict(color=RED, width=2, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[event.acknowledgment_turn + 2], y=[y_pos],
                mode="markers+text", text=["Held"],
                textposition="middle right", textfont=dict(size=10, color=GREEN),
                marker=dict(size=10, color=GREEN, symbol="circle"),
                showlegend=False, hoverinfo="text",
                hovertext="Correction held",
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=max(180, len(events) * 60 + 60),
        xaxis=dict(title="Turn", gridcolor=BORDER, zeroline=False),
        yaxis=dict(visible=False),
    )
    return fig
