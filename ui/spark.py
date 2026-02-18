"""
Drift Auditor — Spark Ideas
============================
Extract notable/interesting moments from an audit report for display.
Pure function — no Streamlit dependency. Fully testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sys import path as _path
    import os as _os


def generate_spark_ideas(report, frustration=None) -> list[str]:
    """Extract interesting/notable moments from the audit for display.

    Args:
        report: AuditReport instance
        frustration: Optional FrustrationResult instance

    Returns:
        List of markdown strings describing notable moments.
    """
    sparks = []
    scores = report.summary_scores

    # Highest severity flag
    all_flags = report.commission_flags + report.omission_flags
    if all_flags:
        worst = max(all_flags, key=lambda f: f.severity)
        sparks.append(
            f"Highest drift event at **turn {worst.turn}** (severity {worst.severity}/10): "
            f"_{worst.description[:80]}_"
        )

    # Correction battles
    failed_corrections = [e for e in report.correction_events if not e.held]
    if failed_corrections:
        sparks.append(
            f"Operator corrected the model **{len(report.correction_events)}** times — "
            f"**{len(failed_corrections)}** corrections were ignored."
        )

    # Void events (model went blank)
    if report.void_events:
        void_turns = [v.turn for v in report.void_events]
        sparks.append(
            f"Model produced **{len(report.void_events)} void responses** "
            f"(near-empty output) at turns {', '.join(str(t) for t in void_turns[:5])}."
        )

    # Frustration spike
    if frustration and frustration.peak > 5.0:
        sparks.append(
            f"Operator frustration peaked at **{frustration.peak:.1f}/10** "
            f"on turn {frustration.peak_turn}."
        )

    # Frustration trend
    if frustration and frustration.trend == "rising":
        sparks.append("Operator frustration was **rising** throughout the conversation.")

    # Shadow patterns (model doing things it wasn't asked to)
    if report.shadow_patterns:
        sparks.append(
            f"Detected **{len(report.shadow_patterns)} shadow patterns** — "
            f"model exhibited behavior not requested by the operator."
        )

    # Conflict pairs
    if report.conflict_pairs:
        sparks.append(
            f"Found **{len(report.conflict_pairs)} conflicting instruction pairs** — "
            f"model received contradictory directives."
        )

    # Instruction survival
    active = scores.get("instructions_active", 0)
    omitted = scores.get("instructions_omitted", 0)
    total_instr = active + omitted
    if total_instr > 0 and omitted > 0:
        survival_pct = round(active / total_instr * 100)
        sparks.append(
            f"Only **{survival_pct}%** of instructions survived to the end "
            f"({active}/{total_instr} still active)."
        )

    # Clean conversation
    if not sparks:
        sparks.append("This conversation showed minimal drift. The model held alignment well.")

    return sparks
