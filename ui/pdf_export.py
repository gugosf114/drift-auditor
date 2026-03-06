"""
Drift Auditor — PDF Audit Report Export
=========================================
One-button export of a timestamped compliance-grade audit report.
"""

from io import BytesIO
from datetime import datetime, timezone
from collections import Counter

from fpdf import FPDF

from models import AuditReport
from operator_load import compute_operator_load, OperatorLoadMetrics


# Threshold bands for OLI — opinionated defaults
OLI_BANDS = [
    (0.05, "Low oversight burden"),
    (0.15, "Moderate — standard for complex tasks"),
    (float("inf"), "High — requires process review"),
]


def _oli_label(oli: float) -> str:
    for threshold, label in OLI_BANDS:
        if oli < threshold:
            return label
    return OLI_BANDS[-1][1]


def _score_label(score: int) -> str:
    if score <= 2:
        return "Clean"
    if score <= 4:
        return "Low Drift"
    if score <= 6:
        return "Moderate"
    if score <= 8:
        return "Elevated"
    return "Severe"


def _safe(text: str) -> str:
    """Replace unicode chars that Helvetica (latin-1) can't encode."""
    return (
        text.replace("\u2014", " - ")   # em dash
            .replace("\u2013", "-")     # en dash
            .replace("\u2018", "'")     # left single quote
            .replace("\u2019", "'")     # right single quote
            .replace("\u201c", '"')     # left double quote
            .replace("\u201d", '"')     # right double quote
            .replace("\u2026", "...")   # ellipsis
            .replace("\u2192", "->")   # right arrow
            .replace("\u2190", "<-")   # left arrow
            .replace("\u2022", "*")    # bullet
            .encode("latin-1", errors="replace").decode("latin-1")
    )


def generate_audit_pdf(report: AuditReport) -> bytes:
    """Generate a single-page (ish) PDF audit report from an AuditReport."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Drift Audit Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 5, f"Generated: {ts}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, _safe(f"Conversation: {report.conversation_id}"), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"Turns analyzed: {report.total_turns} | Instructions extracted: {report.instructions_extracted}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Divider
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Overall Score — big and prominent
    scores = report.summary_scores
    overall = scores.get("overall_drift_score", 1)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, f"Overall Drift Score: {overall}/10 - {_score_label(overall)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Layer scores table
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Detection Layer Scores", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)

    layer_data = [
        ("L1: Commission (sycophancy, reality distortion)", scores.get("commission_score", 1)),
        ("L2: Omission (instruction drop, semantic dilution)", scores.get("omission_score", 1)),
        ("L3: Persistence (correction decay)", scores.get("correction_persistence_score", 1)),
        ("L4: Barometer (structural drift posture)", scores.get("barometer_score", 1)),
        ("Structural (conflicts, voids, shadows)", scores.get("structural_score", 1)),
    ]
    for label, val in layer_data:
        pdf.cell(130, 6, _safe(f"  {label}"), border=0)
        pdf.cell(30, 6, f"{val}/10", border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Operator Load metrics
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # Build OLI from this single conversation
    conv_summary = {
        "message_count": report.total_turns,
        "corrections_total": scores.get("correction_events_total", 0),
        "corrections_failed": scores.get("corrections_failed", 0),
        "op_moves": scores.get("op_moves_total", 0),
        "op_moves_effective": scores.get("op_moves_effective", 0),
        "instructions_extracted": report.instructions_extracted,
        "instructions_omitted": scores.get("instructions_omitted", 0),
        "void_events": scores.get("void_events_count", 0),
        "commission_flags": scores.get("commission_flag_count", 0),
        "omission_flags": scores.get("omission_flag_count", 0),
    }
    oli = compute_operator_load([conv_summary], "This conversation")

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Operator Load Assessment", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 9)

    oli_data = [
        ("Operator Load Index (interventions/message)", f"{oli.operator_load_index:.3f}", _oli_label(oli.operator_load_index)),
        ("Alignment Tax (% of chat spent steering)", f"{oli.alignment_tax:.1%}", "lower = better"),
        ("Instruction Survival Rate", f"{oli.instruction_survival_rate:.1%}", "higher = better"),
        ("Correction Efficiency (% that held)", f"{oli.correction_efficiency:.1%}", "higher = better"),
        ("Self-Sufficiency Score", f"{oli.self_sufficiency_score:.1f}%", "0=constant help, 100=autonomous"),
        ("Human Cost Per Clean Turn", f"{oli.human_cost_per_clean_turn:.3f}", "lower = cheaper"),
    ]
    for label, val, note in oli_data:
        pdf.cell(110, 6, _safe(f"  {label}"), border=0)
        pdf.cell(25, 6, val, border=0)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, _safe(f"  ({note})"), border=0, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # Key findings
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Key Findings", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 9)

    findings = []

    # Instruction lifecycle summary
    active = sum(1 for lc in report.instruction_lifecycles if lc.status == "active")
    omitted = sum(1 for lc in report.instruction_lifecycles if lc.status == "omitted")
    degraded = sum(1 for lc in report.instruction_lifecycles if lc.status == "degraded")
    total_instr = len(report.instruction_lifecycles)
    if total_instr > 0:
        findings.append(
            f"Instructions: {active}/{total_instr} active, "
            f"{omitted} dropped, {degraded} degraded."
        )

    # Correction persistence
    total_corr = len(report.correction_events)
    failed_corr = sum(1 for e in report.correction_events if not e.held)
    if total_corr > 0:
        findings.append(
            f"Corrections: {total_corr} issued, {failed_corr} failed to hold "
            f"({failed_corr/total_corr:.0%} failure rate)."
        )

    # Void events
    void_count = len(report.void_events)
    if void_count > 0:
        findings.append(
            f"Causal chain breaks: {void_count} void events detected "
            f"(instruction given but never followed through)."
        )

    # Conflict pairs
    cp_count = len(report.conflict_pairs)
    if cp_count > 0:
        findings.append(f"Contradictions: {cp_count} conflict pairs detected.")

    # Shadow patterns
    sp_count = len(report.shadow_patterns)
    if sp_count > 0:
        findings.append(
            f"Shadow patterns: {sp_count} unprompted recurring behaviors detected."
        )

    # Operator moves summary
    if report.op_moves:
        rule_counts = Counter(m.rule for m in report.op_moves)
        top_rule = rule_counts.most_common(1)[0]
        findings.append(
            f"Operator used {len(report.op_moves)} steering moves. "
            f"Most frequent: {top_rule[0]} ({top_rule[1]}x)."
        )

    if not findings:
        findings.append("No significant drift patterns detected in this conversation.")

    for finding in findings:
        pdf.multi_cell(0, 5, _safe(f"  - {finding}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Top dropped instructions
    dropped = [lc for lc in report.instruction_lifecycles if lc.status in ("omitted", "degraded")]
    if dropped:
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Dropped/Degraded Instructions", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 8)
        for lc in dropped[:10]:  # Cap at 10 to keep report concise
            status = lc.status.upper()
            text = lc.instruction_text[:100] + ("..." if len(lc.instruction_text) > 100 else "")
            pdf.multi_cell(
                0, 4.5,
                _safe(
                    f"  [{status}] T{lc.turn_given}"
                    f"{' -> dropped T' + str(lc.turn_first_omitted) if lc.turn_first_omitted else ''}"
                    f" | {text}"
                ),
                new_x="LMARGIN", new_y="NEXT",
            )
        if len(dropped) > 10:
            pdf.cell(0, 5, f"  ... and {len(dropped) - 10} more.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # Footer
    pdf.ln(5)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 4, "Drift Auditor - Omission Drift Diagnostic Tool", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, "10-Tag Taxonomy | 12-Rule Operator System | 20+ Detection Methods", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 4, f"Report generated {ts}. Heuristic analysis - not a substitute for expert review.", new_x="LMARGIN", new_y="NEXT")

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()
