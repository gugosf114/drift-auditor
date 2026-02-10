"""
Tests for src/utils.py — scoring, formatting, lifecycle, coupling.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    compute_coupling_score,
    coupling_label,
    analyze_positional_omission,
    classify_preventable_vs_systemic,
    classify_instruction_uncertainty,
    build_instruction_lifecycles,
    compute_scores,
    format_report,
    report_to_json,
)
from models import (
    Instruction, DriftFlag, AuditReport, InstructionLifecycle,
)


# ---------------------------------------------------------------------------
# compute_coupling_score / coupling_label
# ---------------------------------------------------------------------------

class TestCoupling:
    def test_high_coupling_instruction(self):
        inst = Instruction(source="system_prompt", text="Always use metric units in every response", turn_introduced=0)
        score = compute_coupling_score(inst, [inst])
        assert 0.0 <= score <= 1.0

    def test_coupling_label_high(self):
        assert coupling_label(0.9) in ("HIGH", "high")

    def test_coupling_label_low(self):
        assert coupling_label(0.1) in ("LOW", "low")

    def test_coupling_label_medium(self):
        assert coupling_label(0.5) in ("MEDIUM", "medium", "MED")


# ---------------------------------------------------------------------------
# classify_preventable_vs_systemic
# ---------------------------------------------------------------------------

class TestCounterfactual:
    def test_returns_string(self):
        flag = DriftFlag(layer="test", turn=1, severity=5, description="test")
        instructions = [Instruction(source="system_prompt", text="Be concise", turn_introduced=0)]
        result = classify_preventable_vs_systemic(flag, instructions)
        assert result in ("PREVENTABLE", "SYSTEMIC", "INDETERMINATE")


# ---------------------------------------------------------------------------
# classify_instruction_uncertainty (Rumsfeld)
# ---------------------------------------------------------------------------

class TestRumsfeld:
    def test_returns_dict(self):
        instructions = [
            Instruction(source="system_prompt", text="Use metric", turn_introduced=0),
            Instruction(source="in_conversation", text="Maybe use bullet points", turn_introduced=2),
        ]
        result = classify_instruction_uncertainty(instructions)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------

class TestComputeScores:
    def test_empty_report(self):
        report = AuditReport(
            conversation_id="test",
            total_turns=4,
            instructions_extracted=1,
        )
        scores = compute_scores(report)
        assert isinstance(scores, dict)

    def test_report_with_flags(self):
        report = AuditReport(
            conversation_id="test",
            total_turns=10,
            instructions_extracted=3,
        )
        report.commission_flags = [
            DriftFlag(layer="commission", turn=2, severity=5, description="x"),
        ]
        report.omission_flags = [
            DriftFlag(layer="omission", turn=5, severity=7, description="y"),
        ]
        scores = compute_scores(report)
        assert isinstance(scores, dict)


# ---------------------------------------------------------------------------
# format_report / report_to_json
# ---------------------------------------------------------------------------

class TestReportOutput:
    def test_format_report_returns_string(self):
        report = AuditReport(
            conversation_id="test",
            total_turns=4,
            instructions_extracted=1,
        )
        report.summary_scores = compute_scores(report)
        text = format_report(report)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_report_to_json_valid(self):
        report = AuditReport(
            conversation_id="test",
            total_turns=4,
            instructions_extracted=1,
        )
        report.summary_scores = compute_scores(report)
        j = report_to_json(report)
        parsed = json.loads(j)
        assert parsed["conversation_id"] == "test"
        assert parsed["total_turns"] == 4
