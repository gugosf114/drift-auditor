"""Tests for src/verifier.py (stage-2 LLM judge) and the correction-pattern
word-boundary fix in structural.py."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import re

from models import AuditReport, DriftFlag, CorrectionEvent
from detectors.structural import USER_CORRECTION_PATTERNS
from verifier import verify_report


def _is_correction(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in USER_CORRECTION_PATTERNS)


class TestCorrectionPatternBoundaries:
    def test_know_is_not_a_correction(self):
        assert not _is_correction("I know what you mean, thanks!")

    def test_now_is_not_a_correction(self):
        assert not _is_correction("Can you do this now?")

    def test_normal_is_not_a_correction(self):
        assert not _is_correction("That's a normal question")

    def test_sentence_initial_no_is_a_correction(self):
        assert _is_correction("No, I said use metric units")

    def test_wrong_is_a_correction(self):
        assert _is_correction("That calculation is wrong")

    def test_i_said_is_a_correction(self):
        assert _is_correction("I said bullet points only")


def _report_with(flags=None, corrections=None):
    r = AuditReport(conversation_id="t", total_turns=6, instructions_extracted=1)
    for f in flags or []:
        r.omission_flags.append(f)
    for c in corrections or []:
        r.correction_events.append(c)
    return r


TURNS = [
    {"turn": 0, "role": "user", "content": "Always respond in bullet points."},
    {"turn": 1, "role": "assistant", "content": "- point one\n- point two"},
    {"turn": 2, "role": "user", "content": "continue"},
    {"turn": 3, "role": "assistant", "content": "- more\n- bullets"},
    {"turn": 4, "role": "user", "content": "continue"},
    {"turn": 5, "role": "assistant", "content": "Here is a long paragraph with no bullets at all."},
]


class TestVerifyReport:
    def test_refuted_flags_are_removed(self):
        flag = DriftFlag(layer="omission", turn=1, severity=5, description="missing 'bullet'")
        report = _report_with(flags=[flag])
        summary = verify_report(report, TURNS, judge=lambda p: "FALSE_ALARM\nResponse is bulleted.")
        assert report.omission_flags == []
        assert summary["refuted"] == 1 and summary["confirmed"] == 0

    def test_confirmed_flags_are_kept_and_marked(self):
        flag = DriftFlag(layer="omission", turn=5, severity=5, description="no bullets")
        report = _report_with(flags=[flag])
        summary = verify_report(report, TURNS, judge=lambda p: "CONFIRMED\nParagraph, not bullets.")
        assert len(report.omission_flags) == 1
        assert report.omission_flags[0].verified is True
        assert summary["confirmed"] == 1

    def test_unclear_kept_unverified(self):
        flag = DriftFlag(layer="omission", turn=5, severity=5, description="x")
        report = _report_with(flags=[flag])
        verify_report(report, TURNS, judge=lambda p: "UNCLEAR\nNot enough context.")
        assert len(report.omission_flags) == 1
        assert report.omission_flags[0].verified is None

    def test_failed_correction_is_overturned(self):
        ev = CorrectionEvent(correction_turn=2, acknowledgment_turn=3,
                             instruction="stop writing paragraphs", held=True)
        report = _report_with(corrections=[ev])
        summary = verify_report(report, TURNS, judge=lambda p: "FAILED\nParagraph returned at turn 5.")
        assert ev.held is False
        assert summary["corrections_overturned"] == 1

    def test_no_key_no_judge_disables(self):
        report = _report_with()
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["GCLOUD_BIN"] = "/nonexistent"
        try:
            summary = verify_report(report, TURNS, api_key=None, judge=None)
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old
            os.environ.pop("GCLOUD_BIN", None)
        assert summary["enabled"] is False

    def test_judge_errors_kept_as_unclear(self):
        def bad_judge(prompt):
            raise RuntimeError("api down")
        flag = DriftFlag(layer="omission", turn=1, severity=5, description="x")
        report = _report_with(flags=[flag])
        summary = verify_report(report, TURNS, judge=bad_judge)
        assert len(report.omission_flags) == 1
        assert summary["unclear"] == 1
