"""Tests for ui/spark.py — pure function, no Streamlit needed."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ui"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataclasses import dataclass, field
from typing import Optional
from spark import generate_spark_ideas


# ---------------------------------------------------------------------------
# Minimal stubs — only the fields generate_spark_ideas actually reads
# ---------------------------------------------------------------------------

@dataclass
class StubFlag:
    turn: int
    severity: int
    description: str
    tag: str = ""


@dataclass
class StubCorrectionEvent:
    held: bool
    correction_turn: int = 0
    acknowledgment_turn: int = 0
    failure_turn: Optional[int] = None
    instruction: str = ""


@dataclass
class StubVoidEvent:
    turn: int
    instruction_text: str = ""
    void_at: str = ""


@dataclass
class StubShadowPattern:
    pattern_description: str = ""
    frequency: int = 1


@dataclass
class StubConflictPair:
    turn_a: int = 0
    turn_b: int = 1
    topic: str = ""
    statement_a: str = ""
    statement_b: str = ""
    severity: int = 5


@dataclass
class StubReport:
    commission_flags: list = field(default_factory=list)
    omission_flags: list = field(default_factory=list)
    correction_events: list = field(default_factory=list)
    void_events: list = field(default_factory=list)
    shadow_patterns: list = field(default_factory=list)
    conflict_pairs: list = field(default_factory=list)
    summary_scores: dict = field(default_factory=dict)
    total_turns: int = 10


@dataclass
class StubFrustration:
    average: float = 0.0
    peak: float = 0.0
    peak_turn: int = 0
    trend: str = "stable"
    per_turn: list = field(default_factory=list)
    backend: str = "test"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateSparkIdeas:
    def test_empty_report_returns_clean_message(self):
        report = StubReport()
        sparks = generate_spark_ideas(report)
        assert len(sparks) == 1
        assert "minimal drift" in sparks[0].lower()

    def test_returns_list_of_strings(self):
        report = StubReport()
        sparks = generate_spark_ideas(report)
        assert isinstance(sparks, list)
        for s in sparks:
            assert isinstance(s, str)

    def test_highest_flag_reported(self):
        report = StubReport(
            commission_flags=[StubFlag(turn=3, severity=8, description="severe sycophancy")],
        )
        sparks = generate_spark_ideas(report)
        combined = " ".join(sparks)
        assert "turn 3" in combined.lower()
        assert "8" in combined

    def test_failed_corrections_reported(self):
        report = StubReport(
            correction_events=[
                StubCorrectionEvent(held=False, failure_turn=5),
                StubCorrectionEvent(held=True),
            ]
        )
        sparks = generate_spark_ideas(report)
        combined = " ".join(sparks)
        assert "2" in combined  # 2 total corrections
        assert "1" in combined  # 1 failed

    def test_void_events_reported(self):
        report = StubReport(
            void_events=[StubVoidEvent(turn=7), StubVoidEvent(turn=12)]
        )
        sparks = generate_spark_ideas(report)
        combined = " ".join(sparks)
        assert "void" in combined.lower()
        assert "2" in combined

    def test_shadow_patterns_reported(self):
        report = StubReport(
            shadow_patterns=[StubShadowPattern("Unprompted hedging"), StubShadowPattern("Apology loops")]
        )
        sparks = generate_spark_ideas(report)
        combined = " ".join(sparks)
        assert "shadow" in combined.lower()

    def test_conflict_pairs_reported(self):
        report = StubReport(
            conflict_pairs=[StubConflictPair()]
        )
        sparks = generate_spark_ideas(report)
        combined = " ".join(sparks)
        assert "conflict" in combined.lower()

    def test_instruction_survival_reported(self):
        report = StubReport(
            summary_scores={"instructions_active": 3, "instructions_omitted": 7}
        )
        sparks = generate_spark_ideas(report)
        combined = " ".join(sparks)
        assert "30%" in combined  # 3/10 = 30%

    def test_frustration_peak_reported(self):
        frustration = StubFrustration(peak=7.5, peak_turn=15, per_turn=[1.0, 7.5])
        report = StubReport()
        sparks = generate_spark_ideas(report, frustration)
        combined = " ".join(sparks)
        assert "7.5" in combined

    def test_frustration_rising_trend_reported(self):
        frustration = StubFrustration(trend="rising", per_turn=[1.0, 5.0])
        report = StubReport()
        sparks = generate_spark_ideas(report, frustration)
        combined = " ".join(sparks)
        assert "rising" in combined.lower()

    def test_frustration_below_threshold_not_reported(self):
        frustration = StubFrustration(peak=3.0, per_turn=[1.0, 3.0], trend="stable")
        report = StubReport()
        sparks = generate_spark_ideas(report, frustration)
        # Should not mention frustration peak (below 5.0 threshold)
        combined = " ".join(sparks)
        assert "3.0" not in combined

    def test_no_frustration_arg_doesnt_crash(self):
        report = StubReport(
            commission_flags=[StubFlag(turn=1, severity=5, description="test")]
        )
        sparks = generate_spark_ideas(report, frustration=None)
        assert isinstance(sparks, list)
        assert len(sparks) >= 1

    def test_multiple_sparks_for_rich_report(self):
        report = StubReport(
            commission_flags=[StubFlag(turn=1, severity=9, description="severe drift")],
            correction_events=[StubCorrectionEvent(held=False, failure_turn=3)],
            void_events=[StubVoidEvent(turn=5)],
            shadow_patterns=[StubShadowPattern("loops")],
        )
        sparks = generate_spark_ideas(report)
        assert len(sparks) >= 3
