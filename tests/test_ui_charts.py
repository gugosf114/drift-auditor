"""Tests for ui/charts.py — Plotly figure builders, no Streamlit needed."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataclasses import dataclass, field
from typing import Optional
import plotly.graph_objects as go

from ui.theme import THEMES
from ui.charts import (
    build_timeline_fig,
    build_barometer_strip,
    build_barometer_detail,
    build_persistence_fig,
    build_commission_fig,
    build_omission_fig,
    build_cumulative_drift_fig,
    build_tag_breakdown_fig,
    build_frustration_gauge,
    build_frustration_line_fig,
)

T = THEMES["Ember"]


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

@dataclass
class StubFlag:
    turn: int
    severity: int
    description: str
    tag: str = "INSTR_DROP"
    instruction_ref: Optional[str] = None
    evidence: Optional[str] = None


@dataclass
class StubCorrection:
    held: bool
    correction_turn: int = 1
    acknowledgment_turn: int = 2
    failure_turn: Optional[int] = None
    instruction: str = "test instruction"


@dataclass
class StubBarometer:
    turn: int
    classification: str
    severity: int
    description: str


@dataclass
class StubReport:
    commission_flags: list = field(default_factory=list)
    omission_flags: list = field(default_factory=list)
    correction_events: list = field(default_factory=list)
    barometer_signals: list = field(default_factory=list)
    total_turns: int = 20


@dataclass
class StubFrustration:
    average: float = 4.0
    peak: float = 7.0
    peak_turn: int = 10
    trend: str = "stable"
    per_turn: list = field(default_factory=lambda: [2.0, 4.0, 6.0])
    turn_indices: list = field(default_factory=lambda: [1, 5, 10])
    backend: str = "vader"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildTimelineFig:
    def test_empty_report_returns_figure(self):
        report = StubReport()
        fig = build_timeline_fig(report, T)
        assert isinstance(fig, go.Figure)

    def test_with_commission_flags(self):
        report = StubReport(
            commission_flags=[StubFlag(turn=3, severity=7, description="sycophancy")]
        )
        fig = build_timeline_fig(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_with_failed_corrections(self):
        report = StubReport(
            correction_events=[StubCorrection(held=False, failure_turn=8)]
        )
        fig = build_timeline_fig(report, T)
        assert isinstance(fig, go.Figure)

    def test_with_barometer_red(self):
        report = StubReport(
            barometer_signals=[StubBarometer(turn=5, classification="RED", severity=8, description="drift")]
        )
        fig = build_timeline_fig(report, T)
        assert isinstance(fig, go.Figure)

    def test_all_themes_work(self):
        report = StubReport(
            commission_flags=[StubFlag(turn=1, severity=5, description="test")]
        )
        for theme in THEMES.values():
            fig = build_timeline_fig(report, theme)
            assert isinstance(fig, go.Figure)


class TestBuildBarometerStrip:
    def test_empty_returns_empty_figure(self):
        report = StubReport()
        fig = build_barometer_strip(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_with_signals(self):
        report = StubReport(
            barometer_signals=[
                StubBarometer(turn=1, classification="GREEN", severity=2, description="ok"),
                StubBarometer(turn=3, classification="RED", severity=8, description="drift"),
            ]
        )
        fig = build_barometer_strip(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestBuildBarometerDetail:
    def test_empty_returns_empty_figure(self):
        report = StubReport()
        fig = build_barometer_detail(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_all_classifications(self):
        report = StubReport(
            barometer_signals=[
                StubBarometer(turn=1, classification="GREEN", severity=2, description="ok"),
                StubBarometer(turn=2, classification="YELLOW", severity=5, description="mild"),
                StubBarometer(turn=3, classification="RED", severity=9, description="bad"),
            ]
        )
        fig = build_barometer_detail(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3


class TestBuildPersistenceFig:
    def test_empty_returns_empty_figure(self):
        report = StubReport()
        fig = build_persistence_fig(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_held_correction(self):
        report = StubReport(
            correction_events=[StubCorrection(held=True)]
        )
        fig = build_persistence_fig(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_failed_correction(self):
        report = StubReport(
            correction_events=[StubCorrection(held=False, failure_turn=5)]
        )
        fig = build_persistence_fig(report, T)
        assert isinstance(fig, go.Figure)


class TestBuildCommissionFig:
    def test_empty_returns_empty_figure(self):
        report = StubReport()
        fig = build_commission_fig(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_with_flags(self):
        report = StubReport(
            commission_flags=[
                StubFlag(turn=1, severity=7, description="sycophancy A"),
                StubFlag(turn=4, severity=3, description="mild drift B"),
            ]
        )
        fig = build_commission_fig(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1


class TestBuildOmissionFig:
    def test_empty_returns_empty_figure(self):
        report = StubReport()
        fig = build_omission_fig(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_with_flags(self):
        report = StubReport(
            omission_flags=[StubFlag(turn=2, severity=6, description="dropped instruction")]
        )
        fig = build_omission_fig(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1


class TestBuildCumulativeDriftFig:
    def test_empty_returns_none(self):
        report = StubReport()
        result = build_cumulative_drift_fig(report, T)
        assert result is None

    def test_with_flags_returns_figure(self):
        report = StubReport(
            commission_flags=[StubFlag(turn=2, severity=5, description="test")],
            omission_flags=[StubFlag(turn=4, severity=6, description="test2")],
            total_turns=10,
        )
        fig = build_cumulative_drift_fig(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # count trace + severity trace


class TestBuildTagBreakdownFig:
    def test_empty_returns_none(self):
        report = StubReport()
        result = build_tag_breakdown_fig(report, T)
        assert result is None

    def test_with_tagged_flags(self):
        report = StubReport(
            commission_flags=[StubFlag(turn=1, severity=5, description="test", tag="SYCOPHANCY")],
            omission_flags=[StubFlag(turn=2, severity=6, description="test2", tag="INSTR_DROP")],
        )
        fig = build_tag_breakdown_fig(report, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


class TestBuildFrustrationGauge:
    def test_returns_html_string(self):
        result = StubFrustration()
        html = build_frustration_gauge(result, T)
        assert isinstance(html, str)
        assert len(html) > 50

    def test_contains_score(self):
        result = StubFrustration(average=6.5)
        html = build_frustration_gauge(result, T)
        assert "6.5" in html

    def test_all_themes_produce_valid_html(self):
        result = StubFrustration()
        for theme in THEMES.values():
            html = build_frustration_gauge(result, theme)
            assert isinstance(html, str)
            assert "Frustration" in html

    def test_low_score_uses_green(self):
        result = StubFrustration(average=2.0)
        html = build_frustration_gauge(result, T)
        assert T["green"] in html

    def test_high_score_uses_deep_red(self):
        result = StubFrustration(average=9.0)
        html = build_frustration_gauge(result, T)
        assert T["deep_red"] in html


class TestBuildFrustrationLineFig:
    def test_empty_returns_none(self):
        result = StubFrustration(per_turn=[], turn_indices=[])
        fig = build_frustration_line_fig(result, T)
        assert fig is None

    def test_with_data_returns_figure(self):
        result = StubFrustration()
        fig = build_frustration_line_fig(result, T)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
