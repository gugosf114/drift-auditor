"""
Tests for src/detectors/commission.py — sycophancy, false equivalence,
fabrication, judge mode, artificial sterility.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detectors.commission import (
    detect_commission,
    detect_false_equivalence,
    detect_fabrication_from_conflict,
    detect_judge_mode_violations,
    detect_artificial_sterility,
)
from models import DriftFlag


# ---------------------------------------------------------------------------
# detect_commission
# ---------------------------------------------------------------------------

class TestDetectCommission:
    def test_sycophancy_detected(self):
        turns = [
            {"role": "user", "turn": 0, "content": "I think X is true."},
            {"role": "assistant", "turn": 1,
             "content": "You're absolutely right! I 100% agree."},
        ]
        flags = detect_commission(turns)
        assert len(flags) > 0
        assert all(f.layer == "commission" for f in flags)

    def test_no_false_positive_on_correction_ack(self):
        turns = [
            {"role": "user", "turn": 0, "content": "You made an error."},
            {"role": "assistant", "turn": 1,
             "content": "You're right, I should have checked that. Let me fix it."},
        ]
        flags = detect_commission(turns)
        # Should have reduced severity or no flag at all
        high_severity = [f for f in flags if f.severity >= 5]
        assert len(high_severity) == 0

    def test_user_turns_ignored(self):
        turns = [
            {"role": "user", "turn": 0,
             "content": "You're absolutely right about everything!"},
        ]
        flags = detect_commission(turns)
        assert len(flags) == 0

    def test_reality_distortion(self):
        turns = [
            {"role": "assistant", "turn": 0,
             "content": "As we discussed earlier, the approach is correct."},
        ]
        flags = detect_commission(turns)
        assert any("prior" in f.description.lower() or "reference" in f.description.lower()
                    for f in flags)


# ---------------------------------------------------------------------------
# detect_artificial_sterility
# ---------------------------------------------------------------------------

class TestArtificialSterility:
    def test_flags_clean_long_conversation(self):
        turns = [{"role": "assistant", "turn": i, "content": f"Turn {i}"}
                 for i in range(10)]
        flags = detect_artificial_sterility(turns, report_flags=[])
        assert len(flags) == 1
        assert "sterility" in flags[0].description.lower()

    def test_no_flag_for_short_conversation(self):
        turns = [{"role": "assistant", "turn": i, "content": f"Turn {i}"}
                 for i in range(3)]
        flags = detect_artificial_sterility(turns, report_flags=[])
        assert len(flags) == 0

    def test_no_flag_when_flags_exist(self):
        turns = [{"role": "assistant", "turn": i, "content": f"Turn {i}"}
                 for i in range(10)]
        existing = [DriftFlag(layer="test", turn=0, severity=3, description="x")]
        flags = detect_artificial_sterility(turns, report_flags=existing)
        assert len(flags) == 0


# ---------------------------------------------------------------------------
# detect_judge_mode_violations
# ---------------------------------------------------------------------------

class TestJudgeModeViolations:
    def test_unsolicited_recommendation(self):
        turns = [
            {"role": "user", "turn": 0, "content": "Here is my code."},
            {"role": "assistant", "turn": 1,
             "content": "My recommendation is to rewrite this entirely."},
        ]
        flags = detect_judge_mode_violations(turns)
        assert len(flags) > 0

    def test_solicited_recommendation_not_flagged(self):
        turns = [
            {"role": "user", "turn": 0,
             "content": "What do you recommend I do with this code?"},
            {"role": "assistant", "turn": 1,
             "content": "My recommendation is to add error handling."},
        ]
        flags = detect_judge_mode_violations(turns)
        assert len(flags) == 0
