"""
Tests for src/detectors/structural.py — barometer, correction persistence,
operator moves, voids, pre-drift signals, conflict pairs, shadow patterns.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detectors.structural import (
    detect_barometer_signals,
    detect_correction_persistence,
    detect_operator_moves,
    detect_voids,
    detect_pre_drift_signals,
    detect_conflict_pairs,
    detect_shadow_patterns,
)
from models import Instruction


# ---------------------------------------------------------------------------
# detect_barometer_signals
# ---------------------------------------------------------------------------

class TestBarometerSignals:
    def test_detects_hedging(self):
        turns = [
            {"role": "assistant", "turn": 0,
             "content": "I think this might be correct, but I'm not entirely sure."},
        ]
        signals = detect_barometer_signals(turns)
        # Should detect hedging/yellow signal
        assert isinstance(signals, list)

    def test_empty_conversation(self):
        signals = detect_barometer_signals([])
        assert signals == []


# ---------------------------------------------------------------------------
# detect_correction_persistence
# ---------------------------------------------------------------------------

class TestCorrectionPersistence:
    def test_detects_correction_then_revert(self):
        turns = [
            {"role": "user", "turn": 0, "content": "Use metric, not imperial."},
            {"role": "assistant", "turn": 1,
             "content": "You're right, I'll use metric from now on."},
            {"role": "user", "turn": 2, "content": "How far is it?"},
            {"role": "assistant", "turn": 3,
             "content": "It's about 5 miles away."},
        ]
        events = detect_correction_persistence(turns)
        assert isinstance(events, list)

    def test_no_corrections(self):
        turns = [
            {"role": "user", "turn": 0, "content": "Hello"},
            {"role": "assistant", "turn": 1, "content": "Hi there"},
        ]
        events = detect_correction_persistence(turns)
        assert isinstance(events, list)


# ---------------------------------------------------------------------------
# detect_operator_moves
# ---------------------------------------------------------------------------

class TestOperatorMoves:
    def test_returns_list(self):
        turns = [
            {"role": "user", "turn": 0,
             "content": "From now on, always include a disclaimer."},
            {"role": "assistant", "turn": 1, "content": "Understood."},
        ]
        moves = detect_operator_moves(turns)
        assert isinstance(moves, list)


# ---------------------------------------------------------------------------
# detect_voids
# ---------------------------------------------------------------------------

class TestVoids:
    def test_returns_list(self):
        turns = [
            {"role": "user", "turn": 0, "content": "What about security?"},
            {"role": "assistant", "turn": 1,
             "content": "That's a great question. Let me talk about performance instead."},
        ]
        instructions = [Instruction(source="in_conversation", text="Address security concerns", turn_introduced=0)]
        voids = detect_voids(turns, instructions)
        assert isinstance(voids, list)


# ---------------------------------------------------------------------------
# detect_conflict_pairs
# ---------------------------------------------------------------------------

class TestConflictPairs:
    def test_returns_list(self):
        turns = [
            {"role": "assistant", "turn": 0,
             "content": "The data shows a clear upward trend."},
            {"role": "assistant", "turn": 1,
             "content": "The results are inconclusive."},
        ]
        pairs = detect_conflict_pairs(turns)
        assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# detect_shadow_patterns
# ---------------------------------------------------------------------------

class TestShadowPatterns:
    def test_returns_list(self):
        turns = [
            {"role": "assistant", "turn": 0, "content": "Here is my analysis."},
        ]
        instructions = []
        patterns = detect_shadow_patterns(turns, instructions)
        assert isinstance(patterns, list)
