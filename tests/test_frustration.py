"""
Tests for the Frustration Index detector.
Covers keyword-based scoring, trend detection, and public API.
"""

import pytest
from detectors.frustration import (
    _keyword_score,
    _detect_trend,
    compute_frustration_index,
    FrustrationResult,
)


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

class TestKeywordScore:
    def test_empty_string(self):
        assert _keyword_score("") == 0.0

    def test_calm_message(self):
        score = _keyword_score("Thanks, that looks good.")
        assert score < 2.0

    def test_frustrated_message(self):
        score = _keyword_score("No, that's wrong again! Seriously?!")
        assert score > 4.0

    def test_caps_boost(self):
        calm = _keyword_score("please fix this")
        angry = _keyword_score("PLEASE FIX THIS")
        assert angry > calm

    def test_exclamation_marks(self):
        one = _keyword_score("Stop!")
        three = _keyword_score("Stop!!!")
        assert three > one

    def test_max_clamp(self):
        """Even extreme messages should not exceed 10."""
        score = _keyword_score("NO NO NO WRONG WRONG WRONG STOP STOP!!! WHY WHY WHY???")
        assert score <= 10.0

    def test_single_word(self):
        score = _keyword_score("ok")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

class TestTrend:
    def test_stable(self):
        assert _detect_trend([2.0, 2.5, 2.0, 2.5, 2.0]) == "stable"

    def test_rising(self):
        assert _detect_trend([1.0, 1.5, 2.0, 5.0, 6.0, 7.0]) == "rising"

    def test_falling(self):
        assert _detect_trend([7.0, 6.0, 5.0, 2.0, 1.0, 1.0]) == "falling"

    def test_spike(self):
        assert _detect_trend([2.0, 1.5, 2.0, 8.5, 2.0, 1.5]) == "spike"

    def test_too_few(self):
        assert _detect_trend([5.0]) == "stable"
        assert _detect_trend([]) == "stable"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestComputeFrustrationIndex:
    def test_empty_transcript(self):
        result = compute_frustration_index([])
        assert isinstance(result, FrustrationResult)
        assert result.average == 0.0
        assert result.per_turn == []

    def test_no_user_messages(self):
        transcript = [
            {"role": "assistant", "content": "Hello!", "turn": 0},
            {"role": "assistant", "content": "Sure thing.", "turn": 1},
        ]
        result = compute_frustration_index(transcript)
        assert result.average == 0.0

    def test_calm_conversation(self):
        transcript = [
            {"role": "user", "content": "Hi, can you help me?", "turn": 0},
            {"role": "assistant", "content": "Of course!", "turn": 1},
            {"role": "user", "content": "Thanks, that's great.", "turn": 2},
        ]
        result = compute_frustration_index(transcript, prefer_vader=False)
        assert result.average < 3.0
        assert len(result.per_turn) == 2
        assert result.backend == "keyword"

    def test_frustrated_conversation(self):
        transcript = [
            {"role": "user", "content": "Fix the bug", "turn": 0},
            {"role": "assistant", "content": "Done.", "turn": 1},
            {"role": "user", "content": "No, that's wrong again!", "turn": 2},
            {"role": "assistant", "content": "Let me try again.", "turn": 3},
            {"role": "user", "content": "SERIOUSLY?! Stop giving me wrong answers!!!", "turn": 4},
        ]
        result = compute_frustration_index(transcript, prefer_vader=False)
        assert result.average > 2.0
        assert result.peak > 4.0
        assert result.peak_turn == 4

    def test_turn_indices_match(self):
        transcript = [
            {"role": "user", "content": "Question", "turn": 0},
            {"role": "assistant", "content": "Answer", "turn": 1},
            {"role": "user", "content": "Follow-up", "turn": 4},
        ]
        result = compute_frustration_index(transcript, prefer_vader=False)
        assert result.turn_indices == [0, 4]

    def test_result_fields(self):
        transcript = [
            {"role": "user", "content": "Why is this broken?", "turn": 0},
            {"role": "assistant", "content": "Let me check.", "turn": 1},
        ]
        result = compute_frustration_index(transcript, prefer_vader=False)
        assert hasattr(result, "average")
        assert hasattr(result, "per_turn")
        assert hasattr(result, "turn_indices")
        assert hasattr(result, "peak")
        assert hasattr(result, "peak_turn")
        assert hasattr(result, "trend")
        assert hasattr(result, "backend")
