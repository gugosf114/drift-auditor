"""
Frustration Index — Operator Sentiment Proxy
=============================================
Experimental proxy measuring operator frustration from message sentiment.
Not a clinical measure — a lightweight signal based on linguistic markers
in operator (user) messages.

Two tiers:
  1. Keyword-based (zero dependencies, always available)
  2. VADER sentiment (pip install vaderSentiment — better accuracy)

Both produce per-turn scores (0–10) and an aggregate index.

Author: George Abrahamyants
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class FrustrationResult:
    """Container for frustration analysis output."""
    average: float = 0.0               # Overall 0–10
    per_turn: list[float] = field(default_factory=list)   # One score per operator msg
    turn_indices: list[int] = field(default_factory=list)  # Conversation turn index
    peak: float = 0.0                  # Max single-message score
    peak_turn: int = 0                 # Turn where peak occurred
    trend: str = "stable"              # "rising", "falling", "stable", "spike"
    backend: str = "keyword"           # "keyword" or "vader"


# ---------------------------------------------------------------------------
# Keyword markers (Tier 1 — zero dependencies)
# ---------------------------------------------------------------------------

_FRUSTRATION_WORDS = {
    # Strong frustration
    "wrong": 3.0, "no": 1.5, "stop": 2.5, "again": 2.0,
    "seriously": 2.5, "please": 1.0, "why": 1.5,
    "broken": 2.5, "terrible": 3.0, "awful": 3.0,
    "useless": 3.5, "fail": 2.5, "failed": 2.5,
    "stupid": 3.0, "ridiculous": 3.0, "unacceptable": 3.5,
    "disappointed": 2.5, "frustrating": 3.5, "frustrated": 3.5,
    "annoying": 2.5, "annoyed": 2.5, "confusing": 2.0,
    "incorrect": 2.5, "not": 1.0, "never": 2.0,
    "ugh": 2.5, "wtf": 4.0, "damn": 2.5,
}

_CAPS_THRESHOLD = 0.4   # Fraction of alpha chars that are uppercase
_EXCLAMATION_WEIGHT = 1.5
_QUESTION_WEIGHT = 0.5   # Repeated questions signal confusion/frustration


def _keyword_score(text: str) -> float:
    """Score a single message 0–10 using keyword markers + punctuation."""
    if not text or not text.strip():
        return 0.0

    score = 0.0
    words = re.findall(r"[a-zA-Z']+", text.lower())

    # Word matches
    for w in words:
        if w in _FRUSTRATION_WORDS:
            score += _FRUSTRATION_WORDS[w]

    # Caps emphasis (only if message has enough alpha chars)
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) > 5:
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if caps_ratio > _CAPS_THRESHOLD:
            score += 2.0 + (caps_ratio - _CAPS_THRESHOLD) * 5.0

    # Exclamation marks
    excl_count = text.count("!")
    if excl_count > 0:
        score += min(excl_count * _EXCLAMATION_WEIGHT, 4.0)

    # Repeated question marks (confusion/exasperation)
    if text.count("?") >= 2:
        score += _QUESTION_WEIGHT * min(text.count("?"), 4)

    # Normalize: logarithmic compression to 0–10
    if score <= 0:
        return 0.0
    # Soft ceiling via log scaling
    normalized = min(10.0, 2.5 * math.log1p(score))
    return round(normalized, 2)


# ---------------------------------------------------------------------------
# VADER sentiment (Tier 2 — optional dependency)
# ---------------------------------------------------------------------------

def _try_vader_score(text: str) -> Optional[float]:
    """Score using VADER if available. Returns None if not installed."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return None

    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)

    # Invert compound: -1 (most negative) → 10, +1 (most positive) → 0
    # compound range is [-1, 1]
    frustration = (1.0 - vs["compound"]) * 5.0  # base: 0–10

    # Boost for neutral-heavy messages with negative words (VADER under-weights caps)
    if vs["neg"] > 0.15:
        frustration += vs["neg"] * 3.0

    return round(max(0.0, min(10.0, frustration)), 2)


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

def _detect_trend(scores: list[float]) -> str:
    """Classify frustration trajectory over the conversation."""
    if len(scores) < 3:
        return "stable"

    # Split into halves
    mid = len(scores) // 2
    first_half = sum(scores[:mid]) / max(mid, 1)
    second_half = sum(scores[mid:]) / max(len(scores) - mid, 1)
    delta = second_half - first_half

    # Check for spike: any single message > 7 while average < 5
    avg = sum(scores) / len(scores)
    peak = max(scores)
    if peak > 7.0 and avg < 5.0:
        return "spike"

    if delta > 1.5:
        return "rising"
    elif delta < -1.5:
        return "falling"
    return "stable"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_frustration_index(
    transcript: list[dict],
    prefer_vader: bool = True,
) -> FrustrationResult:
    """
    Compute frustration index from a parsed conversation transcript.

    Parameters
    ----------
    transcript : list[dict]
        Output of parse_chat_log — each dict has "role", "content", "turn".
    prefer_vader : bool
        If True and vaderSentiment is installed, use VADER. Falls back to keywords.

    Returns
    -------
    FrustrationResult
        Aggregate + per-turn frustration scores.
    """
    operator_msgs = [
        (msg["turn"], msg["content"])
        for msg in transcript
        if msg.get("role") == "user" and msg.get("content", "").strip()
    ]

    if not operator_msgs:
        return FrustrationResult()

    # Decide backend
    backend = "keyword"
    if prefer_vader:
        test_score = _try_vader_score("test")
        if test_score is not None:
            backend = "vader"

    scores: list[float] = []
    turn_indices: list[int] = []

    for turn_idx, content in operator_msgs:
        if backend == "vader":
            s = _try_vader_score(content)
            if s is None:
                s = _keyword_score(content)
        else:
            s = _keyword_score(content)
        scores.append(s)
        turn_indices.append(turn_idx)

    avg = sum(scores) / len(scores) if scores else 0.0
    peak = max(scores) if scores else 0.0
    peak_idx = scores.index(peak) if scores else 0
    peak_turn = turn_indices[peak_idx] if turn_indices else 0

    return FrustrationResult(
        average=round(avg, 2),
        per_turn=scores,
        turn_indices=turn_indices,
        peak=peak,
        peak_turn=peak_turn,
        trend=_detect_trend(scores),
        backend=backend,
    )
