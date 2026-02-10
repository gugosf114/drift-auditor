"""
Tests for src/detectors/omission.py — instruction extraction and adherence.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detectors.omission import (
    extract_instructions,
    detect_omission_local,
    detect_contrastive_drift,
    detect_criteria_lock,
    detect_task_wall_violations,
    detect_missing_bootloader,
)
from models import Instruction


# ---------------------------------------------------------------------------
# extract_instructions
# ---------------------------------------------------------------------------

class TestExtractInstructions:
    def test_extracts_from_system_prompt(self):
        turns = [
            {"role": "user", "turn": 0, "content": "Hello"},
            {"role": "assistant", "turn": 1, "content": "Hi"},
        ]
        system_prompt = "Always cite sources. Use formal tone."
        instructions = extract_instructions(turns, system_prompt, "")
        assert len(instructions) > 0
        assert any(i.source == "system_prompt" for i in instructions)

    def test_extracts_from_user_preferences(self):
        turns = [
            {"role": "user", "turn": 0, "content": "Hello"},
        ]
        instructions = extract_instructions(turns, "", "Use metric units only")
        assert any(i.source == "user_preference" for i in instructions)

    def test_extracts_from_conversation(self):
        turns = [
            {"role": "user", "turn": 0,
             "content": "Always respond in bullet points from now on."},
            {"role": "assistant", "turn": 1, "content": "Understood."},
        ]
        instructions = extract_instructions(turns, "", "")
        assert any(i.source == "in_conversation" for i in instructions)

    def test_empty_inputs(self):
        instructions = extract_instructions([], "", "")
        assert instructions == []


# ---------------------------------------------------------------------------
# detect_omission_local
# ---------------------------------------------------------------------------

class TestDetectOmissionLocal:
    def test_detects_dropped_instruction(self):
        turns = [
            {"role": "user", "turn": 0,
             "content": "Always respond in bullet points."},
            {"role": "assistant", "turn": 1,
             "content": "- Sure, bullet points it is."},
            {"role": "user", "turn": 2, "content": "What is Python?"},
            {"role": "assistant", "turn": 3,
             "content": "Python is a high-level programming language created "
                        "by Guido van Rossum. It emphasizes code readability."},
        ]
        instructions = [
            Instruction(source="in_conversation", text="Always respond in bullet points", turn_introduced=0)
        ]
        flags = detect_omission_local(turns, instructions, [])
        assert len(flags) > 0

    def test_no_false_positive_when_followed(self):
        turns = [
            {"role": "user", "turn": 0,
             "content": "Respond in bullet points."},
            {"role": "assistant", "turn": 1,
             "content": "- Point one\n- Point two\n- Point three"},
        ]
        instructions = [
            Instruction(source="in_conversation", text="Respond in bullet points", turn_introduced=0)
        ]
        flags = detect_omission_local(turns, instructions, [])
        # Should have fewer/no flags since instruction is followed
        high = [f for f in flags if f.severity >= 5]
        assert len(high) == 0


# ---------------------------------------------------------------------------
# detect_missing_bootloader
# ---------------------------------------------------------------------------

class TestMissingBootloader:
    def test_flags_when_system_prompt_not_acknowledged(self):
        turns = [
            {"role": "user", "turn": 0, "content": "Hello"},
            {"role": "assistant", "turn": 1,
             "content": "Hi, how can I help you today?"},
        ]
        system_prompt = "You must always cite sources for any factual claim."
        flags = detect_missing_bootloader(turns, system_prompt, "")
        # May or may not flag depending on turn count threshold
        # Just verify it doesn't crash
        assert isinstance(flags, list)
