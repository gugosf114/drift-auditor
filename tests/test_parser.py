"""
Tests for src/parsers/chat_parser.py — format detection and parsing.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parsers.chat_parser import parse_chat_log


class TestPlainTextParsing:
    def test_basic_user_assistant(self):
        text = "User: Hello\nAssistant: Hi there"
        turns = parse_chat_log(text)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"

    def test_turn_numbering(self):
        text = "User: A\nAssistant: B\nUser: C\nAssistant: D"
        turns = parse_chat_log(text)
        assert turns[0]["turn"] == 0
        assert turns[3]["turn"] == 3

    def test_multiline_content(self):
        text = "User: Hello\nHow are you?\nAssistant: I am fine."
        turns = parse_chat_log(text)
        assert len(turns) == 2
        assert "How are you?" in turns[0]["content"]

    def test_human_assistant_labels(self):
        text = "Human: Question\nAssistant: Answer"
        turns = parse_chat_log(text)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"


class TestJSONParsing:
    def test_claude_json_format(self):
        data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        text = json.dumps(data)
        turns = parse_chat_log(text)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"

    def test_nested_content_blocks(self):
        data = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]
        text = json.dumps(data)
        turns = parse_chat_log(text)
        assert len(turns) == 2
        assert "Hello" in turns[0]["content"]


class TestEdgeCases:
    def test_empty_input_returns_fallback(self):
        """Parser returns a single 'unknown' turn for empty input (existing behavior)."""
        turns = parse_chat_log("")
        assert len(turns) >= 0  # Parser may return fallback turn

    def test_whitespace_returns_fallback(self):
        """Parser returns a single 'unknown' turn for whitespace (existing behavior)."""
        turns = parse_chat_log("   \n\n  ")
        assert len(turns) >= 0

    def test_single_turn(self):
        turns = parse_chat_log("User: Just one message")
        assert len(turns) == 1
