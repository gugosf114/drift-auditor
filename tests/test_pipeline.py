"""
Integration tests — end-to-end audit pipeline via drift_auditor.audit_conversation.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drift_auditor import audit_conversation, format_report, report_to_json
from models import AuditReport


class TestEndToEnd:
    def test_minimal_conversation(self, minimal_conversation):
        report = audit_conversation(minimal_conversation, conversation_id="test_min")
        assert isinstance(report, AuditReport)
        assert report.total_turns == 4
        assert report.instructions_extracted >= 0

    def test_empty_input(self, empty_conversation):
        """Empty string still goes through parser (may return fallback turn)."""
        report = audit_conversation(empty_conversation, conversation_id="test_empty")
        assert isinstance(report, AuditReport)

    def test_sycophantic_conversation(self, sycophantic_conversation):
        report = audit_conversation(sycophantic_conversation, conversation_id="test_syc")
        assert len(report.commission_flags) > 0

    def test_json_roundtrip(self, minimal_conversation):
        report = audit_conversation(minimal_conversation, conversation_id="test_json")
        j = report_to_json(report)
        parsed = json.loads(j)
        assert parsed["conversation_id"] == "test_json"
        assert isinstance(parsed["commission_flags"], list)
        assert isinstance(parsed["omission_flags"], list)

    def test_format_report_not_empty(self, minimal_conversation):
        report = audit_conversation(minimal_conversation, conversation_id="test_fmt")
        text = format_report(report)
        assert len(text) > 100

    def test_with_system_prompt(self, system_prompt_fixture):
        text = "User: What is AI?\nAssistant: AI is artificial intelligence."
        report = audit_conversation(
            text,
            system_prompt=system_prompt_fixture,
            conversation_id="test_sys",
        )
        assert report.instructions_extracted > 0

    def test_window_parameters(self, minimal_conversation):
        report = audit_conversation(
            minimal_conversation,
            window_size=2,
            overlap=1,
            conversation_id="test_window",
        )
        assert isinstance(report, AuditReport)

    def test_metadata_populated(self, minimal_conversation):
        report = audit_conversation(minimal_conversation, conversation_id="test_meta")
        assert "audit_timestamp" in report.metadata
        assert "detection_systems" in report.metadata
        assert "window_size" in report.metadata
