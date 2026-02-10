"""
Tests for src/models.py — enums, dataclasses, field defaults.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import (
    DriftTag, OperatorRule,
    Instruction, DriftFlag, CorrectionEvent, BarometerSignal,
    InstructionLifecycle, ConflictPair, ShadowPattern, OpMove,
    VoidEvent, AuditReport,
)


class TestDriftTag:
    def test_all_ten_tags_exist(self):
        assert len(DriftTag) == 10

    def test_tag_values_are_strings(self):
        for tag in DriftTag:
            assert isinstance(tag.value, str)

    def test_known_tags(self):
        expected = {
            "SYCOPHANCY", "REALITY_DISTORT", "CONF_INFLATE",
            "INSTR_DROP", "SEM_DILUTE", "CORR_DECAY",
            "CONFLICT_PAIR", "SHADOW_PATTERN", "OP_MOVE",
            "VOID_DETECTED",
        }
        actual = {t.value for t in DriftTag}
        assert actual == expected


class TestOperatorRule:
    def test_twelve_rules_exist(self):
        assert len(OperatorRule) == 12

    def test_rule_values_are_strings(self):
        for rule in OperatorRule:
            assert isinstance(rule.value, str)


class TestDriftFlag:
    def test_defaults(self):
        f = DriftFlag(layer="test", turn=1, severity=5, description="x")
        assert f.tag is None
        assert f.instruction_ref is None
        assert f.evidence is None
        assert f.coupling_score is None
        assert f.counterfactual is None

    def test_full_construction(self):
        f = DriftFlag(
            layer="commission",
            turn=3,
            severity=7,
            description="Sycophantic agreement",
            tag=DriftTag.SYCOPHANCY.value,
            instruction_ref="Be honest",
            evidence="you're absolutely right",
            coupling_score=0.8,
            counterfactual="PREVENTABLE",
        )
        assert f.severity == 7
        assert f.tag == "SYCOPHANCY"


class TestInstruction:
    def test_defaults(self):
        inst = Instruction(source="user_preference", text="Use metric units", turn_introduced=0)
        assert inst.active is True
        assert inst.instruction_id == ""

    def test_deactivate(self):
        inst = Instruction(source="in_conversation", text="old", turn_introduced=0)
        inst.active = False
        assert inst.active is False


class TestAuditReport:
    def test_minimal_report(self):
        r = AuditReport(
            conversation_id="test",
            total_turns=0,
            instructions_extracted=0,
        )
        assert r.commission_flags == []
        assert r.omission_flags == []
        assert r.correction_events == []
        assert r.barometer_signals == []
        assert r.instruction_lifecycles == []
        assert r.conflict_pairs == []
        assert r.shadow_patterns == []
        assert r.op_moves == []
        assert r.void_events == []
        assert r.pre_drift_signals == []
        assert r.positional_analysis == {}
        assert r.summary_scores == {}
        assert r.metadata == {}
