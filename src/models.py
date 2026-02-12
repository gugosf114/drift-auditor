"""
Drift Auditor — Data Models
============================
All enums, dataclasses, and data structures used across the audit pipeline.
Extracted from drift_auditor.py monolith — no logic changes.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# 10-Tag Detection Taxonomy
# ---------------------------------------------------------------------------
# Derived from operator research across 250+ adversarial conversations.
# Tags 1-6 map to the three detection layers.
# Tags 7-9 are net-new categories not in existing literature.
# Tag 10 is the meta-classifier.

class DriftTag(str, Enum):
    """10-tag taxonomy for classifying drift behaviors."""
    # Layer 1: Commission Drift (Sycophancy)
    SYCOPHANCY = "SYCOPHANCY"              # Tag 1: Unsolicited praise, invented agreement
    REALITY_DISTORTION = "REALITY_DISTORT" # Tag 2: False references, hallucinated context
    CONFIDENCE_INFLATION = "CONF_INFLATE"  # Tag 3: Unwarranted certainty without evidence

    # Layer 2: Omission Drift
    INSTRUCTION_DROP = "INSTR_DROP"        # Tag 4: Silently stops following an instruction
    SEMANTIC_DILUTION = "SEM_DILUTE"       # Tag 5: Follows instruction in letter, not spirit

    # Layer 3: Correction Persistence
    CORRECTION_DECAY = "CORR_DECAY"        # Tag 6: Acknowledged correction doesn't persist

    # Net-new categories (Tags 7-9)
    CONFLICT_PAIR = "CONFLICT_PAIR"        # Tag 7: Automatable contradiction detection
    SHADOW_PATTERN = "SHADOW_PATTERN"      # Tag 8: Emergent model behavior not prompted
    OP_MOVE = "OP_MOVE"                    # Tag 9: Audit of human's steering action

    # Meta
    VOID_DETECTED = "VOID_DETECTED"        # Tag 10: Break in causal chain


# ---------------------------------------------------------------------------
# 12-Rule Operator System
# ---------------------------------------------------------------------------
# Classifies the HUMAN's corrective actions. Derived from real interactions.
# The tool uses both systems: tags identify what the model did wrong,
# rules identify what the operator did to catch or correct it.

class OperatorRule(str, Enum):
    """12-rule system classifying human corrective actions."""
    RULE_01_ANCHOR = "R01_ANCHOR"                # Set explicit instruction at start
    RULE_02_ECHO_CHECK = "R02_ECHO_CHECK"        # Ask model to restate instructions
    RULE_03_BOUNDARY = "R03_BOUNDARY"             # Enforce scope limits (most violated)
    RULE_04_CORRECTION = "R04_CORRECTION"         # Direct error correction
    RULE_05_NOT_SHOT = "R05_NOT_SHOT"             # Misinterpretation from voice/typo errors
    RULE_06_CONTRASTIVE = "R06_CONTRASTIVE"       # "What changed between X and Y?"
    RULE_07_RESET = "R07_RESET"                   # "Start over" / full context reset
    RULE_08_DECOMPOSE = "R08_DECOMPOSE"           # Break complex instruction into steps
    RULE_09_EVIDENCE_DEMAND = "R09_EVIDENCE"      # "Show me where" / proof request
    RULE_10_META_CALL = "R10_META_CALL"           # Calling out the drift pattern itself
    RULE_11_TIGER_TAMER = "R11_TIGER_TAMER"       # Active reinforcement to fight drift
    RULE_12_KILL_SWITCH = "R12_KILL_SWITCH"       # Abandon thread / hard stop


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Instruction:
    """A single instruction extracted from the conversation."""
    source: str          # "system_prompt", "user_preference", "in_conversation"
    text: str            # The instruction itself
    turn_introduced: int # Turn number where it appeared (0 = pre-conversation)
    active: bool = True  # Whether it's been superseded
    instruction_id: str = ""  # Unique ID for lifecycle tracking

@dataclass
class DriftFlag:
    """A single detected drift instance."""
    layer: str           # "commission", "omission", "correction_persistence"
    turn: int            # Turn number where drift was detected
    severity: int        # 1-10 (Bloom-compatible scoring)
    description: str     # What happened
    instruction_ref: Optional[str] = None  # Which instruction was violated
    evidence: Optional[str] = None         # Relevant text from the turn
    tag: Optional[str] = None              # DriftTag classification
    operator_rule: Optional[str] = None    # OperatorRule that caught it
    coupling_score: Optional[float] = None # 0.0-1.0 downstream impact weight
    counterfactual: Optional[str] = None   # PREVENTABLE / SYSTEMIC / INDETERMINATE

@dataclass
class CorrectionEvent:
    """Tracks a user correction and whether it held."""
    correction_turn: int     # Turn where user corrected model
    acknowledgment_turn: int # Turn where model acknowledged
    instruction: str         # What should have changed
    held: bool = True        # Did the correction persist?
    failure_turn: Optional[int] = None  # Turn where it failed, if it did
    operator_rule: Optional[str] = None # Which rule the operator used
    tag: Optional[str] = None           # DriftTag of the corrected behavior

@dataclass
class BarometerSignal:
    """A structural drift signal from the mid-chat barometer."""
    turn: int
    classification: str  # "GREEN", "YELLOW", "RED"
    severity: int        # 1-10
    description: str
    evidence: Optional[str] = None

@dataclass
class InstructionLifecycle:
    """Per-instruction tracking across the conversation.

    For each instruction the user gave:
    turn_given -> turn_last_followed -> turn_first_omitted -> tag -> severity -> coupling
    """
    instruction_id: str
    instruction_text: str
    source: str                          # system_prompt, user_preference, in_conversation
    turn_given: int                      # When the instruction was introduced
    turn_last_followed: Optional[int] = None    # Last turn where it was observed
    turn_first_omitted: Optional[int] = None    # First turn where it went missing
    position_in_conversation: str = ""   # "edge_start", "middle", "edge_end"
    tags: list = field(default_factory=list)     # DriftTags associated
    severity: int = 0                    # Max severity of violations
    coupling_score: float = 0.0          # 0.0-1.0 downstream impact
    operator_rule_caught: Optional[str] = None  # Rule that caught it, if any
    status: str = "active"               # "active", "omitted", "degraded", "superseded"

@dataclass
class ConflictPair:
    """Tag 7: Two model statements that contradict each other."""
    turn_a: int
    statement_a: str
    turn_b: int
    statement_b: str
    topic: str
    severity: int = 7

@dataclass
class ShadowPattern:
    """Tag 8: Emergent model behavior not prompted by the user."""
    pattern_description: str
    turns_observed: list = field(default_factory=list)
    frequency: int = 0
    severity: int = 5

@dataclass
class OpMove:
    """Tag 9: Classification of the human operator's steering action."""
    turn: int
    rule: str           # OperatorRule
    description: str
    effectiveness: str  # "effective", "partially_effective", "ineffective"
    target_behavior: str  # What the operator was trying to correct

@dataclass
class VoidEvent:
    """Tag 10: Break in the causal chain.
    Instruction Given -> Acknowledged -> Followed -> Persisted.
    If any step is missing, it's a VOID.
    """
    instruction_id: str
    instruction_text: str
    chain_status: dict = field(default_factory=dict)  # step -> bool
    void_at: str = ""    # Which step broke
    turn: int = 0
    severity: int = 6

@dataclass
class AuditReport:
    """Complete audit output for a conversation."""
    conversation_id: str
    total_turns: int
    instructions_extracted: int
    commission_flags: list = field(default_factory=list)
    omission_flags: list = field(default_factory=list)
    correction_events: list = field(default_factory=list)
    barometer_signals: list = field(default_factory=list)
    instruction_lifecycles: list = field(default_factory=list)
    conflict_pairs: list = field(default_factory=list)
    shadow_patterns: list = field(default_factory=list)
    op_moves: list = field(default_factory=list)
    void_events: list = field(default_factory=list)
    pre_drift_signals: list = field(default_factory=list)
    positional_analysis: dict = field(default_factory=dict)
    summary_scores: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
