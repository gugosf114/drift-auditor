"""
Drift Auditor - Omission Drift Diagnostic Tool
================================================
Multi-turn drift detection for LLM conversations.
10-Tag Taxonomy | 12-Rule Operator System | 20+ Detection Methods

Detects instructions that a language model silently stops following.
Not hallucination. Not sycophancy. Omission: the model received an
instruction, followed it initially, then quietly dropped it.

Detection layers:
  Layer 1 - Commission: Sycophancy, reality distortion, false equivalence
  Layer 2 - Omission: Instruction adherence, contrastive drift, voids
  Layer 3 - Correction Persistence: Acknowledged fixes that fail
  Layer 4 - Structural Barometer: Epistemic posture (RED/YELLOW/GREEN)
  + Conflict pairs, shadow patterns, operator moves, pre-drift signals,
    criteria lock, task wall, bootloader check, structured disobedience,
    judge mode violations, Rumsfeld classification, artificial sterility,
    Oracle counterfactual (PREVENTABLE/SYSTEMIC)

Iron Pipeline architecture with sliding windows prevents self-contamination.
Per-instruction lifecycle tracking with coupling scores.
Edge vs. middle positional analysis.

Author: George Abrahamyants
Built for Anthropic Claude Hackathon, February 2026
Built with Claude Code + Cursor (Opus 4.6)

Refactored into modular structure. This file is the thin orchestrator.
All detection logic, models, and utilities live in their own modules.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

# --- Models ---
from models import (
    DriftTag, OperatorRule,
    Instruction, DriftFlag, CorrectionEvent, BarometerSignal,
    InstructionLifecycle, ConflictPair, ShadowPattern, OpMove, VoidEvent,
    AuditReport,
)

# --- Parser ---
from parsers.chat_parser import parse_chat_log

# --- Detectors ---
from detectors.commission import (
    detect_commission,
    detect_false_equivalence,
    detect_fabrication_from_conflict,
    detect_judge_mode_violations,
    detect_artificial_sterility,
)
from detectors.omission import (
    extract_instructions,
    detect_omission_local,
    detect_omission_api,
    detect_contrastive_drift,
    detect_undeclared_unresolved,
    detect_criteria_lock,
    detect_task_wall_violations,
    detect_missing_bootloader,
)
from detectors.structural import (
    detect_barometer_signals,
    detect_correction_persistence,
    detect_operator_moves,
    detect_voids,
    detect_pre_drift_signals,
    detect_conflict_pairs,
    detect_shadow_patterns,
)

# --- Utilities ---
from utils import (
    compute_coupling_score,
    coupling_label,
    analyze_positional_omission,
    classify_preventable_vs_systemic,
    classify_instruction_uncertainty,
    build_instruction_lifecycles,
    compute_scores,
    format_report,
    report_to_json,
)


# ---------------------------------------------------------------------------
# Taxonomy Configuration
# ---------------------------------------------------------------------------

_TAXONOMY = {}

def _load_taxonomy() -> dict:
    """Load drift taxonomy from config/taxonomy.yaml if available."""
    global _TAXONOMY
    if _TAXONOMY:
        return _TAXONOMY
    if yaml is None:
        return {}
    config_path = Path(__file__).resolve().parent.parent / "config" / "taxonomy.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            _TAXONOMY = yaml.safe_load(f) or {}
    return _TAXONOMY


def get_taxonomy() -> dict:
    """Public accessor for the loaded taxonomy configuration."""
    return _load_taxonomy()


# ---------------------------------------------------------------------------
# Sliding Window Orchestrator
# ---------------------------------------------------------------------------

def audit_conversation(
    raw_text: str,
    system_prompt: str = "",
    user_preferences: str = "",
    window_size: int = 50,
    overlap: int = 10,
    conversation_id: str = "unknown"
) -> AuditReport:
    """
    Full audit pipeline with sliding window for long conversations.

    Pipeline:
      1. Parse the chat log
      2. Extract instruction set (baseline) + assign IDs + coupling scores
      3. Layer 4: Barometer (runs first — feeds Layer 2)
      4. Layer 1: Commission per window
      5. Layer 2: Omission per window (with barometer cross-layer)
      6. Layer 3: Correction persistence (full conversation)
      7. NEW: Operator move detection (Tag 9 / 12-rule system)
      8. NEW: Conflict pair detection (Tag 7)
      9. NEW: Shadow pattern detection (Tag 8)
     10. NEW: Contrastive anchoring (6a)
     11. NEW: Void detection (6b)
     12. NEW: Undeclared unresolved (6c)
     13. NEW: False equivalence detection (6e)
     14. NEW: Pre-drift signal detection (6f)
     15. NEW: Instruction lifecycle tracking
     16. NEW: Edge vs. middle positional analysis (6d)
     17. Aggregate and score

    Window size of 50 turns with 10-turn overlap prevents the auditor
    from accumulating enough context to drift on the audit itself.
    """
    # Parse
    turns = parse_chat_log(raw_text)
    if not turns:
        return AuditReport(
            conversation_id=conversation_id,
            total_turns=0,
            instructions_extracted=0,
            metadata={"error": "No turns parsed from input"}
        )

    # Extract instructions (once — this is the baseline)
    instructions = extract_instructions(turns, system_prompt, user_preferences)

    # Assign instruction IDs
    for idx, inst in enumerate(instructions):
        inst.instruction_id = f"inst_{idx:03d}"

    # Initialize report
    report = AuditReport(
        conversation_id=conversation_id,
        total_turns=len(turns),
        instructions_extracted=len(instructions)
    )

    # Validate window parameters
    if window_size <= 0:
        window_size = 50
    if overlap >= window_size:
        overlap = max(0, window_size // 5)

    # Layer 4: Full conversation barometer (run first — feeds into Layer 2)
    report.barometer_signals = detect_barometer_signals(turns)

    # Sliding window audit for Layers 1 and 2
    seen_flags = set()  # Dedup key: (layer, turn, description)
    start = 0
    while start < len(turns):
        end = min(start + window_size, len(turns))
        window = turns[start:end]

        # Layer 1: Commission detection
        commission_flags = detect_commission(window)
        for f in commission_flags:
            f.tag = f.tag or DriftTag.SYCOPHANCY.value
            dedup_key = (f.layer, f.turn, f.description)
            if dedup_key not in seen_flags:
                seen_flags.add(dedup_key)
                report.commission_flags.append(f)

        # Layer 2: Instruction adherence check (with barometer signals)
        active_instructions = [
            inst for inst in instructions if inst.active
        ]
        # Filter barometer signals to current window for cross-layer boost
        window_barometer = [
            s for s in report.barometer_signals
            if start <= s.turn < end
        ]
        omission_flags = detect_omission_local(window, active_instructions, window_barometer)
        for f in omission_flags:
            f.tag = f.tag or DriftTag.INSTRUCTION_DROP.value
            dedup_key = (f.layer, f.turn, f.description)
            if dedup_key not in seen_flags:
                seen_flags.add(dedup_key)
                report.omission_flags.append(f)

        # Advance window
        start += window_size - overlap

    # Layer 3: Correction persistence (needs full conversation view)
    correction_events = detect_correction_persistence(turns)
    for event in correction_events:
        if not event.held:
            event.tag = DriftTag.CORRECTION_DECAY.value
    report.correction_events = correction_events

    # --- NEW DETECTION SYSTEMS ---

    # Tag 9: Operator move detection (12-rule system)
    report.op_moves = detect_operator_moves(turns)

    # Tag 7: Conflict pair detection
    report.conflict_pairs = detect_conflict_pairs(turns)

    # Tag 8: Shadow pattern detection
    report.shadow_patterns = detect_shadow_patterns(turns, instructions)

    # 6a: Contrastive anchoring
    contrastive_flags = detect_contrastive_drift(turns, instructions)
    for f in contrastive_flags:
        report.omission_flags.append(f)

    # 6b: Void detection
    report.void_events = detect_voids(turns, instructions)

    # 6c: Undeclared unresolved
    unresolved_flags = detect_undeclared_unresolved(turns)
    for f in unresolved_flags:
        report.omission_flags.append(f)

    # 6e: False equivalence detection
    equivalence_flags = detect_false_equivalence(turns)
    for f in equivalence_flags:
        report.commission_flags.append(f)

    # 6f: Pre-drift signal detection
    report.pre_drift_signals = detect_pre_drift_signals(turns)

    # --- 12-RULES DETECTION METHODS ---

    # Criteria Lock (Rule 1 derivative)
    criteria_lock_flags = detect_criteria_lock(turns)
    for f in criteria_lock_flags:
        report.omission_flags.append(f)

    # Task Wall (Chapter 2 derivative)
    task_wall_flags = detect_task_wall_violations(turns)
    for f in task_wall_flags:
        report.omission_flags.append(f)

    # Bootloader Check
    bootloader_flags = detect_missing_bootloader(turns, system_prompt, user_preferences)
    for f in bootloader_flags:
        report.omission_flags.append(f)

    # Structured Disobedience (Chapter 2 derivative)
    disobedience_flags = detect_fabrication_from_conflict(turns)
    for f in disobedience_flags:
        report.commission_flags.append(f)

    # Judge Mode Violation (Rule 3 derivative)
    judge_mode_flags = detect_judge_mode_violations(turns)
    for f in judge_mode_flags:
        report.commission_flags.append(f)

    # Rumsfeld Classification (Rule 2)
    rumsfeld = classify_instruction_uncertainty(instructions)

    # Artificial Sterility (AEGIS Layer 7)
    all_flags = report.commission_flags + report.omission_flags + report.pre_drift_signals
    sterility_flags = detect_artificial_sterility(turns, all_flags)
    for f in sterility_flags:
        report.commission_flags.append(f)

    # Oracle Counterfactual (AEGIS Layer 9) — classify all flags
    for f in report.commission_flags + report.omission_flags:
        if not hasattr(f, 'counterfactual') or True:
            f.counterfactual = classify_preventable_vs_systemic(f, instructions)

    # Instruction lifecycle tracking
    report.instruction_lifecycles = build_instruction_lifecycles(
        turns, instructions, len(turns)
    )

    # 6d: Edge vs. middle positional analysis
    report.positional_analysis = analyze_positional_omission(
        instructions, report.instruction_lifecycles, len(turns)
    )

    # Assign coupling scores to all flags
    coupling_map = {lc.instruction_text: lc.coupling_score
                    for lc in report.instruction_lifecycles}
    for f in report.commission_flags + report.omission_flags:
        if f.instruction_ref and f.instruction_ref in coupling_map:
            f.coupling_score = coupling_map[f.instruction_ref]

    # Score
    report.summary_scores = compute_scores(report)

    # Metadata
    report.metadata = {
        "audit_timestamp": datetime.now().isoformat(),
        "window_size": window_size,
        "overlap": overlap,
        "parser_format": "auto-detected",
        "instructions_by_source": {
            "system_prompt": len([i for i in instructions if i.source == "system_prompt"]),
            "user_preference": len([i for i in instructions if i.source == "user_preference"]),
            "in_conversation": len([i for i in instructions if i.source == "in_conversation"]),
        },
        "detection_systems": {
            "tags_active": [t.value for t in DriftTag],
            "rules_active": [r.value for r in OperatorRule],
            "layers": ["commission", "omission", "correction_persistence", "barometer",
                        "contrastive", "void", "undeclared_unresolved",
                        "false_equivalence", "pre_drift", "conflict_pair",
                        "shadow_pattern", "op_move", "criteria_lock",
                        "task_wall", "bootloader", "structured_disobedience",
                        "judge_mode", "rumsfeld"],
        },
        "rumsfeld_classification": rumsfeld,
    }

    return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    """Run audit from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Drift Auditor - Multi-turn drift diagnostic tool"
    )
    parser.add_argument("chat_file", help="Path to chat transcript (JSON or plain text)")
    parser.add_argument("--system-prompt", help="File containing system prompt")
    parser.add_argument("--preferences", help="File containing user preferences")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--id", default=None, help="Conversation identifier")
    parser.add_argument("--window", type=int, default=50,
                        help="Sliding window size in turns (default: 50)")
    parser.add_argument("--overlap", type=int, default=10,
                        help="Window overlap in turns (default: 10)")

    args = parser.parse_args()

    conv_id = args.id or os.path.basename(args.chat_file)

    system_prompt = ""
    if args.system_prompt:
        with open(args.system_prompt, 'r', encoding='utf-8') as f:
            system_prompt = f.read()

    user_preferences = ""
    if args.preferences:
        with open(args.preferences, 'r', encoding='utf-8') as f:
            user_preferences = f.read()

    # Read chat file
    with open(args.chat_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Run audit
    report = audit_conversation(
        raw_text=raw_text,
        system_prompt=system_prompt,
        user_preferences=user_preferences,
        window_size=args.window,
        overlap=args.overlap,
        conversation_id=conv_id
    )

    # Output
    if args.json:
        print(report_to_json(report))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
