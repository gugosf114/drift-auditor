"""
Drift Auditor — Entry Point
============================
Thin dispatcher (~80 lines). All logic lives in ui/modes/.
"""
import os
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
from detectors.base import DetectorRegistry
import detectors.commission
import detectors.omission
import detectors.structural
from detectors.omission import extract_instructions
from detectors.frustration import compute_frustration_index, FrustrationResult

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
      5. Layer 2: Omission per window — local heuristics + API semantic detection
         (API detection runs when ANTHROPIC_API_KEY is set)
      6. Layer 3: Correction persistence (full conversation)
      7. Operator move detection (Tag 9 / 12-rule system)
      8. Conflict pair detection (Tag 7)
      9. Shadow pattern detection (Tag 8)
     10. Contrastive anchoring (6a)
     11. Void detection (6b)
     12. Undeclared unresolved (6c)
     13. False equivalence detection (6e)
     14. Pre-drift signal detection (6f)
     15. Instruction lifecycle tracking
     16. Edge vs. middle positional analysis (6d)
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

    # Initialize detectors
    window_detectors = DetectorRegistry.get_window_detectors()
    full_detectors = DetectorRegistry.get_full_detectors()

    # Run Barometer first as it feeds into Omission
    barometer_detector = next((d for d in full_detectors if type(d).__name__ == 'BarometerSignalsDetector'), None)
    if barometer_detector:
        report.add_flags(barometer_detector.detect(turns))
        full_detectors.remove(barometer_detector)

    # Sliding window audit for Layers 1 and 2
    seen_flags = set()  # Dedup key: (layer, turn, description)
    start = 0
    while start < len(turns):
        end = min(start + window_size, len(turns))
        window = turns[start:end]

        active_instructions = [inst for inst in instructions if inst.active]
        window_barometer = [s for s in report.barometer_signals if start <= s.turn < end]

        for detector in window_detectors:
            flags = detector.detect(
                window, 
                active_instructions=active_instructions, 
                window_barometer=window_barometer
            )
            # Dedup logic
            for f in flags:
                if hasattr(f, 'layer') and hasattr(f, 'turn') and hasattr(f, 'description'):
                    dedup_key = (f.layer, f.turn, f.description)
                    if dedup_key not in seen_flags:
                        seen_flags.add(dedup_key)
                        report.add_flags([f])
                else:
                    report.add_flags([f])

        # Advance window
        start += window_size - overlap

    # Run remaining full-conversation detectors
    for detector in full_detectors:
        if type(detector).__name__ == 'ArtificialSterilityDetector':
            all_flags = report.commission_flags + report.omission_flags + report.pre_drift_signals
            report.add_flags(detector.detect(turns, report_flags=all_flags))
        else:
            report.add_flags(detector.detect(
                turns, 
                instructions=instructions,
                system_prompt=system_prompt,
                user_preferences=user_preferences
            ))

    # Rumsfeld Classification (Rule 2)
    rumsfeld = classify_instruction_uncertainty(instructions)

    # Oracle Counterfactual (AEGIS Layer 9) — classify all flags
    for f in report.commission_flags + report.omission_flags:
        if not hasattr(f, 'counterfactual') or f.counterfactual is None:
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

# ---------------------------------------------------------------------------
# Batch Summary Writing
# ---------------------------------------------------------------------------

def write_batch_summary(results: list[dict], output_path: str, title: str):
    """Shared logic for writing batch summaries across different parsers."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{title}\n")
        # ... shared distribution and top 20 logic ...
