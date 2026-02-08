# Drift Auditor

Multi-turn drift diagnostic tool that complements Anthropic's Bloom/Petri evaluation framework by analyzing correction persistence and instruction adherence in real conversations — dimensions that single-turn evaluations miss.

## The Gap This Fills

Anthropic's disempowerment paper (January 28, 2026) found drift in 1 in 1,300 conversations but acknowledged their safeguards "operate primarily at the individual exchange level" and "may miss behaviors that emerge across exchanges and over time."

This tool adds detection for the multi-turn patterns they can't see.

## Three Detection Layers

### Layer 1: Commission Detection
Pattern matching for sycophancy, reality distortion, unwarranted confidence. Context gates suppress false positives when agreement appears in legitimate correction acknowledgments.

### Layer 2: Instruction Adherence Check (Local) / Omission Detection (API)
Local: keyword heuristic checking prohibition violations and required behavior absence.
API: sends each (instruction, response) pair to fresh Opus 4.6 context for semantic compliance evaluation.

### Layer 3: Correction Persistence
Tracks when a user corrects the model and the model acknowledges. Verifies the correction holds across subsequent turns. Uses topic signatures to only flag regression on the same corrected behavior. **This is the novel contribution.**

## Usage

```bash
python src/drift_auditor.py chat_transcript.txt
python src/drift_auditor.py chat.txt --system-prompt system.txt --preferences prefs.txt
python src/drift_auditor.py chat.txt --json --id "conversation_id"
```

## Known Limitations

- Layer 2 local version is keyword matching — misses semantic compliance
- Scoring weights are heuristic, not empirically calibrated
- Correction persistence tracks predefined types, not generalized taxonomy

## Built For

Built with Opus 4.6 Hackathon — complementing Anthropic's alignment infrastructure with multi-turn correction persistence analysis.
