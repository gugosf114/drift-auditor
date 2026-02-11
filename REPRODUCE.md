# Reproducing the Results

This document explains how to reproduce the comparative analysis presented in the research.

## Dataset

The analysis uses **512 real conversations** from personal use:

| Source | Conversations | Messages | Export Method |
|--------|--------------|----------|---------------|
| Claude (Sonnet 3.5/3.6) | 187 | 20,407 | Settings → Export Data → `conversations.json` |
| ChatGPT (GPT-4/4o) | 325 | 17,129 | Settings → Data Controls → Export → `conversations.json` |

**Privacy note:** The raw conversation data is not included in this repository because it contains personal conversations. To reproduce, use your own Claude and ChatGPT data exports.

## How to Export Your Data

### Claude
1. Go to [claude.ai](https://claude.ai) → Settings → Export Data
2. You'll receive an email with a download link
3. The export is a JSON file containing all conversations with `chat_messages` arrays

### ChatGPT
1. Go to [chatgpt.com](https://chatgpt.com) → Settings → Data Controls → Export
2. You'll receive an email with a download link
3. Extract the zip — the relevant file is `conversations.json` with nested mapping trees

## Running the Analysis

### Step 1: Install

```bash
pip install -e ".[dev]"
```

### Step 2: Batch Audit (Claude)

```bash
python batch_audit.py /path/to/claude/conversations.json
```

This processes every conversation with 10+ messages (configurable via `--min-messages`).
Results are saved to `batch_results/` as individual JSON reports plus a `_summary.json`.

### Step 3: Batch Audit (ChatGPT)

```bash
python batch_audit_chatgpt.py /path/to/chatgpt/conversations.json
```

Same process, adapted for ChatGPT's nested tree format.
Results are saved to `batch_results_chatgpt/`.

### Step 4: Operator Load Comparison

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from operator_load import run_comparison
run_comparison('batch_results', 'batch_results_chatgpt')
"
```

This reads the batch results from both directories and computes the comparative metrics:
- Operator Load Index
- Alignment Tax
- Self-Sufficiency Score
- Instruction Survival Rate
- Correction Efficiency

Output is written to `operator_load_comparison.txt`.

### Step 5: Adversarial Testing

```bash
export ANTHROPIC_API_KEY=your-key-here
python adversarial_test.py hedging_persistence --target-model claude-sonnet-4-20250514
python adversarial_test.py citation_decay --target-model claude-sonnet-4-20250514
python adversarial_test.py format_compliance --target-model claude-sonnet-4-20250514
python adversarial_test.py boundary_respect --target-model claude-sonnet-4-20250514
python adversarial_test.py correction_persistence --target-model claude-sonnet-4-20250514
```

Each run produces a transcript and audit report in `adversarial_results/`.

## Verifying Against Published Numbers

The key metrics from the analysis:

| Metric | Claude | ChatGPT |
|--------|--------|---------|
| Average drift score | 2.9 / 10 | 3.0 / 10 |
| Operator Load Index | 0.082 | 0.056 |
| Alignment Tax | 16.3% | 11.2% |
| Self-Sufficiency Score | 59.2% | 72.1% |
| Instruction Survival Rate | 56.5% | 45.3% |
| Correction Efficiency | 98.7% | 100.0% |

Your results will differ because you're using different conversations. The methodology is identical — the same detection functions, scoring formulas, and aggregation logic produce the numbers from whatever data you feed in.

## Metric Formulas

All computed in `src/operator_load.py`:

- **Operator Load Index** = `(total_corrections + total_op_moves) / total_messages`
- **Alignment Tax** = `min(1.0, (total_interventions × 2) / total_messages)`
- **Correction Efficiency** = `(corrections_held) / total_corrections`
- **Instruction Survival Rate** = `(total_instructions - omitted) / total_instructions`
- **Self-Sufficiency Score** = `max(0, (1.0 - OLI × 5) × 100)`

## Quick Verification (No Data Required)

To verify the pipeline works without any data export:

```bash
python src/drift_auditor.py examples/sample_conversation.txt
```

Or run the test suite:

```bash
python -m pytest tests/ -v
```
