# Drift Auditor — Project Context

## What This Tool Does
Detects instructions that a language model silently stops following during a conversation. Not hallucination. Not sycophancy. Omission: the model received an instruction, followed it initially, then quietly dropped it without acknowledgment.

This is the behavior Anthropic's disempowerment paper (Jan 28, 2026) identified as undetectable by current safeguards which "operate primarily at individual exchange level" and "may miss behaviors that emerge across exchanges and over time."

## Research Lineage (Stages 1-8)
Better prompts → meta-rules → sandboxes → NotebookLM → mid-chat debugging → binary detection → none foolproof → air-gapped architecture. Each stage has documented failure modes. The tool exists because every simpler approach was tried and failed.

## Architecture

### 10-Tag Detection Taxonomy
Tags 1-6 map to three detection layers. Tags 7-9 are net-new:
1. SYCOPHANCY — unsolicited praise, invented agreement
2. REALITY_DISTORT — false references, hallucinated context
3. CONF_INFLATE — unwarranted certainty without evidence
4. INSTR_DROP — silently stops following an instruction
5. SEM_DILUTE — follows instruction in letter, not spirit
6. CORR_DECAY — acknowledged correction doesn't persist
7. CONFLICT_PAIR — automatable contradiction detection (net-new)
8. SHADOW_PATTERN — emergent model behavior not prompted (net-new)
9. OP_MOVE — audit of human's steering action (net-new, unique)
10. VOID_DETECTED — break in causal chain

### 12-Rule Operator System
Classifies the HUMAN's corrective actions (no existing tool does this):
R01 Anchor | R02 Echo Check | R03 Boundary (most violated) | R04 Correction | R05 Not-Shot (voice errors) | R06 Contrastive | R07 Reset | R08 Decompose | R09 Evidence Demand | R10 Meta Call | R11 Tiger Tamer (most used correction) | R12 Kill Switch

### Detection Methods
- 6a. Contrastive Anchoring — diff early vs late instruction presence
- 6b. Void Detection (AEGIS Layer 7) — Given → Acknowledged → Followed → Persisted
- 6c. Undeclared Unresolved — threads that vanish without closure
- 6d. Edge vs Middle — positional omission rates
- 6e. False Equivalence (AEGIS Layer 8) — same word, shifted meaning
- 6f. Pre-Drift Signals — 4 indicators drift is about to occur

### Key Principles
- Air-Gap: auditor cannot see generator's context (prevents inherited drift)
- Iron Pipeline: sliding window prevents self-contamination
- Coupling Score: "If this instruction vanished, would downstream decisions change?"
- Per-instruction lifecycle: turn given → last followed → first omitted

## Dataset (not in repo)
- 250+ adversarial conversations across Claude, GPT, Gemini
- 22+ manually labeled omission drift instances
- 80+ verbatim exchange examples tagged with operator rules
- Cross-model: same protocol produces comparable results across all three

## Stack
- Python 3.14
- Streamlit (dashboard)
- Plotly (charts)
- Anthropic API (optional, for semantic omission detection)
- GitHub: github.com/gugosf114/drift-auditor

## Competitive Advantage
1. Dataset nobody else has (real adversarial conversations, not synthetic)
2. 8-stage research arc (every simpler approach documented as failed)
3. Directly addresses gap Anthropic identified in their own research
4. Model-agnostic (works on Claude, GPT, Gemini)
5. Dual-lens: audits both model AND operator (unique)
6. Code, not prompts (Newton Sentinel pattern)
