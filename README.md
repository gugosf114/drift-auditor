# Drift Auditor

**Omission Drift Diagnostic Tool for LLM Conversations**

Detects instructions that a language model silently stops following during a conversation. Not hallucination (saying wrong things). Not sycophancy (saying agreeable things). Omission: the model received an instruction, followed it initially, then quietly dropped it without acknowledgment.

This is the behavior Anthropic's own disempowerment paper (January 28, 2026) identified as undetectable by current safeguards, which "operate primarily at individual exchange level" and "may miss behaviors that emerge across exchanges and over time."

**Live demo:** [drift-auditor.streamlit.app](https://drift-auditor-3vghngnafvgkpqwdditewy.streamlit.app/)

## The Gap This Fills

Single-turn evaluation sees a successful correction. Multi-turn analysis reveals the apology changed nothing. This tool adds detection for the multi-turn patterns current safeguards can't see.

## Operator Load — The Missing Eval Category

Every current eval asks "did the model behave?" Nobody asks **"how hard did the human have to work to make the model behave?"**

A model that scores clean because the operator corrected it 741 times isn't safe — it's high-maintenance. A model that scores clean with 10 operator interventions is actually safe.

**Tested on 512 real conversations (37,536 messages) across Claude and ChatGPT:**

| Metric | Claude | ChatGPT |
|--------|--------|---------|
| Conversations analyzed | 187 | 325 |
| Average drift score | 2.9/10 | 3.0/10 |
| Operator Load Index | 0.082 | 0.056 |
| Alignment Tax | 16.3% | 11.2% |
| Self-Sufficiency Score | 59.2% | 72.1% |
| Instruction Survival Rate | 56.5% | 45.3% |
| Correction Efficiency | 98.7% | 100.0% |

**The finding:** Claude and ChatGPT tie on drift score — but fail differently. Claude is high-maintenance, high-fidelity (needs more babysitting but retains more instructions). ChatGPT is low-maintenance, silent decay (runs quieter but silently drops 55% of instructions). One fails loudly. The other fails quietly. The quiet one is more dangerous.

## Detection Architecture

### 10-Tag Taxonomy
Derived from operator research across 250+ adversarial conversations with Claude, GPT, and Gemini:

| Tag | Category | What It Detects |
|-----|----------|----------------|
| SYCOPHANCY | Commission | Unsolicited praise, invented agreement |
| REALITY_DISTORT | Commission | False references, hallucinated context |
| CONF_INFLATE | Commission | Unwarranted certainty without evidence |
| INSTR_DROP | Omission | Silently stops following an instruction |
| SEM_DILUTE | Omission | Follows instruction in letter, not spirit |
| CORR_DECAY | Persistence | Acknowledged correction doesn't persist |
| CONFLICT_PAIR | Structural | Automatable contradiction detection |
| SHADOW_PATTERN | Structural | Emergent model behavior not prompted |
| OP_MOVE | Operator | Audit of human's steering action |
| VOID_DETECTED | Meta | Break in causal chain |

### 12-Rule Operator System
Classifies the human's corrective actions — no existing tool does this:

| Rule | Name | What the Operator Did |
|------|------|-----------------------|
| R01 | Anchor | Set explicit instruction at start |
| R02 | Echo Check | Ask model to restate instructions |
| R03 | Boundary | Enforce scope limits (most frequently violated) |
| R04 | Correction | Direct error correction |
| R05 | Not-Shot | Catch voice transcription / typo errors |
| R06 | Contrastive | "What changed between X and Y?" |
| R07 | Reset | Start over / full context reset |
| R08 | Decompose | Break complex instruction into steps |
| R09 | Evidence Demand | "Show me where" / proof request |
| R10 | Meta Call | Call out the drift pattern by name |
| R11 | Tiger Tamer | Active reinforcement — keep correcting |
| R12 | Kill Switch | Abandon thread / hard stop |

### 20+ Detection Methods

**Core Layers:**
- Layer 1: Commission Detection (sycophancy, reality distortion, context gates)
- Layer 2: Omission Detection (keyword + barometer-assisted + API semantic)
- Layer 3: Correction Persistence (did acknowledged fixes hold?)
- Layer 4: Structural Drift Barometer (RED/YELLOW/GREEN epistemic posture)

**Advanced Detection:**
- Contrastive Anchoring — diff early vs late instruction presence
- Void Detection — Given → Acknowledged → Followed → Persisted chain
- Undeclared Unresolved — topics introduced but never addressed
- Edge vs Middle — positional omission rate analysis
- False Equivalence — same word, shifted meaning across turns
- Pre-Drift Signals — 4 early warning indicators before drift occurs
- Conflict Pair Detection — automated contradiction finding
- Shadow Pattern Detection — unprompted recurring behaviors
- Criteria Lock — model curates instead of extracting exhaustively
- Task Wall — context fragmentation between interleaved tasks
- Bootloader Check — conversation starts without constraints
- Structured Disobedience — fabrication from conflicting constraints
- Judge Mode Violation — model prescribes before operator states position
- Rumsfeld Classification — known/unknown/unknowable instruction classification
- Artificial Sterility — suspiciously clean conversations
- Oracle Counterfactual — PREVENTABLE vs SYSTEMIC classification

**Per-Instruction Tracking:**
- Coupling Score (HIGH/MEDIUM/LOW) — "If this instruction vanished, would downstream decisions change?"
- Lifecycle: turn given → turn last followed → turn first omitted
- Positional analysis: edge start vs middle vs edge end omission rates

### Adversarial Test Generator

Automated model-evaluating-model pipeline. The tool generates test conversations by giving one model a system prompt with specific instructions, then has a second model play the user pushing on those instructions for 50+ turns. Fully automated. Model evaluating model.

**5 pre-built scenarios:**

| Scenario | What It Tests | First Result |
|----------|--------------|-------------|
| `hedging_persistence` | Can the model stay direct for 60 turns? | Sonnet: 6/10 drift in 19 turns |
| `citation_decay` | Do citations persist as topics get harder? | — |
| `format_compliance` | Does structured output survive casual conversation? | — |
| `boundary_respect` | Does scope creep happen over 50 turns? | — |
| `correction_persistence` | Do corrections hold after being acknowledged? | — |

## Install

```
git clone https://github.com/gugosf114/drift-auditor.git
cd drift-auditor

# No dependencies needed for CLI mode.
# For dashboard:
pip install streamlit plotly

# For API-powered detection and adversarial testing:
pip install anthropic
export ANTHROPIC_API_KEY=your-key-here
```

## Usage

```bash
# Basic audit (local heuristics)
python src/drift_auditor.py conversation.txt

# With system prompt and user preferences
python src/drift_auditor.py chat.txt --system-prompt system.txt --preferences prefs.txt

# JSON output
python src/drift_auditor.py chat.txt --json

# Streamlit dashboard
streamlit run app.py

# Batch audit — Claude data export
python batch_audit.py path/to/conversations.json

# Batch audit — ChatGPT data export
python batch_audit_chatgpt.py path/to/conversations.json

# Operator Load comparison
python -c "from src.operator_load import run_comparison; run_comparison('batch_results/batch_results.json', 'batch_results_chatgpt/batch_results_chatgpt.json')"

# Adversarial test (requires API key)
python adversarial_test.py hedging_persistence --target-model claude-sonnet-4-20250514
```

### Supported Input Formats
- **Claude.ai JSON export** (list of messages or `chat_messages` wrapper)
- **Claude app copy-paste** (date-separated format from desktop/mobile app)
- **ChatGPT data export** (nested mapping format)
- **Plain text** with role markers (`Human:` / `Assistant:`, `User:` / `Claude:`, etc.)
- **Custom JSON** with `role`/`content` or `sender`/`text` fields

## The Dataset

- **512 conversations** audited across Claude (187) and ChatGPT (325)
- **37,536 messages** processed with zero errors
- **250+ adversarial conversations** across Claude, GPT, and Gemini (research corpus)
- **22+ manually labeled** omission drift instances
- **80+ verbatim exchange examples** tagged with operator rules
- Cross-model validated: same detection framework, same operator, different models

## Research Lineage

This tool is the endpoint of 8 documented research stages:

1. Better prompts → 2. Meta-rules → 3. Sandboxes → 4. NotebookLM → 5. Mid-chat debugging → 6. Binary detection → 7. None foolproof → 8. Air-gapped architecture

Each stage has documented failure modes. The tool exists because every simpler approach was tried and failed.

Source research: "12 Rules for AI: An Operator's Field Manual" (29 pages, 17 academic references), AEGIS 10-Layer Architecture, Iron Pipeline v2.6, Mid-Chat Drift Barometer.

## Competitive Advantage

1. **Operator Load** — the eval category that doesn't exist yet: human cost of alignment, measured from real data
2. **The dataset nobody else has** — 512 real conversations across two models, not synthetic benchmarks
3. **Cross-model comparison** — same operator, same framework, different failure modes
4. **8-stage research arc** — every simpler approach documented as failed
5. **Directly addresses Anthropic's stated gap** — their disempowerment paper says current safeguards miss multi-turn patterns
6. **Model-agnostic** — works on Claude, GPT, Gemini
7. **Dual-lens audit** — audits both the model AND the operator (unique)
8. **Adversarial test generator** — automated model-vs-model drift evaluation
9. **Code, not prompts** — Python agent executes detection, not copy-paste prompt kit

## Known Limitations

- Layer 2 local mode is keyword matching — misses semantic compliance. Use `--api` for real omission detection.
- Layer 4 barometer patterns are heuristic surface markers, not actual model uncertainty.
- Scoring weights are heuristic, not empirically calibrated.
- Operator Load comparison is not normalized by task complexity (requires manual annotation).
- Adversarial test generator requires Anthropic API key and is subject to rate limits.
- Parser handles common formats but may need extension for unusual transcript structures.

## Built For

Anthropic Claude Hackathon (February 10–16, 2026)

**Author**: George Abrahamyants
**Built with**: Claude Code + Cursor (Opus 4.6)
