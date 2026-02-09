# 12 Rules for AI — Mapping to Drift Auditor

## Source
"12 Rules for AI: An Operator's Field Manual" by George Abrahamyants
29 pages, 17 academic references, operator logs as evidence.

## Rule → Tool Mapping

| Rule | Name | Status in Tool | Implementation |
|------|------|---------------|----------------|
| Ch 1 | First Downfall (Authorship Transfer) | Partial | SHADOW_PATTERN detects model taking over |
| Ch 2 | Second Downfall (Collapse After Correct Process) | Partial | Correction Persistence (Layer 3) |
| R1 | Name the Disease (Behavioral Escalation Bias) | Implemented | Commission Detection (Layer 1) |
| R2 | Rumsfeld Protocol (Unknown Unknowns) | Not yet | Could classify instruction uncertainty |
| R3 | Operator Boundary | Implemented | OP_MOVE R03_BOUNDARY detection |
| R4 | Force the Contradiction | Implemented | CONFLICT_PAIR (Tag 7) |
| R5 | Not-Shot Rule | Implemented | OP_MOVE R05_NOT_SHOT detection |
| R6 | The Ledger | N/A | The tool IS the ledger (automated) |
| R7 | Never Let Model Architect Structure | Partial | Barbell Principle → Edge vs Middle |
| R8 | Sunday Rule (No Continuity) | N/A | Design principle, not detection target |
| R9 | Shadow Memory | Implemented | SHADOW_PATTERN (Tag 8) |
| R10 | Manufactured Agreement | Implemented | CONFLICT_PAIR + Commission Detection |
| R11 | Tiger Tamer | Implemented | OP_MOVE R11_TIGER_TAMER |
| R12 | Document Failures | Implemented | The entire tool automates this |

## Key Concepts from the Paper → Tool Features

| Paper Concept | Tool Feature |
|--------------|-------------|
| Confession Loop | Correction Persistence — model admits error then repeats it |
| Appeasement Loop | Commission Detection — sycophancy markers |
| Three-Strike Loop | Correction events with multiple failures |
| Task Wall | Not yet — detect context fragmentation |
| Criteria Lock | Not yet — detect model curating vs extracting |
| Bootloader | Not yet — flag conversations without constraints |
| Structured Disobedience | Not yet — detect fabrication from conflicting constraints |
| Judge Mode | Not yet — detect model speaking before operator |
| Barbell Principle | Edge vs Middle positional analysis |
| Cognitive Mirroring | Pre-drift signals — model approximating operator emotions |
| Newton Protocol (IRAC) | Void Detection — Given→Acknowledged→Followed→Persisted |

## Not-Yet-Implemented (Candidates for Next Build)

1. **Criteria Lock Detection** — flag when model returns curated subset vs full extraction
2. **Task Wall Detection** — flag context fragmentation between tasks
3. **Bootloader Check** — flag conversations starting without explicit constraints
4. **Structured Disobedience** — flag fabrication from conflicting constraints
5. **Judge Mode Violation** — flag model generating analysis before operator states position
6. **Rumsfeld Classification** — classify instructions by known/unknown/unknowable
