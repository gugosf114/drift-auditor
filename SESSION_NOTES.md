# Session Notes

## Session 1 (Feb 8-9, 2026)

Built: Full drift auditor — 20+ detection methods, 10-tag taxonomy, 12-rule operator system, per-instruction lifecycle tracking, coupling scores, batch runners (Claude + ChatGPT), adversarial test generator (5 scenarios), cross-model leaderboard, Operator Load eval (the new metric category).

Deployed: Streamlit Cloud at https://drift-auditor-3vghngnafvgkpqwdditewy.streamlit.app/

Tested: 512 conversations (187 Claude + 325 ChatGPT), 37,536 messages. Cross-model comparison done. Operator Load comparison done.

Key finding: Claude and ChatGPT tie on drift score (2.9 vs 3.0) but diverge on failure mode. Claude = high-maintenance high-fidelity (0.082 OLI, 56.5% instruction survival). ChatGPT = low-maintenance silent decay (0.056 OLI, 45.3% instruction survival). One fails loudly, the other fails quietly.

First adversarial test: Sonnet scored 6/10 drift in 19 turns on hedging persistence. Auto-generated contradiction at turn 10.

GitHub: 22 commits on gugosf114/drift-auditor.

## Next Session Priorities
1. Verify Streamlit deployment shows updated UI + leaderboard
2. Run more adversarial test scenarios (API key in env var, anthropic package installed)
3. Add Operator Load metrics to the Streamlit dashboard
4. Polish UI and documentation

## Context Files
- Brain farts: C:\Users\georg\Desktop\Claude_Context\
- Claude export: C:\Users\georg\Desktop\Claude_Context\Data Export 2 8 26\
- ChatGPT export: C:\Users\georg\Desktop\Claude_Context\conversations.json
- Dossier: c:\Users\georg\Downloads\george_dossier (1).md
- Cursor rules auto-load every chat (no paste needed)
- Repo: C:\Users\georg\Documents\GitHub\drift-auditor
