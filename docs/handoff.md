# Handoff — spin off a work session

Use this when starting a **new agent session** dedicated to Anvil (separate from aiops / mia-rl Ship).

## Bootstrap

1. Clone or open this repo as the workspace root.  
2. Tell the agent: **`read start.md`**.  
3. One line: **`Outcome: …`** (e.g. Phase 0 OpenAPI sketch + fake backend test).  
4. Optional: `Track: anvil`.

Do **not** preload mia-rl scoreboard, coach ontology, or full dual-Spark ops history.

## Context you may need later (pull-on-miss)

| Topic | Where |
|-------|--------|
| Full design | `docs/design.md` |
| Phases | `docs/roadmap.md` |
| Rules | `docs/governance.md` |
| Local DS4 serve (personal lab) | private `aiops` — not required for Anvil OSS |
| Coach Ship | private `mia-rl` — out of scope here |

## Good first Outcomes

- Sketch `anvil/protocol` types for the four verbs  
- Fake in-memory backend + one unit test for SFT step  
- CI: ruff + pytest on stubs  
- Recipe outline for VLM SFT (docs only)  

## Done looks like

PR merged against an exit criterion in `docs/roadmap.md`, or a clear blocked note with next decision.
