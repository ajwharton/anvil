# Handoff — spin off a work session

Use this when starting a **new agent session** dedicated to Anvil (separate from aiops / mia-rl Ship).

## Bootstrap

1. Clone or open this repo as the workspace root.  
2. Tell the agent: **`read start.md`**.  
3. One line: **`Outcome: …`** (e.g. Phase 3 forge VLM smoke, or MCP desktop config).  
4. Optional: `Track: anvil`.

Do **not** preload mia-rl scoreboard, coach ontology, or full dual-Spark ops history.

## Product dual focus (keep in mind)

| Mode | Job |
|------|-----|
| **Individual** | Recipes, four verbs, web UI, observe cliffs |
| **Agentic** | Same SSOT via API/MCP; Anvil owns tools/harness/prompts; **user brings the brain** |

Thesis: `docs/product.md` · control split: `docs/agentic-control.md` · prompts: `prompts/agent/`.

## Context you may need later (pull-on-miss)

| Topic | Where |
|-------|--------|
| Product / agent thesis | `docs/product.md`, `docs/agentic-control.md` |
| Full design | `docs/design.md` |
| Phases | `docs/roadmap.md` |
| Vision / robotics data | `docs/phase3-vision.md`, `docs/datasets-robotics.md` |
| Rules | `docs/governance.md` |
| MCP / harness code | `anvil/agent/` |
| Local DS4 serve (personal lab) | private `aiops` — not required for Anvil OSS |
| Coach Ship | private `mia-rl` — out of scope here |

## Good first Outcomes

- Phase 3: forge `vlm_smoke` on Qwen2.5-VL + real frames  
- Agent: document Cursor/Claude Desktop MCP config for `anvil mcp`  
- Recipe-graph doc (“if cliff X → try Y”)  
- Live method-switch tooling beyond knobs patch  
- Bridge/LeRobot → JSONL convertor smoke  

## Done looks like

PR **opened** (not agent-merged) against an exit criterion in `docs/roadmap.md` or a product/docs gate in `docs/product.md` / `docs/agentic-control.md`, ready for Andrew to review and merge — or a clear blocked note with next decision.

## Git / PR (Mia-aligned)

- Never push to `main`.  
- `feat/…` or `docs/…` branch → `gh pr create` → stop; human merges.  
- Details: `docs/development-process.md`.
