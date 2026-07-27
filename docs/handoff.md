# Handoff — spin off a work session

Use this when starting a **new agent session** dedicated to Anvil (separate from aiops / mia-rl Ship).

## Bootstrap (thin)

1. Open this repo as the workspace root.  
2. **`read start.md`** once.  
3. One line: **`Outcome: …`** (optional `Track: anvil`).  
4. Act. **Do not** multi-doc boot; pull-on-miss only.

## Product dual focus

| Mode | Job |
|------|-----|
| **Individual** | Recipes, four verbs, web UI, observe cliffs |
| **Agentic** | Same SSOT via API/MCP; Anvil owns tools/harness/prompts; **user brings the brain** |

**Purpose:** Anvil forges **sovereign domain experts** from base models.  
**Mechanism:** live sufficiency — instrument **while** data is applied; decide **how much is enough** / **when to shift gears** before the model turns southward. See `docs/product.md`.

**Prioritize:** Expert-v0 → v1 → v2 in `docs/roadmap.md` (historical Phases 0–5 are archive).

## Where we are (2026-07-26)

| Done | Open (next) |
|------|-------------|
| Phases 0–2.5 foundation + Expert-v0 **path** (smoke + probes) | **Expert-v0 lab:** Bridge → 1k on forge |
| Convert + VLM metrics + held-out SFT/VLM probes | Expert-v1 method ladder |
| `docs/expert-v0-smoke.md` + `scripts/expert_v0_smoke.py` | Expert-v2 scale ops · Path: robot/edge |

**Recommended next Outcome:**  
`Expert-v0 lab — Bridge episode_pack → 1k + expert_v0_smoke on forge`  
(or: Expert-v1 preference observe)

## Pull-on-miss only

| Topic | Where |
|-------|--------|
| Product / purpose | `docs/product.md` |
| Expert-v0 checklist | `docs/expert-v0-smoke.md` |
| Expert-v0/v1/v2 + archive phases | `docs/roadmap.md` |
| Primary workflow | `docs/design.md` §1.0 |
| Agent operator brief | `docs/agent-context.md` |
| Vision slices | `docs/phase3-vision.md` |
| Robotics corpora | `docs/datasets-robotics.md` |
| Lab smokes / cron | `docs/lab-smokes.md`, `scripts/run_lab_smokes.sh` |
| Agent control ownership | `docs/agentic-control.md`, `anvil/agent/` |
| Design | `docs/design.md` |
| PR process | `docs/development-process.md` |

## Lab facts (no secrets in commits)

- Weights/data: forge `/mnt/data/models`, `/mnt/data/anvil-observe`, `/mnt/data/anvil-runs`  
- Default VLM: `Qwen2.5-VL-3B-Instruct`; text GRPO: `qwen2.5-1.5b-instruct`  
- Never commit weights, datasets, or private host dumps  

## Git / PR (Mia-aligned)

- Never push to `main`. Branch → `gh pr create` → **Andrew merges**.  
- Small PRs. Design/product thesis via `docs/` first when large.

## Done looks like

PR opened against an **Expert-v0/v1/v2** (or Path) checkbox, tests green, ready for human merge — or a clear blocked note.
