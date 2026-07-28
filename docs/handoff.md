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
**Mechanism:** live sufficiency — instrument **while** data is applied; decide **how much is enough** / **when to shift gears**. See `docs/product.md`.  
**Differentiator:** recipes + meta-recipes + **personal recipe book** (`docs/recipes.md`).

**Prioritize:** Expert-v1 / P.Recipes (Expert-v0 **closed**).

## Where we are (2026-07-28)

| Done | Open (next) |
|------|-------------|
| Expert-v0/v1 observe + southward auto-stop | meta-exec default live SFT/GRPO runners |
| LocalBackend real DPO + DPO probes | Bridge/OXE scale ladder (Expert-v2) |
| Book/meta HTTP+MCP+harness surfaces | Throughput defaults / multi-hour lab smokes |
| `run_vlm_queue` vision stage advance | Org packs + self-host notes |
| J-lens shelved (code kept; tests optional `-m jlens`) | |
| **Checkpoint + resume** SFT/GRPO (`run_dir/resume.json` + `save_state`) | DPO recipe resume if needed |

**Habit:** `python scripts/lab_smokes.py --profile quick` often.  
**Resume:** `run_sft` / `run_grpo` with `run_dir` + `checkpoint_every=N`, then `resume=True` (same total `steps`).  
**J-lens:** `pytest -m jlens` or `pytest tests/jlens/ -o addopts=` (not default CI).

**Recommended next Outcome:**  
`meta-exec default SFT/GRPO runners` or Expert-v2 scale ladder

## Pull-on-miss only

| Topic | Where |
|-------|--------|
| Product / purpose | `docs/product.md` |
| Recipes / personal book | `docs/recipes.md` · roadmap **P.Recipes** |
| Expert-v0 checklist | `docs/expert-v0-smoke.md` |
| Expert ladder + archive | `docs/roadmap.md` |
| Primary workflow | `docs/design.md` §1.0 |
| Agent operator brief | `docs/agent-context.md` |
| PR process | `docs/development-process.md` |

## Lab facts (no secrets in commits)

- Weights/data: forge `/mnt/data/models`, `/mnt/data/anvil-observe`, `/mnt/data/anvil-runs`  
- Recipe book lab: `/mnt/data/anvil-recipes` (`ANVIL_RECIPE_BOOK`)  
- Default VLM: `Qwen2.5-VL-3B-Instruct`  
- Never commit weights, datasets, or private host dumps  

## Git / PR (Mia-aligned)

- Never push to `main`. Branch → `gh pr create` → **Andrew merges**.  

## Done looks like

PR against **Expert-v1 / P.Recipes** checkbox, tests green, ready for human merge.
