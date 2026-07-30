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

## Where we are (2026-07-29)

| Done | Open (next) |
|------|-------------|
| Expert-v0/v1 observe + southward + DPO | DPO recipe resume if needed |
| Meta-exec **default live runners** | Full DP multi-GPU **train** (not planned) |
| **Scale ladder** + `multi_hour` smokes | |
| **Forge VLM ≥1k** observe + export + ckpt | |
| **Org packs** + experience→patience | |
| **Multi-worker sample pool** (`sample_endpoints`) | |
| Checkpoint + resume SFT/GRPO/VLM | |
| J-lens shelved (optional `-m jlens`) | |

**Habit:** `python scripts/lab_smokes.py --profile quick` often.  
**Multi-worker sample:** `docs/multi-worker.md` · `run_grpo(sample_endpoints=[…])`  
**Org packs:** `ANVIL_ORG_RECIPE_PACK` · `docs/org-self-host.md`  

**Recommended next Outcome:**  
DPO resume parity, or forge dual-vLLM dogfood of sample pool

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
