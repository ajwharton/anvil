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
**Mechanism:** live sufficiency — instrument **while** data is applied; decide **how much is enough** / **when to shift gears** before the model turns southward. Not fire-and-forget full budgets then eval. See `docs/product.md`.

## Where we are (2026-07-26)

| Done | Open (next) |
|------|-------------|
| Phases 0–2.5 (text SFT/RL, GRPO observe, early-stop, recipe queue, agent MCP v0) | **P.Sufficiency:** held-out probes + southward-turn for SFT/VLM |
| Phase 3.A vision core (renderer, pixel fusion, 3B forge smoke, LeRobot tiny) | **3.B lab:** extract Bridge → episode_pack → 1k `anvil_jsonl` on forge |
| **P3.6 metrics:** `run_vlm_sft(run_dir=)` → `/observe` loss curve | **3.C probes / early-stop** + **3.D** multi-hour |
| **P3.5 convert CLI:** `convert_robotics_corpus.py` (episode_pack → CAS/JSONL) | **Phase 4:** offline robot RL, action tokens, Jetson |
| Lab smokes `scripts/lab_smokes.py` (not GitHub CI) | |

**Recommended next Outcome:**  
`3.B lab — Bridge episode_pack → 1k rows + VLM SFT observe`  
(or: held-out frame probes during VLM SFT)

## Pull-on-miss only

| Topic | Where |
|-------|--------|
| Product / live sufficiency | `docs/product.md` |
| Phases + P.Sufficiency/P.Ops/P.Decide | `docs/roadmap.md` |
| Vision slices | `docs/phase3-vision.md` |
| Robotics corpora | `docs/datasets-robotics.md` |
| Lab smokes / cron | `docs/lab-smokes.md`, `scripts/run_lab_smokes.sh` |
| Agent control | `docs/agent-context.md` (session brief), `docs/agentic-control.md`, `anvil/agent/` |
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

PR opened against a **roadmap checkbox** (3.B / 3.C / 3.D / P.Sufficiency / Phase 4), tests green, ready for human merge — or a clear blocked note.
