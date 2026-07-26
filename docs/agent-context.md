# Agent context — how to operate Anvil

**Audience:** any agent (or human) that will **plan, start, watch, and adjust** post-training runs.  
**Load this once** at session start for agent control. Prefer tools/HTTP over guessing.  
**Related:** [`agentic-control.md`](agentic-control.md) (who owns what) · [`product.md`](product.md) (why) · [`prompts/agent/`](../prompts/agent/) (portable prompts)

**Purpose (locked one-liner):** Anvil forges **sovereign domain experts** from base models.  
You operate the forge: place data, run methods, watch live signal, shift gears, export adapters.  
You do **not** invent a second train stack or scrape the HTML UI.

---

## Session bootstrap (agent)

1. Confirm control plane: `anvil_health` / `GET /api/health` (default `http://127.0.0.1:7600`).  
2. Load policy: human red lines + [`prompts/agent/safety_policy.md`](../prompts/agent/safety_policy.md).  
3. Load habit: [`prompts/agent/watch_loop.md`](../prompts/agent/watch_loop.md) + this file’s metric tables.  
4. State one **Outcome** (what success looks like for *this* run or session).  
5. Prefer **recipes + gates** over inventing knobs.  
6. Act only via **MCP / HTTP / audited tools**. Never invent metrics you did not read.

If a tool is missing, say so and fall back to documented human paths (`anvil-web`, CLI scripts). Do not scrape `/observe` HTML.

---

## Surfaces (SSOT)

| Surface | What it is | Where |
|---------|------------|--------|
| **Control plane** | Runs, knobs, pause/resume/patch, audit | `anvil-web` HTTP · MCP `anvil_*` · `AnvilControlClient` |
| **Observe plane** | Live train curves + probes | `ANVIL_OBSERVE_ROOT/<run_id>/metrics.jsonl` (+ `probes.jsonl`) · `/api/observe/*` · MCP `anvil_observe_*` |
| **Recipes / gates** | Architecture-aware plans | `/api/recipes`, `/api/plan` · `anvil_list_recipes` / `anvil_plan` / `anvil_suggest` |
| **Data** | Examples, trajectories, `cas://` media | `anvil.data.*`, convert CLI, media store — **not** git blobs |
| **Prompts** | Operator brain text | `prompts/agent/*` |

**Two run IDs (do not confuse them):**

| Kind | ID space | Listed by |
|------|----------|-----------|
| Control-plane run | web store id | `anvil_list_runs` / `GET /api/runs` |
| Observe run | directory name under `ANVIL_OBSERVE_ROOT` | `anvil_observe_list` / `GET /api/observe` |

Productized GRPO / VLM SFT with `run_dir=…` write **observe** runs. Control-plane train steps may be separate (toy web path). Tail **observe** metrics for live sufficiency.

---

## MCP tools (v0 names)

Require control plane up: `anvil mcp --url http://127.0.0.1:7600` (or harness).

| Family | Tools | Use for |
|--------|-------|---------|
| Discover | `anvil_health`, `anvil_overview`, `anvil_list_recipes`, `anvil_suggest` | Orientation |
| Plan | `anvil_plan` | RecipePlan + gates (`force` only under policy) |
| Control runs | `anvil_list_runs`, `anvil_get_run`, `anvil_create_run`, `anvil_train`, `anvil_pause`, `anvil_resume`, `anvil_patch_knobs`, `anvil_export`, `anvil_sample` | Lifecycle |
| Observe | `anvil_observe_list`, `anvil_observe_metrics`, `anvil_observe_probes` | Live sufficiency loop |
| Audit | `anvil_audit` | Gate overrides / decisions |

HTTP equivalents live under `/api/*` and `/api/observe/*` (same SSOT). Prefer JSON/SSE; never HTML scrape.

---

## Watch loop (every open train job)

Cycle (details in [`watch_loop.md`](../prompts/agent/watch_loop.md)):

1. **Read** — status, step, recipe/loss, last N metrics, last probes, tripwires.  
2. **Classify** — Healthy | Noisy | Cliff | Broken.  
3. **Act or wait** — one primary change at a time; fix infra before switching methods.  
4. **Log** — class, evidence (metric names + values), action or no-op.

**Scalars lie.** If reward/loss looks good but probe text is garbage, format-hacked, or off-task → treat as **Cliff**.

---

## What to watch by job type

Records are JSON lines in `metrics.jsonl` (`type: "step"` or `"event"`). GRPO and SFT share the file; use `job` when present (`grpo` | `sft` | `vlm_sft`).

### GRPO / on-policy RL

| Field | Meaning | Worry when |
|-------|---------|------------|
| `reward_mean` | Mean rollout reward this step | Flat forever after warm-up; or ↑ while probes ↓ |
| `group_reward_std_mean` | Within-group reward std (**advantage signal**) | ≈ 0 → **advantage collapse** (gradient dead) |
| `reward_std` | Across-batch reward spread | Extreme collapse or explosion |
| `loss` | Train loss (IS/PPO family) | NaN / non-finite |
| `is_mean_ratio` / `fb.mean_ratio` | Importance-sampling mean ratio | Far from ~1 for long → sample/train out of sync |
| `wall_time_s` | Step wall clock | Sudden multi-× slowdown (infra) |
| `adapter_synced` | Sample worker got a fresh adapter | Always false when sample worker expected |
| `type: event`, `event: early_stop` | Run abandoned on dead signal | Read `reason`; do not burn more steps |

**Probes (`probes.jsonl`):** `text`, `reward` under live policy. Read text, not only reward.

**Default acts:** early-stop already on in product GRPO when signal is dead; else pause → diversify data/reward → lower LR → short SFT recovery → resume or new stage (recipe queue).

### SFT / VLM SFT

| Field | Meaning | Worry when |
|-------|---------|------------|
| `loss` | CE (or recipe loss) | Non-finite; stuck high; ↓ while probes garbage |
| `n_image_refs` | Vision refs in batch | 0 on a VLM job that should see frames |
| `n_tokens` | Token count (when backend reports) | 0 / missing unexpectedly |
| `n_datums` | Batch size | 0 |
| `wall_time_s` | Step time | Pathological slowdown |
| `job` | `sft` or `vlm_sft` | — |

**Probes:** pass held-out Examples into `run_sft` / `run_vlm_sft` (`probes=`, `probe_every=`) → `probes.jsonl` with greedy `text`, optional `target` + match `reward`. See [`expert-v0-smoke.md`](expert-v0-smoke.md).

**Default acts:** stop or lower LR on probe garbage; check renderer train/sample consistency; freeze policy (vision encoder usually frozen); data labels / CAS refs.

### Preference (DPO-like) — when observe lands

Same watch habit; expect family-specific cliffs (proxy up, probes worse). Until DPO observe is first-class, treat missing metrics as **Broken/incomplete instrumentation**, not “healthy.”

---

## Classify → act (short)

| Class | Evidence (examples) | Prefer |
|-------|---------------------|--------|
| **Healthy** | Reward/probes improving or stable; group std > 0; IS ~1 | Wait; maybe denser probes |
| **Noisy** | High variance, short window | More steps/groups before judging |
| **Cliff** | Collapse, probe↓ reward↑, IS drift, SFT loss↓ probes garbage | [`method_switch.md`](../prompts/agent/method_switch.md) — **one** change |
| **Broken** | Tool errors, empty batch, missing observe dir, sampler dead | Pause/stop; fix infra; do not switch recipe to “heal” infra |

Method-switch edges (conservative): preference stall → try on-policy if verifiable reward exists; advantage collapse → pause RL / diversify / SFT recovery; IS drift → sync adapter more often; probe hack → stop optim, fix reward — full table in `method_switch.md`.

---

## Data placement (agent checklist)

You bring the corpus and rewards. Anvil places and instruments.

| Goal | Path |
|------|------|
| VLM / robot instruction rows | Convert → CAS + JSONL (`scripts/convert_robotics_corpus.py`, `anvil.data.convert`) then `examples_from_vlm_jsonl` / `run_vlm_sft` |
| Media | `LocalMediaStore` / `cas://` only — no multi‑MB blobs in plans or git |
| Live curves | Pass `run_dir` (under `ANVIL_OBSERVE_ROOT/<id>`) into `run_grpo` / `run_vlm_sft` / `run_sft` |
| Scale | Smoke few steps → then scale rows/steps; do not launch huge sweeps unasked |

---

## Hard rules (non-negotiable)

- Never invent metrics; only tool/artifact values.  
- Never force architecture gates unless policy allows; if force → reasons (audited).  
- Prefer small LoRA + documented losses over full-weight experiments.  
- No secrets, private keys, or private LAN inventories in commits or transcripts.  
- Human stop is absolute.  
- Do not commit datasets or weights.  
- Prefer one smoke before scaling.  
- **Brain is yours** — Anvil owns tools/harness/prompts, not the frontier model account.

---

## What “done” looks like (domain expert lens)

There is rarely a single deterministic stop. Prefer:

1. **Live probes** acceptable on held-out domain tasks.  
2. **No active cliff** (collapse, hack, IS desync).  
3. **Adapter export** (PEFT/path human expects).  
4. **Transcript** of why you stopped or switched (for the next agent/human).

If signals conflict, stop and report — do not burn the full budget hoping scalars recover.

---

## Pull-on-miss

| Need | Open |
|------|------|
| Why / dual focus / sufficiency | `docs/product.md` |
| Harness vs brain ownership | `docs/agentic-control.md` |
| Phase gates | `docs/roadmap.md` |
| Full API/backend design | `docs/design.md` |
| Prompt pack files | `prompts/agent/` |
| Vision / robotics data | `docs/phase3-vision.md`, `docs/datasets-robotics.md` |
| Code: MCP / client | `anvil/agent/` |
| Code: metrics writer | `anvil/observe/metrics.py` |

---

## Version note

When MCP tool names or metric field names change, update **this file and** `prompts/agent/*` in the **same PR** as the code change.
