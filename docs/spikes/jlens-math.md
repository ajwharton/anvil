# Spike: J-Lens on multi-step math (J0)

**Status:** **ran on forge — NO-GO** (2026-07-17)  
**Roadmap:** Phase 2.5 last open gate (panel still blocked)  
**Script:** [`scripts/jlens_spike.py`](../../scripts/jlens_spike.py)  
**Paper:** Gurnee, Sofroniew, Lindsey et al., *Verbalizable Representations Form a Global Workspace in Language Models* (Anthropic / Transformer Circuits, 2026-07-06)  
**Code:** [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) (Apache-2.0, reference impl)

## Goal

Before any permanent `/observe` J-Lens panel or `run_grpo` hook:

1. Fit a Jacobian lens on a **small dense** lab model (default Qwen2.5-1.5B-Instruct).  
2. Apply it on multi-step arithmetic prompts.  
3. Score whether **intermediate concepts light up in layer order** (operands / ops / partials → answer).  
4. Record a binary **GO / NO-GO** for product work (J1+).

Non-goals: web UI, per-token rollout instrumentation, vLLM hooks, consciousness claims.

## Lab layout

| Asset | Path |
|-------|------|
| Base model | `/mnt/data/models/qwen2.5-1.5b-instruct` (on-disk name; HF id `Qwen/Qwen2.5-1.5B-Instruct`) |
| Fitted lens | `/mnt/data/models/lenses/qwen2.5-1.5b-instruct/jacobian_lens.pt` |
| Fit meta | `/mnt/data/models/lenses/qwen2.5-1.5b-instruct/jacobian_lens.meta.json` |
| Run artifacts | `/mnt/data/anvil-runs/jlens-spike-20260717-191152/` |

Weights and lens checkpoints stay on forge NVMe — never commit them.

## Runbook (forge)

```bash
source /mnt/data/anvil-venv/bin/activate
pip install 'git+https://github.com/anthropics/jacobian-lens.git'   # once

cd /mnt/data/anvil && git checkout main && git pull
python scripts/jlens_spike.py check \
  --model-path /mnt/data/models/qwen2.5-1.5b-instruct

python scripts/jlens_spike.py all \
  --model-path /mnt/data/models/qwen2.5-1.5b-instruct \
  --device cuda \
  --fit-n 32 \
  --out /mnt/data/anvil-runs/jlens-spike-$(date +%Y%m%d-%H%M%S)

# exit 0 = gate GO, exit 2 = NO-GO with artifacts written
```

## Gate criteria (script-enforced)

| Signal | Threshold |
|--------|-----------|
| Mean `intermediate_order_score` over scored probes | **≥ 0.6** |
| Probes where answer appears in lens top-k at some layer | **≥ half** of probes |

`intermediate_order_score` = fraction of consecutive *hit* stages whose earliest lighting layer is non-decreasing (see script / `anvil.observe.jlens`). Missing stages are skipped (need ≥2 hits to score a probe).

**GO** → permanent panel + J2 apply hook are unblocked.  
**NO-GO** → stay CLI-only; J1 schema may remain for future records; do not build web panel.

---

## Results (2026-07-17 forge run)

| Field | Value |
|-------|--------|
| Date | 2026-07-17 |
| Host | **forge** (NVIDIA GB10, CUDA, torch 2.11.0+cu130, transformers 5.14.1) |
| Model path | `/mnt/data/models/qwen2.5-1.5b-instruct` |
| Lens path | `/mnt/data/models/lenses/qwen2.5-1.5b-instruct/jacobian_lens.pt` (~122 MB) |
| `fit_n` | **32** (smoke; paper uses larger corpus) |
| Fit wall time | **~1084 s** (~18 min) |
| mean order score | **`null`** (0 of 3 probes scored — no stage hits) |
| answer top-k hits | **0 / 3** |
| **Gate** | **NO-GO** |
| Script exit code | **2** |
| Artifact dir | `/mnt/data/anvil-runs/jlens-spike-20260717-191152/` |
| Env | `/mnt/data/anvil-venv` + `jlens` from `git+https://github.com/anthropics/jacobian-lens.git` |
| Anvil commit on forge | `cf68e9b` (main, post J1 merge) |

### Gate dump (from `jlens_spike_results.json`)

```json
{
  "go": false,
  "mean_intermediate_order_score": null,
  "n_probes": 3,
  "n_order_scored": 0,
  "n_answer_in_topk": 0,
  "decision": "NO-GO — keep CLI-only; try larger model or more fit data"
}
```

### Per-probe (protocol = last prompt token, top-k=8)

| Probe | order | stage_layers | answer_min_rank | late-layer tops (L26) |
|-------|-------|--------------|-----------------|------------------------|
| `add_then_mul` | null | all `None` | null | Sure / First / Okay / Step / Let / Alright |
| `sub_chain` | null | all `None` | null | Okay / Starting / Sure / Let / First / Alright |
| `double_plus` | null | all `None` | null | Sure / Okay / First / To / Step / Let |

Mid-layers (e.g. L9–L14) were dominated by unrelated tokens (`SELL`, `salary`, `chemistry`, `Combine`, …) — not operands / partials / answers.

### Notes / diagnosis

1. **Stack works.** Fit + save + apply + JSON/MD artifacts all completed without OOM or API crash. `jlens` installs cleanly on forge’s anvil-venv.  
2. **Protocol mismatch (likely).** Spike applies the lens only at **position −1 of the prompt**, before any completion. Late layers correctly light up *discourse openers* (“Sure”, “First”, “Okay”) — the model is poised to **start** a step-by-step reply, not to report intermediate arithmetic already computed. That is a useful negative result for product design: **last-prompt-only J-Lens is the wrong window for math-order debugging.**  
3. **Follow-up diagnostic (same lens, not in gate script):** a prompt that *already contains* CoT steps (`Step 1: 3 + 4 = 7` / `Step 2: 7 * 2 =`) produced some late-layer stage hits and answer-rank noise, with rough order ≈ 0.5 — still **below GO**, and digit matching can false-positive (e.g. `"1"` ⊆ `"14"`). Needs a dedicated protocol (mid-generation positions and/or CoT-in-prompt positions), not a threshold tweak alone.  
4. **Fit quality.** `fit_n=32` synthetic web-ish strings is a smoke fit, not paper-scale. Re-fit with larger / math-leaning corpus before blaming the model alone.  
5. **LoRA path.** Not run (base only).

### Decision

- [ ] GO — proceed to permanent panel / J2 real apply in loop  
- [x] **NO-GO** — do **not** build permanent J-Lens UI yet  

**Allowed after NO-GO:**

- Keep **J1** (`jlens.jsonl` schema + API tail) — already on main.  
- Keep forge CLI spike + lens checkpoint for further experiments.  
- Next experiments (optional, separate PRs):  
  1. Spike protocol v2: generate short CoT *or* inject partial solution, apply at multiple positions.  
  2. Re-fit with larger `fit_n` and/or math-heavy prompts.  
  3. Retry on a larger dense instruct if v2 still fails (e.g. 3–4B on forge).  

**Blocked until GO:**

- Permanent `/observe` J-Lens panel (J4)  
- Treating J-Lens as a product acceptance feature  

## Product next steps

| Step | Status |
|------|--------|
| **J0** forge spike + writeup | **done — NO-GO** |
| **J1** artifact schema (`jlens.jsonl`, `log_jlens`, API tail) | **landed** |
| **J2** `run_grpo(jlens_every=…)` real apply | **blocked** on GO (fake-slice stub OK for CI only) |
| **J3** optional J-Lens worker | blocked |
| **J4** permanent `/observe` panel | **blocked** (spike GO only) |
| **J5** tripwires in UI | blocked |

Still **debugger cadence**, never hot-path rollouts. Attribution in README deferred until a GO product path exists.

## Roadmap

Annotate Phase 2.5 J-lens line: spike **executed**, gate **NO-GO**, panel remains open/blocked.
