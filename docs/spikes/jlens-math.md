# Spike: J-Lens on multi-step math (J0)

**Status:** template — fill after forge run  
**Roadmap:** Phase 2.5 last open gate  
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
| Base model | `/mnt/data/models/Qwen2.5-1.5B-Instruct` (or HF cache equivalent) |
| Fitted lens | `/mnt/data/models/lenses/Qwen2.5-1.5B-Instruct/jacobian_lens.pt` |
| Run artifacts | `/mnt/data/anvil-runs/jlens-spike-*/` |

Weights and lens checkpoints stay on forge NVMe — never commit them.

## Runbook (forge)

```bash
# shared venv example — adjust to lab reality
source /mnt/data/anvil-venv/bin/activate   # or create one
pip install -U torch transformers
pip install 'git+https://github.com/anthropics/jacobian-lens.git'

cd /mnt/data/anvil   # or your checkout
python scripts/jlens_spike.py check \
  --model-path /mnt/data/models/Qwen2.5-1.5B-Instruct

python scripts/jlens_spike.py all \
  --model-path /mnt/data/models/Qwen2.5-1.5B-Instruct \
  --device cuda \
  --out /mnt/data/anvil-runs/jlens-spike-$(date +%Y%m%d)

# exit 0 = gate GO, exit 2 = NO-GO with artifacts written
```

If 1.5B is NO-GO: re-run with a 3–4B dense instruct on the same script (`--model-path` / `--hf-id`).

## Gate criteria (script-enforced)

| Signal | Threshold |
|--------|-----------|
| Mean `intermediate_order_score` over scored probes | **≥ 0.6** |
| Probes where answer appears in lens top-k at some layer | **≥ half** of probes |

`intermediate_order_score` = fraction of consecutive *hit* stages whose earliest lighting layer is non-decreasing (see script). Missing stages are skipped (need ≥2 hits to score a probe).

**GO** → open J1 PR (artifact schema + fake tests); permanent panel only after J1–J2.  
**NO-GO** → stay CLI-only; note model size / fit_n / failures below; do not build web panel.

## Results (fill in)

| Field | Value |
|-------|--------|
| Date | |
| Host | forge / hammer |
| Model path | |
| Lens path | |
| `fit_n` | |
| mean order score | |
| answer top-k hits | |
| **Gate** | GO / NO-GO |
| Script exit code | 0 / 2 |
| Artifact dir | `/mnt/data/anvil-runs/…` |

### Notes / screenshots

- Mid-layer tops for `add_then_mul` (paste or link):  
- Failure modes (empty stages, OOM, API skew):  
- LoRA follow-up (optional): base lens on PEFT adapter path — same / worse?

### Decision

- [ ] GO — proceed to J1 (`anvil/observe/jlens.py` schema + tests)  
- [ ] NO-GO — larger model / more fit data / abandon panel for now  

## Product next steps

| Step | Status |
|------|--------|
| **J1** artifact schema (`jlens.jsonl`, `log_jlens`, API tail) | **landed** — no real lens required |
| **J2** `run_grpo(jlens_every=…)` apply hook | after forge **GO** (or stub with fake slice for CI) |
| **J3** optional J-Lens worker | optional |
| **J4** permanent `/observe` panel | **spike GO only** |
| **J5** tripwires in UI | after J4 |

Still **debugger cadence**, never hot-path rollouts.

## Copy into roadmap

When filled, tick or annotate the J-lens spike line in `docs/roadmap.md` and link this file.
