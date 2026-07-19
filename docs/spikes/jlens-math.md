# Spike: J-Lens on multi-step math (J0)

**Status:** Spike **parked** (2026-07-19). Protocol v3 `solve` remains a valid measurement (answer readout works; see §Second opinion). Rank-resolved J2 entry **failed** on 1.5B (v5) and on **Qwen2.5-7B mixed** fit (v7 — mean order 0.5 &lt; 0.6). Position-fix v6 was a clean negative.  
**Product call:** **Shelve J2–J5 product work.** Keep J1 schema + forge CLI as optional research tooling. **Anvil’s RL debugger value is metrics / probes / cliffs / live control — not J-Lens.** Re-open only against the high bar below.  
**Roadmap:** Phase 2.5 core complete without J-Lens panel; J0 measurement archive lives here  
**Script:** [`scripts/jlens_spike.py`](../../scripts/jlens_spike.py) (`--protocol solve,last_prompt,cot_in_prompt,generate`)  
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
| Fitted lens (v1 smoke) | `/mnt/data/models/lenses/qwen2.5-1.5b-instruct/jacobian_lens.pt` |
| Fitted lens (**v2 proper**, 1.5B) | `/mnt/data/models/lenses/qwen2.5-1.5b-instruct-v2/jacobian_lens.pt` (fit_n=128, mixed, dim_batch=64, ~36 min) |
| Fitted lens (**7B mixed**, v7) | `/mnt/data/models/lenses/qwen2.5-7b-instruct-v0/jacobian_lens.pt` (fit_n=128, mixed WikiText+math, dim_batch=64, **~4.9 h**) |
| Fit meta | `jacobian_lens.meta.json` alongside each lens |
| Run artifacts | forge: `…/jlens-spike-*`, `/mnt/data/anvil/results/jlens-solve-v5/`, `/mnt/data/anvil/results/jlens-solve-7b-mixed/` |

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

### Decision (v1)

- [ ] GO — proceed to permanent panel / J2 real apply in loop  
- [x] **NO-GO** — do **not** build permanent J-Lens UI yet  

---

## Results v2 (2026-07-17 forge — protocol improvements)

Same model + **same fitted lens** (`fit_n=32`). Script: protocol v2 on branch / PR.

| Field | Value |
|-------|--------|
| Date | 2026-07-17 |
| Artifact dir | `/mnt/data/anvil-runs/jlens-spike-v2-20260717-194416/` |
| Protocols | `cot_in_prompt`, `generate`, `last_prompt` |
| Digit matching | exact (no `"1"` ⊆ `"14"` false positive) |
| **Overall gate** | **NO-GO** (exit 2) |

### Per-protocol gates

| Protocol | mean order | answer top-k hits | Gate |
|----------|------------|-------------------|------|
| `last_prompt` (v1) | null | 0/3 | NO-GO |
| `cot_in_prompt` | **0.25** | 0/3 | NO-GO |
| `generate` | **0.75** | 0/3 | NO-GO (order ok, **never** answer in top-k) |

Best protocol by order: **`generate`** (mean 0.75) — still fails the dual gate because **0/3 probes** show the final answer in lens top-k. Position-order scores were all `null` (stages never hit mid-layer across positions in order).

### What improved vs v1

- CoT-in-prompt and generate **do** produce *some* stage hits (partial operands / ops), so v1’s pure “no signal” was partly protocol.
- Order scores can look respectable on thin stage subsets (e.g. generate/`double_plus` layer_order=1.0 on 2–3 hit stages).

### What did *not* improve

- **Final answer never appears** in top-k under digit-safe matching (final-layer tops often `1,2,4,8` or discourse/math verbs — not `14` / `12` / `13`).
- Mid-layer tops remain noisy / multilingual junk (`Wei`, `补`, `数学`, `SELL`, …).
- Smoke lens + 1.5B is not showing a clean “intermediate workspace → answer” ladder on this task set.

### Decision (v2) / product call

- [ ] GO — invest in permanent panel + train-loop apply  
- [x] **NO-GO again** — **deprioritize permanent J-Lens product work**

**Pursue / keep (low cost):**

- **J1** schema + API (already landed) — inert until something writes real slices  
- Forge CLI spike + saved lens for occasional research  
- Existing RL debugger (metrics, probes, adapter sync) — **this is the product path**

**Do not schedule next (unless new evidence):**

- J2 real apply in `run_grpo`  
- J3 J-Lens worker  
- J4 permanent `/observe` panel  
- J5 J-Lens UI tripwires  
- Larger-model spike “just to hope” without a better task design  

**Re-open criteria (high bar):** paper-scale fit on a stronger base **and** a protocol where ≥ half probes show **answer in top-k** *and* mean order ≥ 0.6 on a fixed probe set. Until then, treat J-Lens as **interesting research, not Anvil roadmap load-bearing**.

## Product next steps

| Step | Status |
|------|--------|
| **J0 v1** last-prompt spike | **done — NO-GO** (protocol artifact) |
| **J0 v2** CoT / generate / digit-safe | **done — NO-GO** (scoring artifacts; superseded) |
| **J0 v3** few-shot solve + digit-aware + sanity | **done — GO** (see §Second opinion) |
| **J0 v4** rank-resolved scoring + foil control | **done** — artifact 4 found & fixed (weak top-k is digit-prior-prone) |
| **J0 v5** proper fit (v2 lens) + J2 entry scoring | **done — J2 entry NOT met (1/3)** (see §Third pass) |
| **J0 v6** intermediate at own-token positions | **done — fails harder** (0/6 inter strong; order null) — see §Fourth pass |
| **J0 v7** 7B mixed fit + solve | **done — J2 entry NOT met** (mean order **0.5**; see §Fifth pass) |
| **J1** artifact schema + API + spike bridge | **landed** (keep; inert without writers) |
| **J2** real apply hook in `run_grpo` | **shelved** — not product path |
| **J3–J5** worker / panel / UI tripwires | **shelved** with J2 |

Attribution in README stays deferred. Spike is **archive + re-open bar**, not a scheduled workstream.

---

## Second opinion (2026-07-18): v1/v2 NO-GO was mostly measurement artifacts

Independent re-review of the setup, artifacts, and `jlens` package after the
v2 NO-GO. Three artifacts in the measurement stack — not the lens, and only
partly the model — produced the negative verdict.

### Artifact 1 — digit-blind answer scoring (structural, gate-breaking)

Qwen2.5 tokenizes **every number digit-by-digit**: `"14" → ["1","4"]`. A
single-token "answer in top-k" check can therefore **never** hit on this
tokenizer. Worse, all three v1/v2 answers (12/13/14) start with `"1"`, so the
strongest real signal (first answer digit at rank 1 at late layers) was
invisible and un-distinguishable from a digit prior. v3 scores the answer as
a **digit sequence at consecutive prediction positions** and adds probes
whose answers do **not** start with 1 (34/25/22).

### Artifact 2 — truncated `generate` window

v2 `generate` gave a chatty 1.5B instruct model 48 tokens; in 2/3 probes the
completion was cut while still *restating the problem* (`"### Step 1:"`). The
measured positions contained **no completed computation** — of course the
answer wasn't in top-k. v3 few-shots a terse `Step 1 / Step 2 / Answer`
format so greedy decoding always completes the arithmetic, and only scores
probes the model actually solved (`answer_correct`).

### Artifact 3 — no lens-vs-model sanity check

`lens.apply` returns `(lens_logits, model_logits, input_ids)`; v1/v2
discarded `model_logits`. Comparing them directly: the smoke-fit lens agrees
with the model's own top-1 at **0.81–0.97** of measured positions (final
layer). On `cot_in_prompt`, lens final-layer top-1 equals the model's top-1
on all 3 probes — the lens was faithful all along; the scoring was not.

### v3 results (forge, same smoke lens `fit_n=32`, CPU apply)

| Field | Value |
|-------|-------|
| Date | 2026-07-18 |
| Protocol | `solve` (few-shot terse CoT → apply on full continuation) |
| Artifact dir | `/mnt/data/anvil-runs/jlens-spike-v3-final-20260718-085742/` |
| Probes | 6 (3 original + 3 with non-"1" answers) |
| `answer_correct` (greedy solves right) | **6/6** |
| Answer digit-seq in top-k (≥1 layer) | **6/6** (gate needs ≥ 3) |
| Answer hit layers | **23–26** on all 6 |
| Intermediate hit layers | 23–26 (4 probes), 25–26 (2 probes) |
| Mean order score | **0.667** (gate needs ≥ 0.6) |
| Sanity (lens/model top-1 agreement) | 0.81–0.90 |
| **Gate** | **GO via protocol=solve** |
| Side artifact | `jlens.jsonl` — 6 J1-schema records (endpoint/tripwire-ready) |

### What v3 does *not* show (honest limits)

1. **Late-layer band, not a ladder.** All hits sit in L23–26. The paper's
   "mid-layer workspace" is not visible — consistent with the lorem-padded
   `fit_n=32` smoke fit corrupting mid-layer readouts (junk tokens:
   `Wei`, `补`, `数学`, `SELL`). A proper fit (varied wikitext + math CoT,
   `fit_n ≥ 128`, ~1–4 h GPU) is required before claiming layer-order
   semantics, and before J2.
2. **Order margin is thin.** 0.667 clears 0.6 on 6 probes; 2 "Double X"
   probes invert (intermediate readable *later* than the answer). Treat as
   "signal present", not "workspace confirmed".
3. **top-k=8, digit granularity.** Rank-sensitive analyses should raise k.

### Revised decision

- [x] **GO (v3)** — J-Lens becomes a first-class debugger-plane feature
      (research preview): schema, scoring, spike CLI, and artifact bridge are
      product code; J2 train-loop apply is the next gate.
- [ ] ~~NO-GO~~ — superseded; v1/v2 negative results re-attributed to scoring
      artifacts above (still valid as *protocol* evidence: last-prompt-only
      readout is the wrong window).

**J2 entry criteria:** re-fit lens (varied corpus, `fit_n ≥ 128`) → re-run
`solve` → require 6/6 answer hits *and* mean order ≥ 0.6 *and* earliest hit
layer ≤ 20 on ≥ half the probes (evidence the mid-layer band opened up).
Then wire `run_grpo` apply via `RunMetricsWriter.log_jlens`.

**Product framing:** with the smoke fit, J-Lens is a *research preview*:
late-layer verbalization readouts + sanity metric, exposed via
`jlens.jsonl` / `GET /api/observe/{run_id}/jlens`. The permanent panel (J4)
earns its place when the proper fit demonstrates ordered mid-layer readout.

---

## Third pass (2026-07-18): proper fit + rank-resolved scoring

The J2 entry run: re-fit the lens properly and re-measure. This pass found a
**fourth measurement artifact** and fixed it; the J2 criteria are scored on
the corrected metric.

### Artifact 4 — weak top-k hits are digit-prior-prone

v3 counted an answer hit when every digit appeared anywhere in the top-8 at
its consecutive position. After `"Answer: "`, Qwen's digit prior keeps small
digits in the top-8 at mid layers **regardless of computation** — e.g. the
first digit of `14` sits at rank 2–6 from L0 onward on every probe. Weak
membership can therefore "hit" without the residual stream containing the
value.

Fix (v4, in `anvil.observe.jlens.strong_hit_layers` + the spike): a **strong
hit** requires *exact ranks* — every digit of the value at rank ≤ 3 at its
position. Plus a **foil control**: rank a digit absent from both values at
the answer position; if the foil ranks about as well as the answer's first
digit, the hits are prior, not content.

### v5 run (forge GPU, v2 proper lens)

| Field | Value |
|-------|-------|
| Date | 2026-07-18 |
| Lens | **v2** — fit_n=128, mixed WikiText-103 + synthetic math CoT, dim_batch=64 (~36 min on GB10) |
| Protocol | `solve`, rank-resolved scoring |
| Artifact dir | `/mnt/data/anvil/results/jlens-solve-v5/` |
| `answer_correct` | **6/6** |
| Answer **strong** hits (all digits rank ≤ 3) | **6/6** |
| Foil separation | answer-d1 mean rank **3.4–13.9** vs foil **18.1–80.0** on all 6 — strong hits are genuine residual-stream content, not prior |
| Sanity (lens/model top-1) | 0.85–0.94 |

Per-probe layers (strong):

| Probe | order | ans strong | inter strong | d1~ | foil~ |
|-------|-------|-----------|--------------|-----|-------|
| `add_then_mul` | 0.0 | 22–26 | 23–26 | 9.3 | 36.5 |
| `sub_chain` | 0.0 | **12**, 22–26 | 23–26 | 9.0 | 22.7 |
| `double_plus` | 0.0 | 22–26 | 25–26 | 10.0 | 80.0 |
| `mul_34` | 0.0 | 22–26 | 23–26 | 13.9 | 30.7 |
| `sub_25` | 1.0 | 23–26 | 23–26 | 3.4 | 18.1 |
| `dbl_22` | 0.0 | **9**, 12, 13, 17, 23–24 | 25–26 | 3.7 | 27.2 |

### J2 entry criteria scorecard

| Criterion | Result | Verdict |
|-----------|--------|---------|
| 6/6 answer hits | 6/6 **strong** hits | **PASS** |
| mean order ≥ 0.6 | **0.167** (1/6) | **FAIL** |
| earliest hit ≤ 20 on ≥ half | **2/6** (`sub_chain` L12, `dbl_22` L9) | **FAIL** |

**J2 entry: not met (1 of 3).** The v3 GO for *answer readout* stands and is
now prior-proofed; the *workspace ladder* (intermediate readable before the
answer) does not hold at 1.5B under this protocol.

### What the rank trajectories actually show

Digit 1 is prior-dominated (rank ≤ 6 from L0 everywhere). Digit 2+ carries
the computation signal: it starts at rank hundreds–1400 in early layers and
declines roughly monotonically — reaching single digits at L9–L14 on
`dbl_22`/`sub_chain`, L22+ elsewhere. The lens **does** watch the value
condense; the v2 fit opened the mid-layer band that the smoke fit had
poisoned. But intermediates only become lens-readable at L23+, at or after
the answer — either the 1.5B model does not serialize through a lens-readable
intermediate at the probed boundary position (token *before* the value), or
the position choice itself is wrong.

### Next experiment (cheapest first)

1. ~~**Position fix**~~ → done, negative (§Fourth pass).
2. ~~**7B mixed fit**~~ → done, order 0.5 still fails J2 (§Fifth pass).

Spike **parked** after §Fifth pass. J1 path stays; J2–J5 not scheduled.

---

## Fourth pass (2026-07-18): intermediate at own-token positions (v6)

Cheapest experiment after v5: keep the v2 proper lens, change only where
intermediates are scored.

| Mode | Position |
|------|----------|
| Answer (unchanged) | residual at token *before* first answer digit (`ans_score_mode=boundary`) |
| Intermediate (v6) | residual at the intermediate digit token(s) themselves (`inter_score_mode=value`) |

### v6 run (forge GPU, same v2 lens)

| Field | Value |
|-------|-------|
| Date | 2026-07-18 |
| Branch / commit | `spike/jlens-v6-posfix` @ `760c0f4` |
| Lens | v2 proper (`…/qwen2.5-1.5b-instruct-v2/jacobian_lens.pt`) |
| Artifact dir | `/mnt/data/anvil/results/jlens-solve-v6-posfix/` |
| `answer_correct` | **6/6** |
| Answer **strong** hits | **6/6** (same late band as v5) |
| Intermediate **strong** hits | **0/6** (v5 had late-layer strong hits on all 6) |
| Intermediate **weak** top-k | 0/6 except `dbl_22` L3 only |
| Mean order | **null** (0 order-scored; need ≥1 inter strong layer) |
| Sanity | 0.81–0.94 |

### Interpretation

At the intermediate's *own* token, the lens does **not** verbalize that
value: sole-digit intermediates sit at rank hundreds–thousands even at L26
(e.g. `add_then_mul` inter `7` → rank 669 at L26). Multi-digit intermediates
show a sharp first-digit / weak second-digit pattern early, then collapse —
never all digits rank ≤ 3.

That matches a **next-token** lens: residual *before* writing the
intermediate predicts it (v5 boundary hits); residual *at* the written
digit predicts the *following* token (newline / `Step 2`), not the digit
itself. Own-token scoring is the wrong window for this apply API.

### J2 entry scorecard (v6)

| Criterion | Result | Verdict |
|-----------|--------|---------|
| 6/6 answer hits | 6/6 strong | **PASS** |
| mean order ≥ 0.6 | null | **FAIL** |
| earliest hit ≤ 20 on ≥ half | n/a (no inter strong) | **FAIL** |

**J2 entry: not met (1 of 3).** Position fix closed as a dead end.
Default scoring stays: answer + intermediate both at **boundary-before-value**
(`--inter-score-mode boundary`, default). The v6 experiment is retained as
`--inter-score-mode value` (negative control).

### Next (historical — superseded by §Fifth pass)

1. ~~**7B** re-fit + solve~~ → done, still fails J2 order (see §Fifth pass).
2. Per-position rank curves remain optional research if re-opened.

---

## Fifth pass (2026-07-19): Qwen2.5-7B mixed fit + solve

Overnight forge run after installing `datasets` so `--fit-corpus mixed`
actually loads WikiText-103 (64 wiki + 64 math CoT).

| Field | Value |
|-------|-------|
| Date | 2026-07-18 → 2026-07-19 |
| Model | `/mnt/data/models/qwen2.5-7b-instruct` (HF `Qwen/Qwen2.5-7B-Instruct`) |
| Lens | **v0 mixed** — fit_n=128, dim_batch=64, bf16, **wikitext: 64 prompts** |
| Fit wall | **17576 s (~4.9 h)** |
| Lens size | ~662 MB |
| Protocol | `solve`, `--inter-score-mode boundary` |
| Artifact dir | `/mnt/data/anvil/results/jlens-solve-7b-mixed/` |
| `answer_correct` | **6/6** |
| Answer **strong** hits | **6/6** |
| Mean order (strong) | **0.50** (3/6 probes order=1.0) |
| Sanity top-1 | **0.95–1.0** (excellent lens/model agreement) |
| Gate | **NO-GO** (script; mean order &lt; 0.6) |

Per-probe strong layers:

| Probe | order | ans strong | inter strong |
|-------|-------|------------|--------------|
| `add_then_mul` | 1.0 | 24–26 | **13**, 14, 16, 22–26 |
| `sub_chain` | 1.0 | 25–26 | 25–26 |
| `double_plus` | 0.0 | 24–26 | 25–26 |
| `mul_34` | 0.0 | 24–26 | 25–26 |
| `sub_25` | 1.0 | 23–26 | **16**, 23–26 |
| `dbl_22` | 0.0 | **8**, 23–26 | 25–26 |

### J2 entry scorecard (7B)

| Criterion | Result | Verdict |
|-----------|--------|---------|
| 6/6 answer hits | 6/6 strong | **PASS** |
| mean order ≥ 0.6 | **0.50** | **FAIL** |
| earliest ans hit ≤ 20 on ≥ half | **1/6** (`dbl_22` L8) | **FAIL** |

**J2 entry: not met (1 of 3).** Better than 1.5B v5 (order 0.167 → 0.50);
intermediates sometimes open mid-layer (L13–16). Still not a clean workspace
ladder, and foil/d1 mean ranks across all layers are noisy on 7B — treat
rank maps as research data, not product signals.

### Product decision (2026-07-19)

- [x] **Park the spike.** Interesting measurement stack; not load-bearing for
      Anvil’s RL platform or debugger story.
- [x] **Keep** J1 schema + CLI + this writeup as optional research tooling.
- [x] **Do not schedule** J2 train-loop apply, J3 worker, J4 panel, J5
      tripwires.
- [x] **Ship value without J-Lens:** metrics, probes, advantage/IS cliffs,
      adapter sync, live knobs, agent/MCP control, vision path.

**Re-open bar (unchanged high bar):** fixed probe set, rank-resolved
boundary scoring, 6/6 strong answer hits, mean order ≥ 0.6, earliest
strong answer hit ≤ 20 on ≥ half the probes — on a protocol that remains
honest under foil controls. Prefer new task design over “try a bigger
base again.”

---
