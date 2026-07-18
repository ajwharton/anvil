# Roadmap

Status legend: **todo** · **doing** · **done** · **blocked**

## North star

An open post-training toolkit: four verbs, LoRA-first, train/sample consistency, vision in the data model, export to lab serve and **Jetson/edge**—**usable by individuals, optimized for agent control** (live observe + API/MCP + audited method switches when DPO/GRPO/SFT hit cliffs).

Success looks like: a researcher, roboticist, or **their agent** can SFT/RL a small LLM/VLM on own GPUs, see negative returns as they form, and change recipe/method without rewriting infrastructure.

Product thesis: [`docs/product.md`](product.md).

## Phase 0 — Spec & stubs *(complete)*

**Exit criteria**

- [x] Public repo + thin `start.md` / governance / this roadmap  
- [x] Design SSOT in `docs/design.md`  
- [x] Typed sketch of client verbs (`forward_backward`, `optim_step`, `sample`, `save_state`, export) — see `anvil/client/`, `anvil/protocol/`  
- [x] Golden-test harness (fake backend) for SFT one-step — `tests/test_sft_golden.py`  
- [x] Web control plane for knobs + runs (`anvil-web`, spark-dashboard look)  
- [ ] Optional: OpenAPI YAML export of the same types (nice-to-have; Python types are SSOT for now)

**Non-goals:** real multi-node train, production scheduler.

## Phase 1 — Local Anvil (text, single GPU) *(complete)*

**Locked design decisions**

- Verbs are **hand-implemented over torch + PEFT — no HF Trainer**. A Trainer
  would swallow the `forward_backward` / `optim_step` separation that *is* the
  API contract.
- Train and sample share one renderer. `HFChatRenderer` enforces the
  *sample-prompt is a token-exact prefix of the training render* invariant in
  tests (`RendererConsistencyError` on violation).
- Core package stays dependency-free; transformers+jinja2 behind the `hf`
  extra, torch+peft behind the `local` extra.

**Exit criteria**

- [x] `HFChatRenderer` + train/sample prefix-consistency tests — `anvil/render/hf.py`, `tests/test_hf_renderer.py`  
- [x] `LocalBackend` in-process (hand-rolled verbs, real LoRA grads) + CPU golden SFT test — `anvil/backends/local.py`, `tests/test_local_backend.py`; wired as `local://`  
- [x] `anvil serve --backend local` thin CLI shell — `anvil/cli.py`, `anvil/serve/app.py` (+ `RemoteBackend` client half, pulled forward from Phase 2); GPU-host deployment happens with the smoke below  
- [x] GPU smoke: SFT loop on a small dense model (0.5B–4B) via LoRA on forge — Qwen2.5-1.5B-Instruct, rank 16, 100 steps, bf16/CUDA: loss 1.92 → 0.00, greedy sample `'4<|im_end|>'`  
- [x] Export adapter (real PEFT dir) → load in vLLM (or HF) for sample — verified via HF: exported adapter on plain transformers+peft answers 4/4 demo prompts with correct `<|im_end|>`; vLLM load optional follow-up  
- [x] Minimal recipe: `recipes/sl_loop.py` running against `local://` — verified in-process and over HTTP; exit code gates on loss decrease  

**Non-goals:** dual-Spark TP train; vision; RL losses (LocalBackend v0 is CE-only).

## Phase 2 — Sample/train split *(complete)*

**Exit criteria**

- [x] Async futures / queue: `VerbQueue` (single-worker FIFO) behind
      `ServiceClient(queue=True)` — verbs return genuinely non-blocking
      `AnvilFuture`s with submission-order execution; same caller surface as
      the inline path (design §4.3)  
- [x] Dedicated sample worker (vLLM) with adapter hot-swap — `anvil/workers/sample.py`
      (`VLLMSampleBackend`, sampling verbs only over a vLLM engine; `load_snapshot`
      registers PEFT dirs with a fresh LoRA id per push so vLLM never serves a
      stale cached adapter) + `POST /v1/adapters/{id}/load_snapshot` (registered
      only for `SnapshotLoader` backends) + `anvil serve --backend vllm-sample --model`  
- [x] Simple on-policy RL recipe — GRPO/exact-match toy runs end-to-end on
      LocalBackend (sample → reward → group advantages → IS fwd/bwd → optim)  
- [x] IS/PPO loss family in LocalBackend — importance_sampling + ppo (ε=0.2)
      over GRPO-shaped datums; CISPO/DRO/DPO still raise NotImplementedError  
- [x] GRPO datum carries prompt+completion with old-policy logprobs aligned to
      target positions — `recipes.grpo.datum_from_rollout` replaces the
      completion-only toy shape  
- [x] HTTP transport for the four verbs — landed early in Phase 1 (`anvil serve`
      + `RemoteBackend`, LAN trust model with optional bearer token)  
- [x] Stop-string support in LocalBackend sampling — batch-wide early-exit criteria + whole-token truncation at the earliest stop string (`stop_reason="stop"`)  
- [x] Gate-override audit events: every `force=True` past a blocked recipe is
      logged with recipe, shape, and reasons — `anvil/control/audit.py` +
      `/api/audit` (start of the control-plane audit trail)  

## Phase 2.5 — RL observability (the RL debugger) *(core complete; J-Lens shelved)*

**Why:** RL runs fail quietly. Reward climbs while the policy degrades, group
rewards homogenize and advantages collapse to zero, entropy crashes — and the
final eval is the last place any of it shows up. The product answer is a
debugger for RL: while training runs, continuously probe the live policy,
graph the signals that precede the downturn, and watch the rollover into
negative marginal gains as it happens instead of after.

**Ordering (agreed 2026-07-17):** finish the Phase 2 vLLM sample worker first
(it is the sync substrate), then metrics scaffolding, then the live inference
tester, then the J-lens bolt-on last.

**Weight-sync tiers** (inference on a model that is simultaneously training —
the base never changes, only the LoRA adapter does, and that is megabytes):

- Tier 0 (exists): `LocalBackend` samples from the live adapter in-process —
  free, slow (HF generate), fine for probes
- Tier 1 (P2 worker): write adapter to tmpfs, `load_snapshot` hot-swap into
  the running vLLM engine on a K-step cadence; sub-second, no base reload
- Tier 2 (optional, later): in-memory weight push via vLLM worker RPC
  (`apply_model`/`collective_rpc`, the TRL-colocate / veRL / NeMo-RL pattern)
  + `sleep`/`wake_up` for single-box colocation — no memory-editor hacks

**Exit criteria**

- [x] Metrics scaffolding: per-run `metrics.jsonl` appended every RL step —
      reward mean/std, within-group reward std (advantage-collapse tripwire),
      IS `mean_ratio` drift, entropy, loss; SSE endpoint + live charts in
      anvil-web — `anvil/observe/metrics.py` (`RunMetricsWriter`),
      `run_grpo(run_dir=...)`; `/api/observe/*` + `/observe/{run_id}` page
      (entropy lands when the sampler exposes it)
- [x] Live inference tester (Tier 0): fixed probe set sampled greedily from
      the *current* policy every K steps during a run — `run_grpo(probes=...,
      probe_every=K, detokenize=...)` → `probes.jsonl`, scored with the reward
      fn; probe completions rendered inline next to the curves — eyes catch
      reward hacking before scalars do (Tier 1 vLLM-worker probing still open)
- [x] Adapter-sync cadence knob in `run_grpo` (every K steps push
      `snapshot_for_sample` → sample worker `load_snapshot`) —
      `sample_endpoint` / `sample_backend` + `sync_every` + `sample_adapter_id`;
      metrics record `adapter_synced` / `snapshot_path`; web RL debugger knobs
      (`probe_every`, `sync_every`, sample endpoint, write_metrics) on
      `anvil-web` + `/api/defaults.rl_knobs`
- [x] J-lens spike **v1** on forge (2026-07-17): last-prompt protocol **NO-GO**
      (writeup [`docs/spikes/jlens-math.md`](spikes/jlens-math.md)).
- [x] J-lens spike **v2** on forge (2026-07-17): `cot_in_prompt` + `generate` +
      digit-safe match — still **NO-GO** (best mean order 0.75 on generate but
      **0/3 answer top-k hits**). Artifacts
      `/mnt/data/anvil-runs/jlens-spike-v2-20260717-194416/`.
- [x] J1 artifact path (schema): `anvil/observe/jlens.py` +
      `RunMetricsWriter.log_jlens` → `jlens.jsonl`;
      `GET /api/observe/{run_id}/jlens` (optional; no panel).
- [ ] J-lens **permanent panel / train-loop apply** — **shelved** until re-open
      criteria in the spike writeup (not near-term roadmap load-bearing).
      Grounding paper remains Gurnee et al. 2026-07-06 if revived.

**Non-goals:** per-token J-lens on rollouts (debugger view, not hot path);
autointerp beyond the J-lens readout.

## Phase 3 — Vision first-class *(current)*

**Working plan:** [`docs/phase3-vision.md`](phase3-vision.md) · **datasets:** [`docs/datasets-robotics.md`](datasets-robotics.md)

**Exit criteria**

- [x] Media store (content-addressed refs) — `LocalMediaStore` + `put_path` / path resolve  
- [x] Multimodal message schema + serde — `Example`/`Message` public JSON; **Trajectory** for robot demos  
- [x] JSONL / path ingest — `anvil.data.ingest` (Bridge/OXE convert *into* this shape)  
- [x] Multimodal **renderer** (processor-backed `HFVLMRenderer`) + train/sample prefix tests  

- [~] VLM SFT recipe wiring (`run_vlm_sft` + renderer); **real frames on forge** still open  
- [x] Freeze/LoRA knobs for encoder vs projector vs LM — recipe plans + LocalBackend enforce

**Near-term test sources:** BridgeData V2, OXE subsample, LeRobot hub sets, Robo2VLM — see datasets doc.

## Phase 4 — Robot / Jetson edge loop

**Exit criteria**

- [ ] Offline trajectory format (obs/action/reward + frame refs)  
- [ ] Export path documented (ONNX and/or TRT and/or GGUF as applicable)  
- [ ] Distill or small-student path for edge FPS/power notes  
- [ ] Optional `backend=jetson` sample stub (may be remote process)  

## Phase 5 — Multi-tenant lab (optional)

**Exit criteria**

- [ ] Auth + adapter isolation for shared lab hardware  
- [ ] Warm base pool design (optional; single-tenant may remain default)  
- [ ] Audit log for multi-user actions (builds on the Phase 2 gate-override events)  

## Explicit non-goals (until revisited by RFC)

- Full-parameter FT of 100B+ MoE on two Sparks as the default path  
- Proprietary third-party API wire compatibility or trademark collision  

- Arbitrary remote code execution for custom losses (named plugins only)  
- Replacing day-to-day dual-Spark **serve** of large base models (Anvil is train/adapt/export)

## Near-term handoff checklist

For a spin-off agent session:

1. `read start.md`  
2. Outcome: e.g. “Phase 0 OpenAPI + fake backend golden test”  
3. Pull-on-miss: `docs/design.md`, this file  
4. No mia-rl Ship work in this repo  

## Changelog (project-level)

| Date | Note |
|------|------|
| 2026-07-16 | Repo scaffold; design imported; Phase 0 open |
| 2026-07-16 | Phase 0 API stubs + fake backend + SFT golden test (`0.0.1`) |
| 2026-07-16 | Web control plane `anvil-web` (spark-dashboard visual language) |
| 2026-07-16 | Knowledge pour: `recipes/families.py` (14 per-family records — fused-Phi targets, MoE router ban, Gemma-2 softcap, R1-distill LR), operator notes on all recipes, +2 recipes (reasoning traces, continued pretrain), RL rollout temp 1.0 |
| 2026-07-16 | HF model-card inspect + basic SFT/VLM/GRPO recipes (research-shaped) |
| 2026-07-16 | 15-recipe catalog with recommended/stretch/blocked architecture gates |
| 2026-07-16 | `HFChatRenderer` + prefix-consistency tests; `LocalBackend` (torch+PEFT, hand-rolled verbs) with CPU golden SFT; `local://` endpoint; `[hf]`/`[local]` extras; ruff clean (`0.0.2`) |
| 2026-07-16 | Opinionated gates: `ModelTooSmallError` hidden-size floor (block <16 / warn <32, `allow_tiny_models` escape), zero-trainable-param check, renderer transformers>=4.44 + jinja2 floors |
| 2026-07-16 | `anvil serve` — HTTP transport for the four verbs + `RemoteBackend` (serde codec, FastAPI shell, bearer token); Phase 2 transport item pulled forward |
| 2026-07-16 | Fix: serve error mapping via concrete exception classes (blanket `Exception` handler re-raised through ServerErrorMiddleware) |
| 2026-07-16 | `recipes/sl_loop.py` — minimal SFT loop + smoke gate; verified against `local://` in-process and over HTTP |
| 2026-07-17 | **Phase 1 complete** — forge GPU smoke: Qwen2.5-1.5B-Instruct LoRA rank 16 × 100 steps on GB10 (bf16/CUDA), loss 1.92 → 0.00, trained adapter answers `'4<|im_end|>'`; exported PEFT dir reloaded in plain HF transformers+peft, 4/4 correct with proper EOS. Forge env: `/mnt/data/anvil-venv` (torch 2.11.0+cu130), repo at `/mnt/data/anvil`, runs under `/mnt/data/anvil-runs` |
| 2026-07-17 | Phase 2 opened — stop-string support in LocalBackend sampling (batch early-exit + per-row whole-token truncation, `stop_reason="stop"`) |
| 2026-07-17 | Gate-override audit events: `anvil/control/audit.py` + `/api/audit`; `plan_recipe(record_override=)`; also fixes latent `POST /api/plan` 422 (closure-local `PlanIn` under future-annotations) |
| 2026-07-17 | IS/PPO loss family lands in LocalBackend over GRPO-shaped datums (`grpo.datum_from_rollout`: model_input = prompt+completion[:-1], old-policy logprobs aligned to completion targets). Golden tests caught two real bugs: sampler logprobs came from warped HF *scores* (post top-p/temp) instead of raw logits — old-policy logprobs must be the true policy distribution; and the completion-position slice gathered along the vocab dim instead of the sequence dim. Both fixed; `mean_ratio≈1.0` on-policy and +advantage raises completion logprobs |
| 2026-07-17 | GRPO loop runs end-to-end on LocalBackend — `run_grpo(endpoint="local://")` with real logprobs and grads (sample → reward → group advantages → IS fwd/bwd → optim); covered by `test_grpo_loop_local_backend` |
| 2026-07-17 | Async futures / queue (design §4.3): `VerbQueue` single-worker FIFO behind `ServiceClient(queue=True)`; `forward_backward`/`optim_step`/`sample`/`compute_logprobs` return non-blocking `AnvilFuture`s, sync verbs route through the queue for serialization; `queue=False` keeps the inline path. 95/95 tests, ruff clean |
| 2026-07-17 | Phase 2.5 scoped — RL observability (the RL debugger): per-run `metrics.jsonl` + SSE live charts, fixed-probe inference tester on the live policy every K steps, adapter-sync cadence knob; J-lens spike LAST, gated on reproducing "intermediate steps light up in order" (Anthropic global-workspace paper, 2026-07-06, `anthropics/jacobian-lens`). Agreed ordering: vLLM worker → metrics → probe tester → J-lens |
| 2026-07-17 | vLLM sample worker lands: `VLLMSampleBackend` (sampling verbs only; training verbs 501) with LoRA hot-swap via `SnapshotLoader.load_snapshot` + `POST /v1/adapters/{id}/load_snapshot`; fresh LoRA int id per push defeats vLLM's stale-adapter cache; `_prompt_logprob_series` trims vLLM's trailing continuation-logprob quirk so logprobs align to prompt length. 11 tests with a fake vllm module; 106/106, ruff clean |
| 2026-07-17 | vLLM worker verified cross-node: mac → `forge:8741` (vllm 0.25.1, Qwen2.5-1.5B) — greedy base sample, 404 on unknown adapter, Phase 1 adapter hot-swap shifts output (`' Paris. The capital of France'` → `' Paris. the capital of Paris'`) with deterministic re-push on a fresh LoRA id, `compute_logprobs` prompt-aligned, 501 on training verbs. Forge env: bench-venv needed `ninja` + its bin on PATH for vLLM's JIT; `LoRARequest` deep-imported (not top-level in 0.25) |
| 2026-07-17 | P2.5 metrics scaffolding lands: `anvil/observe/metrics.py` (`RunMetricsWriter` → `metrics.jsonl`/`probes.jsonl`, `advantage_collapsed` tripwire, `schema_version` on every record); `run_grpo(run_dir=..., probes=..., probe_every=K, detokenize=...)` emits per-step reward mean/std + within-group reward std + IS mean_ratio passthrough + loss + wall time, and greedy probes of the LIVE policy scored with the reward fn; anvil-web serves `/api/observe/*` (tail + SSE stream) and a standalone `/observe/{run_id}` page with live reward/group-std chart, collapse banner, and probe panel. 7 new tests, 113/113, ruff clean |
| 2026-07-17 | P2.5 adapter-sync cadence: `run_grpo(sample_endpoint=|sample_backend=, sync_every=K)` pushes `snapshot_for_sample` → `load_snapshot` on the sample worker; FakeBackend + RemoteBackend implement `load_snapshot`; metrics carry `adapter_synced`/`snapshot_path`/`sample_endpoint`; anvil-web RL debugger section exposes probe_every, sync_every, sample_endpoint, sample_adapter_id, write_metrics (+ defaults.rl_knobs). J-lens remains the last open 2.5 gate |
| 2026-07-17 | J0 spike runbook: `scripts/jlens_spike.py` + `docs/spikes/jlens-math.md` (forge GO/NO-GO) |
| 2026-07-17 | J1 J-Lens artifact path: `anvil/observe/jlens.py` schema v1, `log_jlens` → `jlens.jsonl`, order-score helpers + `jlens_order_collapsed`, `GET /api/observe/{id}/jlens` — no real lens dep; permanent panel still spike-gated |
| 2026-07-17 | J0 forge run **NO-GO**: Qwen2.5-1.5B + fit_n=32 (~18 min), lens at `/mnt/data/models/lenses/qwen2.5-1.5b-instruct/`; apply on 3 math probes at last prompt token → 0 stage hits / 0 answer hits (late tops = Sure/First/Okay). Artifacts `/mnt/data/anvil-runs/jlens-spike-20260717-191152/`. Writeup filled; permanent panel remains blocked |
| 2026-07-17 | J0 protocol **v2** re-spike **NO-GO**: cot_in_prompt / generate / last_prompt + digit-safe match; generate best mean order 0.75 but 0/3 answer top-k. **Product call: shelve permanent J-Lens panel** (keep J1 schema + CLI). Artifacts `/mnt/data/anvil-runs/jlens-spike-v2-20260717-194416/` |
| 2026-07-17 | **Phase 3 opened (P3.0):** media store `put_path`/path resolve; Example/Message public serde; `Trajectory` → VLM SFT examples; `anvil.data.ingest` JSONL path ingest; docs `phase3-vision.md` + `datasets-robotics.md` (OXE, Bridge, OpenVLA, LeRobot, Robo2VLM). Next: processor-backed VLM renderer |
| 2026-07-17 | **P3.1** `HFVLMRenderer`: processor-backed multimodal render; media-store image load; train/sample prefix invariant; image_refs on SFT Datum; fake-processor unit tests (no multi-GB download) |
| 2026-07-17 | **P3.2** LocalBackend image modality + LoraTargets freeze/target_modules; CE metrics n_image_refs; run_sft/run_vlm_sft accept renderer; HFVLMRenderer auto on local:// + media_store |
