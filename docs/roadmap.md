# Roadmap

Status legend: **todo** · **doing** · **done** · **blocked**

## North star

A **Tinker-shaped** open toolkit: four verbs, LoRA-first, train/sample consistency, vision in the data model, export to lab serve and **Jetson/edge**.

Success looks like: a researcher or roboticist can SFT/RL a small LLM/VLM from a laptop client against own GPUs without rewriting distributed systems each time.

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

## Phase 2 — Sample/train split *(current)*

**Exit criteria**

- [ ] Dedicated sample worker (vLLM) with adapter hot-swap or snapshot  
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

## Phase 3 — Vision first-class

**Exit criteria**

- [ ] Media store (content-addressed refs)  
- [ ] Multimodal message schema + renderer  
- [ ] VLM SFT recipe + optional classifier-style recipe  
- [ ] Freeze/LoRA knobs for encoder vs projector vs LM  

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
- Proprietary Tinker wire compatibility / trademark use  
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
