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

## Phase 1 — Local Anvil (text, single GPU) *(current)*

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
- [ ] `anvil serve --backend local` thin CLI shell on one GPU host  
- [ ] GPU smoke: SFT loop on a small dense model (0.5B–4B) via LoRA on forge  
- [ ] Export adapter (real PEFT dir) → load in vLLM (or HF) for sample — manual verification  
- [ ] Minimal recipe: `recipes/sl_loop.py` running against `local://`  

**Non-goals:** dual-Spark TP train; vision; RL losses (LocalBackend v0 is CE-only).

## Phase 2 — Sample/train split

**Exit criteria**

- [ ] Dedicated sample worker (vLLM) with adapter hot-swap or snapshot  
- [ ] Async futures / queue (API-compatible even if local is sync under the hood)  
- [ ] Simple on-policy RL recipe (e.g. GRPO/math exact-match toy)  
- [ ] IS/PPO loss family in LocalBackend (v0 is CE-only and raises `NotImplementedError`)  
- [ ] GRPO datum carries **prompt+completion** with old-policy logprobs aligned to
      target positions — replaces the completion-only toy shape in `anvil/recipes/grpo.py`  
- [ ] HTTP transport for the four verbs (control plane serves LAN clients; today
      the verbs are in-process only)  
- [ ] Stop-string support in LocalBackend sampling  
- [ ] Gate-override audit events: every `force=True` past a blocked recipe is
      logged with recipe, shape, and reasons (start of the control-plane audit trail)  

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
