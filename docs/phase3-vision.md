# Phase 3 — Vision first-class (working plan)

**Status:** core platform **done**; **productization open** (2026-07-26)  
**Exit criteria:** `docs/roadmap.md` §Phase 3 (3.A–3.D)  
**Datasets:** `docs/datasets-robotics.md`

## Product goal

A researcher, roboticist, or **their agent** can SFT (and later RL) a **~3–4B VLM** on a **real robotics corpus** (Bridge / OXE subsample / Robo2VLM — not toy frames only), with freeze defaults that make sense for edge export (LM + projector LoRA, vision encoder frozen). Vision runs must remain **observable and controllable** on the same HTTP/MCP SSOT as text RL (see [`product.md`](product.md)).

**Bar:** large-ish robotics set + live Anvil watch — same product quality as GRPO observe, not smoke-only.

## Ordered slices

| Slice | Deliverable | Status |
|-------|-------------|--------|
| **P3.0** | Media store harden + message/trajectory serde + JSONL ingest + robotics dataset notes | **done** |
| **P3.1** | Processor-backed VLM renderer (`HFVLMRenderer`) with train/sample prefix tests | **done** |
| **P3.2** | `LocalBackend` image modality + freeze policy + VLM SFT renderer wiring | **done** |
| **P3.3** | `scripts/vlm_smoke.py` — CAS frame + `run_vlm_sft` (fake always; forge local://) | **done** (forge 2026-07-25) |
| **P3.3b** | Pixel fusion in LocalBackend + LeRobot demo (`robot_vlm_sft_demo.py`) | **done** |
| **P3.4** | Classifier / rubric recipe; web knobs for vision freeze | **open** |
| **P3.5** | **Production robotics convert pipeline** (Bridge/OXE/LeRobot → CAS + JSONL); lab corpus on NVMe | **open** (roadmap 3.B) |
| **P3.6** | **VLM/SFT observe SSOT** — metrics.jsonl + probes + `/observe` for vision runs | **open** (roadmap 3.C) |
| **P3.7** | **Multi-hour VLM jobs** — checkpoint/resume, batching notes, lab_smoke full vision profile | **open** (roadmap 3.D) |
| **P4** | Offline robot loop + action tokenization + vision RL + Jetson export | **open** (`roadmap` §Phase 4) |

## Invariants

1. **Refs not blobs** in batches — `cas://sha256/…` via `LocalMediaStore`.  
2. **Same renderer** for train and sample once P3.1 lands.  
3. **Default freeze:** `vision_encoder=False`, `mm_projector=True`, `language=True`.  
4. **No multi‑GB datasets in git** — lab NVMe only.  
5. **Watchability** — vision train must not be a second, log-only path; metrics/SSE share GRPO observe SSOT.

## Robotics LoRA / RL datasets

Use BridgeData V2 / OXE subsample / LeRobot / Robo2VLM as **product sources**, converted into Example or Trajectory rows. See datasets doc for mapping, scale ladder, and smoke checklist (must go green on real corpus, not only synthetic).

## Non-goals (Phase 3)

- Full OXE pretrain on day one (subsample → scale ladder is the path)  
- Action-head architectures beyond text-tokenized actions (Phase 4 recipe)  
- Permanent J-Lens panel (J4; spike parked 2026-07-19 — not on vision critical path)  
