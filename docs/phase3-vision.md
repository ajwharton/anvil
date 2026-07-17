# Phase 3 — Vision first-class (working plan)

**Status:** open (2026-07-17)  
**Exit criteria:** see `docs/roadmap.md` §Phase 3  
**Datasets:** `docs/datasets-robotics.md`

## Product goal

A researcher or roboticist can SFT (and later RL) a **VLM** with **image refs** through the same four verbs, with freeze defaults that make sense for edge export (LM + projector LoRA, vision encoder frozen).

## Ordered slices

| Slice | Deliverable | Status |
|-------|-------------|--------|
| **P3.0** | Media store harden + message/trajectory serde + JSONL ingest + robotics dataset notes | **done** |
| **P3.1** | Processor-backed VLM renderer (`HFVLMRenderer`) with train/sample prefix tests | **this PR** |
| **P3.2** | `LocalBackend` / `local://` multimodal CE path (or explicit “text loss + frozen vision” smoke) | next |
| **P3.3** | `run_vlm_sft` on forge with real frames (Bridge or toy CAS) + export | next |
| **P3.4** | Classifier / rubric recipe; web knobs for vision freeze | later |
| **P4** | Full offline robot loop (Jetson export) builds on Trajectory | later |

## Invariants

1. **Refs not blobs** in batches — `cas://sha256/…` via `LocalMediaStore`.  
2. **Same renderer** for train and sample once P3.1 lands.  
3. **Default freeze:** `vision_encoder=False`, `mm_projector=True`, `language=True`.  
4. **No multi‑GB datasets in git** — lab NVMe only.

## Robotics LoRA / RL datasets

Use BridgeData V2 / OXE subsample / LeRobot / Robo2VLM as **test sources**, converted into Example or Trajectory rows. See datasets doc for mapping and smoke checklist.

## Non-goals (Phase 3)

- Full OXE pretrain  
- Action-head architectures beyond text-tokenized actions  
- Permanent J-Lens panel (shelved)  
