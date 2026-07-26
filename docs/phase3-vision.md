# Phase 3 — Vision first-class (working plan)

**Status:** core platform **done**; **productization open** (2026-07-26)  
**Exit criteria:** `docs/roadmap.md` §Phase 3 (3.A–3.D)  
**Cross-cutting SSOT:** `docs/roadmap.md` §Platform goal — Live post-training sufficiency  
**Datasets:** `docs/datasets-robotics.md` · **Product thesis:** `docs/product.md`

## Product goal

A researcher, roboticist, or **their agent** can SFT (and later RL) a **~3–4B VLM** on a **real robotics corpus**, with freeze defaults that make sense for edge export. Vision runs use the **same live sufficiency loop** as text: metrics/probes/cliffs **while** data is applied, so “enough” and “shift gears” are mid-run decisions—not fire-and-forget full epochs then eval.

## Ordered slices

| Slice | Deliverable | Status |
|-------|-------------|--------|
| **P3.0–P3.3b** | Media, schema, renderer, pixel fusion, forge VLM/LeRobot smoke | **done** |
| **P3.4** | Classifier / rubric recipe; web knobs for vision freeze | **open** |
| **P3.5** | Production robotics convert (CLI shipped); lab Bridge extract still open | **partial** |
| **P3.6** | VLM/SFT observe SSOT (metrics.jsonl + /observe); probes still open | **partial** |
| **P3.7** | Multi-hour VLM jobs (roadmap 3.D / P.Ops) | **open** |
| **P4** | Offline robot RL, action tokens, vision RL, Jetson | **open** |

## Invariants

1. **Refs not blobs** — `cas://` via `LocalMediaStore`.  
2. **Same renderer** for train and sample.  
3. **Default freeze:** vision encoder off; projector + LM LoRA on.  
4. **No multi‑GB datasets in git.**  
5. **No second-class observe path** — vision is not log-only while text GRPO gets `/observe`.  

## Non-goals (Phase 3)

- Full OXE pretrain on day one (scale ladder instead)  
- Continuous action heads beyond text-tokenized actions (Phase 4 recipe)  
- Permanent J-Lens panel (parked)  
