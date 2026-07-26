# Phase 3 — Vision first-class (working plan)

**Status:** core platform **done**; open items re-homed under **Expert-v0/v1** + robotics path (2026-07-26)  
**Exit criteria:** `docs/roadmap.md` Expert ladder + historical §Phase 3 archive  
**Cross-cutting SSOT:** `docs/roadmap.md` Expert-v0/v1 (live sufficiency)  
**Datasets:** `docs/datasets-robotics.md` · **Product thesis:** `docs/product.md`

## Product goal

A researcher, roboticist, or **their agent** can SFT (and later RL) a **~3–4B VLM** on a **real robotics corpus**, with freeze defaults that make sense for edge export. Vision runs use the **same live sufficiency loop** as text: metrics/probes/cliffs **while** data is applied, so “enough” and “shift gears” are mid-run decisions—not fire-and-forget full epochs then eval.

## Ordered slices

| Slice | Deliverable | Status |
|-------|-------------|--------|
| **P3.0–P3.3b** | Media, schema, renderer, pixel fusion, forge VLM/LeRobot smoke | **done** |
| **P3.4** | Classifier / rubric recipe; web knobs for vision freeze | **open** |
| **P3.5** | Convert CLI shipped; lab Bridge extract → Expert-v0 | **partial** |
| **P3.6** | VLM metrics + /observe done; probes → Expert-v0/v1 | **partial** |
| **P3.7** | Multi-hour VLM → Expert-v2 | **open** |
| **P4** | Robot offline/on-policy + Jetson → Path: robotics / edge | **open** |

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
