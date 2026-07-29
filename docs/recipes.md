# Recipes, meta-recipes, and the personal recipe book

**Status:** product differentiator (first-class) — catalog **v0 shipped**; personal book + meta-recipes **roadmap**  
**Related:** [`product.md`](product.md) · [`roadmap.md`](roadmap.md) §Recipe book · [`design.md`](design.md) §1.0–1.1 · code: `anvil/recipes/`

---

## Why this is first-class

Post-training quality is not only weights and data. It is **compiled operator judgment**:

- which method and freeze policy for this *family* of models  
- how long to wait on a flat loss before calling plateau  
- when to overshoot (calibration) vs stop (production)  
- which stage follows which cliff  

Competitors ship trainers and sometimes “recipes” as static YAML for one script.  
**Anvil’s differentiator:** recipes are a **first-class control object**—gated, observable, switchable mid-run, agent-readable—and every forge can grow a **personal recipe book** from live runs, not only consume what we ship.

```text
  shipped atlas          personal book (you own)
  (Anvil catalog)        (~/.anvil/recipes or org store)
         │                        │
         └──────── plan / suggest ┘
                      │
                      ▼
              train under observe
                      │
                      ▼
              promote learnings → new recipe / meta-recipe version
```

---

## Vocabulary

| Term | Meaning |
|------|---------|
| **Recipe** | Named, versionable plan: pattern + knobs + gates + stop/probe defaults + notes. Binds to **pattern** and optionally **shape / model family** (not only one snapshot id). |
| **Meta-recipe** | Policy *over* recipes: stage graph, cliff → next recipe, calibration vs production mode, patience priors. “If advantage collapse → SFT recovery; if DPO flat → GRPO.” |
| **Shipped atlas** | In-repo / product catalog (`anvil/recipes/catalog.py`, …)—conservative priors we dogfood. |
| **Personal recipe book** | Operator-local library: promoted runs, family-level habits, org packs. **Sovereign**—not required to leave the lab. |
| **Experience** | Observe artifacts + stop reasons + optional overshoot notes that *justify* a recipe version. |

Today: atlas + personal book + promote-from-run + **org packs** +
**experience → production patience** (`suggest_for_model` priors) + meta-recipe
executor. Calibration-mode recipes do not shift production patience.

---

## Binding scope (recipes ride families)

| Scope | Use when |
|-------|----------|
| **Pattern** | Job type only (SFT, GRPO, …)—thinnest defaults |
| **Shape** | Card-derived architecture class (dense chat, VLM, MoE, …)—gates already think this way |
| **Family** | Lineage (“Qwen2.5-VL *”, “Qwen2.5 instruct dense *”)—shared templates/habits; **portable across siblings** |
| **Instance** | One pinned base/path—lab smoke or regulated pin; exception, not the default identity of a recipe |

A recipe **moves on top of a family**: new size in the same family inherits priors; card/gates still block impossible combos.

---

## Two modes of learning (patience / plateaus)

Post-training effect of data is still more **art** than science; false plateaus exist.

| Mode | Recipe default | Intent |
|------|----------------|--------|
| **Production** | Early-stop with type-scoped patience; max_steps = cap | Ship expert under live sufficiency |
| **Calibration** | Longer budget / weak stop; still full observe | Measure plateau shape; **update** recipe priors |

Personal books should tag which runs may update production patience. Overnight overshoot without labeling is calibration by accident—keep the data, promote deliberately.

Longer-term ambition (not a near gate): finer **per-datum / per-batch** effect on a probe bank so recipes can cite evidence beyond mean loss curves.

---

## Personal recipe book (operator product)

An individual or org using Forge should be able to:

1. **Run** with shipped or local recipe  
2. **Watch** metrics/probes (same SSOT as always)  
3. **Promote** a finished run (or stage graph) → named recipe in **their** book  
4. **Reuse** on the next base in the same family via suggest/plan  
5. **Version** and share inside the org without uploading to Anvil  

Shipped atlas stays the public floor. The book is the private ceiling of experience.

---

## Meta-recipes

Examples (product intent):

- **Ladder** — SFT → preference → GRPO with exit conditions per stage  
- **Recovery** — on cliff X, switch to recipe Y (audited)  
- **Domain pack** — “tabletop VLM specialist”: convert defaults + VLM SFT + probe bank + export  
- **Calibration harness** — forced overshoot + write experience record for patience update  

GRPO `rl_queue` is an early meta-recipe. Generalize: machine-readable, agent-switchable, human-editable.

---

## Differentiator checklist

| We claim | We do not claim |
|----------|-----------------|
| Recipes + meta-recipes are core product surface | One true schedule for all post-training |
| Personal book is first-class for sovereign users | Anvil must host or own your book |
| Family-scoped transfer of learnings | Perfect science of single-datum RL effect (yet) |
| Agents plan/switch via the same recipe objects | HTML-only recipe UI as the SSOT |

---

## Implementation map (when coding)

| Piece | Likely home |
|-------|-------------|
| Shipped atlas | `anvil/recipes/catalog.py` (exists) |
| Plan + gates | `plan_recipe`, web `/api/recipes` (exists) |
| Thin meta | `rl_queue` (exists) |
| Personal store | e.g. `~/.anvil/recipes/` or `ANVIL_RECIPE_BOOK` path |
| Promote API | control plane + MCP `anvil_save_recipe` / list book |
| Experience link | run_id + metrics summary embedded or referenced in recipe meta |

Roadmap gates: [`roadmap.md`](roadmap.md) §Recipe book (P.Recipes).
