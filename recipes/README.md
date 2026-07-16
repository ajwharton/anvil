# Recipes

**Bounded catalog + gates.** Knobs stay small; architecture boundaries are explicit.

People can stretch past a gate (`force=True`), but Anvil always names whether a combo is:

| Gate | Meaning |
|------|---------|
| **recommended** | Default happy path for this shape |
| **stretch** | Allowed with warnings (size, rank, expert recipes) |
| **blocked** | Refused unless `force=True` |

## Catalog (15)

| ID | Group | Recommended shapes |
|----|-------|-------------------|
| `sft_chat_dense` | train | dense_lm |
| `sft_chat_moe` | train | moe_lm |
| `vlm_sft_lab` | train | dense_vlm (~7B lab) |
| `vlm_sft_edge` | edge | edge_student (e.g. VL-3B) |
| `vlm_classifier` | train | edge_student, dense_vlm |
| `vlm_encoder_lora` | train | stretch only (open vision encoder) |
| `rl_verifiable_dense` | rl | dense_lm, edge_student |
| `rl_verifiable_vlm` | rl | dense_vlm, edge_student |
| `rl_verifiable_moe` | rl | moe stretch |
| `preference_dpo_dense` | preference | dense_lm |
| `preference_dpo_vlm` | preference | dense_vlm, edge_student |
| `robot_offline_edge` | edge | edge_student |
| `distill_to_edge` | edge | edge_student (student side) |
| `tool_agent_sft` | train | dense_lm / vlm / edge |
| `eval_sample_only` | eval | all (no train) |

## Python

```python
from anvil.recipes import (
    list_recipes,
    gate_recipe,
    plan_recipe,
    suggest_for_model,
    recipes_for_shape,
)

list_recipes()  # full catalog
gate_recipe("vlm_sft_edge", shape="edge_student", param_count=3_75e9)
# → recommended

plan = plan_recipe(
    base_model="Qwen/Qwen2.5-VL-3B-Instruct",
    recipe_id="vlm_sft_edge",
)
# blocked combos raise unless force=True:
# plan_recipe(..., recipe_id="sft_chat_moe", force=True)

suggest_for_model("Qwen/Qwen2.5-VL-3B-Instruct")
# shape + gated cards + default_recipe_id
```

## Shape matrix (simplified)

| Shape | Default recipe | Also happy |
|-------|----------------|------------|
| `dense_lm` | `sft_chat_dense` | RL dense, DPO dense, tools |
| `moe_lm` | `sft_chat_moe` | DPO stretch, RL MoE stretch |
| `dense_vlm` | `vlm_sft_lab` | classifier, RL VLM, DPO VLM |
| `edge_student` | `vlm_sft_edge` | robot offline, distill, classifier |

## Design rule

Shape comes from the **HF model card**; the **catalog** maps shape → recipe boundaries; **research** (Tinker / TRL / GRPO) supplies loop families; **we** still fine-tune on our data.
