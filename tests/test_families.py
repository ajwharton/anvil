"""Family knowledge layer + recipe-knowledge pour — unit tests (no HF downloads)."""

from __future__ import annotations

from anvil.recipes.catalog import _BY_ID, RECIPES, list_recipes
from anvil.recipes.families import FAMILIES, lookup_family
from anvil.recipes.profiles import ModelShape, plan_recipe


def test_family_lookup_phi_fused():
    fam = lookup_family("microsoft/Phi-4-mini-instruct", model_type="phi3")
    assert fam is not None
    assert fam.id == "phi3"
    assert "qkv_proj" in fam.lora_targets
    assert "q_proj" not in fam.lora_targets


def test_family_lookup_qwen3_moe():
    fam = lookup_family("Qwen/Qwen3-30B-A3B", model_type="qwen3_moe")
    assert fam is not None
    assert fam.id == "qwen3-moe"
    assert "router" not in fam.lora_targets
    assert any("router" in c.lower() or "expert" in c.lower() for c in fam.cautions)


def test_family_lookup_order_specific_first():
    # qwen3-moe must win over generic qwen3
    fam = lookup_family("Qwen/Qwen3-30B-A3B", model_type="qwen3_moe")
    assert fam is not None and fam.id == "qwen3-moe"
    fam2 = lookup_family("Qwen/Qwen3-4B", model_type="qwen3")
    assert fam2 is not None and fam2.id == "qwen3"


def test_family_unknown_returns_none():
    assert lookup_family("some-org/novel-arch-9000", model_type="novelarch") is None


def test_plan_recipe_phi_overrides_targets():
    plan = plan_recipe(base_model="microsoft/Phi-4-mini-instruct", use_card=False)
    assert plan.peft_target_modules == ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
    assert any("family=phi3" in r for r in plan.rationale)
    assert any("FUSED" in c for c in plan.cautions)


def test_plan_recipe_phi_lr_capped():
    # phi family carries sft_lr_cap=1e-4; plan lr must never exceed it.
    # (derive already gives 1e-4 for this base, so the cap is a no-op here —
    # the caution only fires when derivation/recipe defaults exceed the cap)
    plan = plan_recipe(base_model="microsoft/Phi-4-mini-instruct", use_card=False)
    assert plan.learning_rate <= 1e-4
    fam = lookup_family("microsoft/Phi-4-mini-instruct", model_type="phi3")
    assert fam is not None and fam.sft_lr_cap == 1e-4


def test_plan_recipe_qwen3_moe_family():
    plan = plan_recipe(base_model="Qwen/Qwen3-30B-A3B", use_card=False)
    assert plan.shape == ModelShape.MOE_LM
    assert "q_proj" in plan.peft_target_modules
    assert any("router" in c.lower() or "expert" in c.lower() for c in plan.cautions)


def test_plan_recipe_gemma2_softcap_caution():
    plan = plan_recipe(base_model="google/gemma-2-9b-it", use_card=False)
    assert any("eager" in c for c in plan.cautions)


def test_plan_recipe_rl_default_temperature():
    plan = plan_recipe(
        base_model="Qwen/Qwen2.5-0.5B", pattern="rl_verifiable", use_card=False
    )
    assert plan.temperature == 1.0
    assert any("GRPO practice" in r for r in plan.rationale)


def test_plan_recipe_user_override_beats_family():
    plan = plan_recipe(
        base_model="microsoft/Phi-4-mini-instruct",
        use_card=False,
        overrides={"peft_target_modules": ["qkv_proj"], "learning_rate": 9e-4},
    )
    assert plan.peft_target_modules == ("qkv_proj",)
    assert plan.learning_rate == 9e-4


def test_new_recipes_exist_with_notes():
    assert "sft_reasoning_traces" in _BY_ID
    assert "sft_continued_pretrain" in _BY_ID
    assert "think" in _BY_ID["sft_reasoning_traces"].notes
    assert "forgetting" in _BY_ID["sft_continued_pretrain"].notes


def test_all_recipes_carry_notes():
    missing = [r.id for r in RECIPES if not r.notes.strip()]
    assert missing == []


def test_notes_surfaced_in_public_listing():
    rows = list_recipes()
    assert all("notes" in row for row in rows)
    dense = next(r for r in rows if r["id"] == "sft_chat_dense")
    assert "Workhorse" in dense["notes"]


def test_catalog_size_bounds():
    assert 10 <= len(RECIPES) <= 20


def test_reasoning_recipe_gates_dense_recommended():
    from anvil.recipes.catalog import GateLevel, gate_recipe

    g = gate_recipe("sft_reasoning_traces", shape=ModelShape.DENSE_LM, param_count=8_000_000_000)
    assert g.level == GateLevel.RECOMMENDED
    g2 = gate_recipe("sft_reasoning_traces", shape=ModelShape.DENSE_VLM, param_count=8_000_000_000)
    assert g2.level == GateLevel.BLOCKED


def test_family_ids_unique():
    ids = [f.id for f in FAMILIES]
    assert len(ids) == len(set(ids))
