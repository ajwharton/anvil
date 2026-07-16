"""Recipe catalog + architecture gates."""

from __future__ import annotations

from anvil.recipes.catalog import (
    GateLevel,
    default_recipe_id_for_shape,
    gate_recipe,
    list_recipes,
    recipes_for_shape,
)
from anvil.recipes.profiles import ModelShape, plan_recipe


def test_catalog_size():
    recipes = list_recipes()
    assert 10 <= len(recipes) <= 20
    ids = {r["id"] for r in recipes}
    assert "sft_chat_dense" in ids
    assert "vlm_sft_edge" in ids
    assert "rl_verifiable_moe" in ids
    assert "robot_offline_edge" in ids
    assert "eval_sample_only" in ids


def test_edge_student_recommends_edge_vlm():
    g = gate_recipe("vlm_sft_edge", shape=ModelShape.EDGE_STUDENT, param_count=3_750_000_000)
    assert g.level == GateLevel.RECOMMENDED
    assert g.ok


def test_moe_blocked_on_vlm_edge_recipe():
    g = gate_recipe("vlm_sft_edge", shape=ModelShape.MOE_LM)
    assert g.level == GateLevel.BLOCKED
    assert not g.ok


def test_dense_vlm_on_edge_recipe_is_stretch():
    g = gate_recipe("vlm_sft_edge", shape=ModelShape.DENSE_VLM, param_count=8_000_000_000)
    assert g.level == GateLevel.STRETCH


def test_huge_params_hard_block():
    g = gate_recipe(
        "sft_chat_dense",
        shape=ModelShape.DENSE_LM,
        param_count=100_000_000_000,
    )
    assert g.level == GateLevel.BLOCKED


def test_encoder_lora_recipe_always_stretch():
    g = gate_recipe("vlm_encoder_lora", shape=ModelShape.DENSE_VLM)
    assert g.level == GateLevel.STRETCH


def test_rank_gate_stretch():
    g = gate_recipe(
        "sft_chat_dense",
        shape=ModelShape.DENSE_LM,
        rank=128,
    )
    assert g.level == GateLevel.STRETCH
    assert any("rank" in r for r in g.stretch_reasons)


def test_recipes_for_shape_orders_recommended_first():
    rows = recipes_for_shape(ModelShape.EDGE_STUDENT, has_vision=True, param_count=3_7e9)
    assert rows
    assert rows[0]["gate"]["level"] == "recommended"
    assert default_recipe_id_for_shape(ModelShape.EDGE_STUDENT) == "vlm_sft_edge"


def test_plan_recipe_by_id_applies_gates():
    plan = plan_recipe(
        base_model="Qwen/Qwen2.5-VL-3B-Instruct",
        recipe_id="vlm_sft_edge",
        fetch_remote=False,
    )
    assert plan.recipe_id == "vlm_sft_edge"
    assert plan.gate is not None
    assert plan.gate["level"] in {"recommended", "stretch"}
    assert plan.lora.rank == 16  # edge default


def test_plan_blocked_raises_without_force():
    import pytest

    with pytest.raises(ValueError, match="blocked"):
        plan_recipe(
            base_model="Qwen/Qwen2.5-VL-3B-Instruct",
            recipe_id="sft_chat_moe",  # blocked for edge VLM
            fetch_remote=False,
            force=False,
        )


def test_plan_blocked_allowed_with_force():
    plan = plan_recipe(
        base_model="Qwen/Qwen2.5-VL-3B-Instruct",
        recipe_id="sft_chat_moe",
        fetch_remote=False,
        force=True,
    )
    assert plan.recipe_id == "sft_chat_moe"
    assert plan.gate is not None
    assert plan.gate["level"] == "blocked"
    assert plan.gate["ok"] is False
