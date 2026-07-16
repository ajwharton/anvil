"""Architecture → pattern → knobs derivation."""

from __future__ import annotations

from anvil.recipes import JobPattern, ModelShape, infer_shape, plan_recipe, suggest_for_model


def test_infer_qwen_vl_3b_is_edge_student():
    assert infer_shape("Qwen/Qwen2.5-VL-3B-Instruct") == ModelShape.EDGE_STUDENT


def test_infer_qwen_vl_7b_is_dense_vlm():
    assert infer_shape("Qwen/Qwen2.5-VL-7B-Instruct") == ModelShape.DENSE_VLM


def test_infer_phi_is_dense_lm():
    assert infer_shape("microsoft/Phi-4-mini-instruct") == ModelShape.DENSE_LM


def test_vlm_sft_freezes_encoder():
    plan = plan_recipe(
        base_model="Qwen/Qwen2.5-VL-3B-Instruct",
        pattern=JobPattern.VLM_SFT,
    )
    assert plan.loss_fn == "cross_entropy"
    assert plan.lora.vision_encoder is False
    assert plan.lora.mm_projector is True
    assert "image" in plan.modalities
    assert plan.export_hint in {"onnx", "peft"}
    knobs = plan.as_knobs()
    assert knobs["vision_encoder_lora"] is False


def test_rl_uses_importance_sampling_and_lower_lr():
    plan = plan_recipe(
        base_model="Qwen/Qwen3.5-4B",
        pattern=JobPattern.RL_VERIFIABLE,
    )
    assert plan.loss_fn == "importance_sampling"
    assert plan.learning_rate <= 5e-5
    assert plan.max_steps >= 200


def test_suggest_orders_vision_recipes_for_vlm():
    s = suggest_for_model("Qwen/Qwen2.5-VL-3B-Instruct")
    assert s["shape"] == "edge_student"
    assert s["default_recipe_id"] == "vlm_sft_edge"
    ids = [c["recipe_id"] for c in s["recipes"]]
    assert "vlm_sft_edge" in ids
    # recommended recipes come first
    assert s["recipes"][0]["gate"]["level"] == "recommended"


def test_overrides_patch_knobs():
    plan = plan_recipe(
        base_model="Qwen/Qwen2.5-VL-3B-Instruct",
        pattern="vlm_sft",
        overrides={"rank": 8, "learning_rate": 1e-3},
    )
    assert plan.lora.rank == 8
    assert plan.learning_rate == 1e-3
