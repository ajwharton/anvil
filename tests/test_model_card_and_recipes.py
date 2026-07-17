"""Model-card inspection + basic SFT/GRPO recipe loops."""

from __future__ import annotations

import json
from pathlib import Path

from anvil.recipes import inspect_base_model, plan_recipe
from anvil.recipes.grpo import group_advantages, run_grpo
from anvil.recipes.sft import run_sft
from anvil.recipes.vlm_sft import build_plan as vlm_plan
from anvil.recipes.vlm_sft import run_vlm_sft


QWEN_VL_CONFIG = {
    "architectures": ["Qwen2_5_VLForConditionalGeneration"],
    "model_type": "qwen2_5_vl",
    "hidden_size": 2048,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "max_position_embeddings": 128000,
    "image_token_id": 151655,
    "video_token_id": 151656,
    "torch_dtype": "bfloat16",
    "vision_config": {
        "depth": 32,
        "hidden_size": 1280,
        "patch_size": 14,
    },
}

QWEN_README = """---
pipeline_tag: image-text-to-text
tags:
- multimodal
---

# Qwen2.5-VL-3B-Instruct

Qwen2.5-VL-3B, which is a solution for edge AI, understands images and acts as a visual agent.
"""


def test_local_card_qwen_vl_edge(tmp_path: Path):
    d = tmp_path / "Qwen2.5-VL-3B-Instruct"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(QWEN_VL_CONFIG), encoding="utf-8")
    (d / "README.md").write_text(QWEN_README, encoding="utf-8")

    card = inspect_base_model(str(d), fetch_remote=False)
    assert card.has_vision is True
    assert card.model_type == "qwen2_5_vl"
    assert card.shape.value == "edge_student"
    assert card.shape_confidence in {"medium", "high"}
    assert "q_proj" in card.peft_target_modules
    assert "vlm_sft" in card.recommended_patterns


def test_plan_uses_card_shape(tmp_path: Path):
    d = tmp_path / "Qwen2.5-VL-3B-Instruct"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(QWEN_VL_CONFIG), encoding="utf-8")
    (d / "README.md").write_text(QWEN_README, encoding="utf-8")

    plan = plan_recipe(
        base_model=str(d),
        pattern="vlm_sft",
        fetch_remote=False,
    )
    assert plan.shape.value == "edge_student"
    assert plan.lora.vision_encoder is False
    assert plan.export_hint == "onnx"
    assert plan.peft_target_modules
    assert plan.sources  # research citations attached


def test_run_sft_fake():
    result = run_sft(base_model="toy/lm", steps=2, endpoint="fake://")
    assert result.steps_run == 2
    assert len(result.losses) == 2
    assert result.adapter_id


def test_run_vlm_sft_local_card(tmp_path: Path, monkeypatch):
    d = tmp_path / "Qwen2.5-VL-3B-Instruct"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(QWEN_VL_CONFIG), encoding="utf-8")
    (d / "README.md").write_text(QWEN_README, encoding="utf-8")
    monkeypatch.setenv("ANVIL_MODELS_ROOT", str(tmp_path))

    plan = vlm_plan(str(d), fetch_remote=False)
    assert plan.pattern.value == "vlm_sft"
    out = run_vlm_sft(base_model=str(d), steps=2, fetch_remote=False)
    assert out.steps_run == 2


def test_grpo_group_advantages_and_loop():
    assert group_advantages([1.0, 0.0, 1.0, 0.0]) == [0.5, -0.5, 0.5, -0.5]
    out = run_grpo(base_model="toy/lm", steps=2, group_size=2)
    assert out.steps_run == 2
    assert out.plan.loss_fn == "importance_sampling"
    assert len(out.mean_reward) == 2


def test_grpo_loop_local_backend():
    """Full GRPO loop against LocalBackend — sample → reward → group advantages
    → IS forward_backward → optim_step, all with real logprobs and grads."""
    out = run_grpo(
        base_model="hf-internal-testing/tiny-random-gpt2",
        prompts=[[10, 11, 12, 13]],
        group_size=2,
        steps=2,
        endpoint="local://",
    )
    assert out.steps_run == 2
    assert out.plan.loss_fn == "importance_sampling"
    assert len(out.losses) == 2 and all(x == x for x in out.losses)  # finite
    assert len(out.mean_reward) == 2
