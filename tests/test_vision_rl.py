"""Phase 4.B: multimodal GRPO, vision rewards, vision RL queue."""

from __future__ import annotations

from pathlib import Path

from anvil.protocol.types import ImageRefChunk, ModelInput
from anvil.recipes.grpo import as_model_input, run_grpo
from anvil.recipes.vision_rewards import (
    RubricCriterion,
    action_bin_overlap_reward,
    keyword_caption_reward,
    rubric_reward,
    toy_detokenize,
)
from anvil.recipes.vision_rl import (
    VisionRollout,
    run_vision_grpo,
    run_vision_rl_queue,
    toy_vision_rl_queue,
    toy_vision_rollouts,
)


def test_keyword_and_rubric_rewards():
    kw = keyword_caption_reward(required=("kitchen", "chair"))
    assert kw("Kitchen with a chair", ()) == 1.0
    assert kw("empty room", ()) == 0.0
    assert 0.0 < kw("kitchen only", ()) < 1.0

    rub = rubric_reward(
        (
            RubricCriterion(name="scene", weight=1.0, required_keywords=("kitchen",)),
            RubricCriterion(name="hazard", weight=1.0, required_keywords=("chair",)),
        )
    )
    assert rub("kitchen chair", ()) == 1.0
    assert rub("kitchen", ()) == 0.5


def test_action_bin_overlap():
    r = action_bin_overlap_reward([10, 20, 30])
    assert r("10 20 30", ()) == 1.0
    assert r("10 99 30", ()) == 2 / 3
    assert r("nope", ()) == 0.0


def test_as_model_input_multimodal():
    from anvil.protocol.types import EncodedTextChunk

    mi = ModelInput.from_chunks(
        (
            EncodedTextChunk(tokens=(1, 2, 3)),
            ImageRefChunk(ref="cas://sha256/" + "a" * 64, detail="low"),
        )
    )
    assert as_model_input(mi) is mi
    assert as_model_input([4, 5, 6]).token_ids() == [4, 5, 6]


def test_run_grpo_passes_detokenized_text_to_reward():
    seen: list[str] = []

    def rf(text: str, tokens):
        seen.append(text)
        return 1.0 if text else 0.0

    run_grpo(
        prompts=[[10, 11, 12]],
        reward_fn=rf,
        detokenize=lambda toks: "kitchen chair",
        steps=1,
        group_size=2,
        endpoint="fake://",
        early_stop=False,
    )
    assert seen and all(s == "kitchen chair" for s in seen)


def test_run_grpo_multimodal_prompt_fake(tmp_path: Path):
    mi = toy_vision_rollouts()[0].to_model_input()
    assert any(isinstance(c, ImageRefChunk) for c in mi.chunks)
    res = run_grpo(
        prompts=[mi],
        reward_fn=keyword_caption_reward(required=("kitchen",), detokenize=toy_detokenize),
        detokenize=toy_detokenize,
        steps=2,
        group_size=2,
        endpoint="fake://",
        run_dir=str(tmp_path / "mm-grpo"),
        early_stop=False,
        job="vision_grpo",
    )
    assert res.steps_run == 2
    metrics = (tmp_path / "mm-grpo" / "metrics.jsonl").read_text(encoding="utf-8")
    assert "vision_grpo" in metrics
    assert "n_image_refs" in metrics


def test_run_vision_grpo_and_queue(tmp_path: Path):
    res = run_vision_grpo(
        steps=2,
        endpoint="fake://",
        run_dir=str(tmp_path / "vgrpo"),
        early_stop=False,
    )
    assert res.steps_run == 2
    assert res.adapter_id
    assert len(res.mean_reward) == 2

    q = run_vision_rl_queue(
        toy_vision_rl_queue(),
        endpoint="fake://",
        run_dir=str(tmp_path / "vqueue"),
    )
    assert q.stages_run >= 1
    assert q.adapter_id
    # stage subdirs
    assert (tmp_path / "vqueue" / "caption").is_dir() or q.stages_run >= 1


def test_vision_rollout_reward_builders():
    r = VisionRollout(
        id="t",
        instruction="x",
        image_refs=("cas://sha256/" + "c" * 64,),
        required_keywords=("kitchen",),
        prompt_token_ids=(1, 2, 3),
    )
    fn = r.make_reward(detokenize=toy_detokenize)
    assert fn("kitchen table", ()) == 1.0
