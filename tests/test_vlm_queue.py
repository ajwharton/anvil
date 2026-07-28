"""VLM/SFT multi-stage queue — early-stop advance (roadmap 3.C)."""

from __future__ import annotations

from anvil.observe.metrics import read_jsonl
from anvil.protocol.messages import Example, Message, TextPart
from anvil.recipes.vlm_queue import VLMQueueRecipe, VLMStage, run_vlm_queue
from anvil.recipes.vlm_sft import run_vlm_sft


def _ex(u: str, a: str) -> Example:
    return Example(
        messages=(
            Message(role="user", content=(TextPart(text=u),)),
            Message(role="assistant", content=(TextPart(text=a),)),
        )
    )


def test_run_vlm_sft_production_early_stop(tmp_path):
    """Vision job path inherits SFT production early-stop (job=vlm_sft)."""
    res = run_vlm_sft(
        endpoint="fake://",
        examples=[_ex("grasp?", "yes")],
        steps=200,
        fetch_remote=False,
        run_dir=str(tmp_path / "vlm-es"),
        early_stop_mode="production",
        early_stop_patience=15,
        stop_on_southward=False,
    )
    assert res.steps_run < 200
    assert res.early_stop_reason is not None
    assert "plateau" in res.early_stop_reason
    steps = read_jsonl(tmp_path / "vlm-es" / "metrics.jsonl")
    assert any(s.get("job") == "vlm_sft" for s in steps if s.get("type") == "step")


def test_vlm_queue_advances_on_plateau(tmp_path):
    recipe = VLMQueueRecipe(
        id="vlm-cur",
        name="two-stage vision",
        stages=(
            VLMStage(
                id="stage-a",
                examples=(_ex("a?", "1"),),
                max_steps=200,
                early_stop_patience=12,
                job="vlm_sft",
            ),
            VLMStage(
                id="stage-b",
                examples=(_ex("b?", "2"),),
                max_steps=200,
                early_stop_patience=12,
                job="vlm_sft",
            ),
        ),
        early_stop_patience=12,
        advance_on=("loss_plateau",),
        advance_on_budget=True,
    )
    res = run_vlm_queue(
        recipe,
        endpoint="fake://",
        run_dir=str(tmp_path / "vq"),
        fetch_remote=False,
    )
    assert res.stages_run == 2
    assert res.stages[0].advanced is True
    assert res.stages[0].result.early_stop_reason is not None
    assert "plateau" in (res.stages[0].result.early_stop_reason or "")
    assert res.stages[1].advanced is False
    assert res.adapter_id is not None
    # same adapter across stages
    assert res.stages[0].result.adapter_id == res.stages[1].result.adapter_id

    qevents = [
        r
        for r in read_jsonl(tmp_path / "vq" / "metrics.jsonl")
        if r.get("event") in {"vlm_stage_start", "vlm_stage_end"}
    ]
    assert any(e.get("event") == "vlm_stage_start" for e in qevents)
    assert any(e.get("advanced") is True for e in qevents if e.get("event") == "vlm_stage_end")
