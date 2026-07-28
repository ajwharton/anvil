"""SFT/VLM early-stop (production vs calibration)."""

from __future__ import annotations

from anvil.observe.metrics import read_jsonl
from anvil.protocol.messages import Example, Message, TextPart
from anvil.recipes.sft import (
    DEFAULT_SFT_EARLY_STOP_PATIENCE,
    run_sft,
    sft_early_stop_reason,
    sft_loss_improved,
)


def test_sft_loss_improved_relative():
    assert sft_loss_improved(1.0, 0.98, rel_eps=0.01)  # 2% drop
    assert not sft_loss_improved(1.0, 0.995, rel_eps=0.01)  # 0.5% drop


def test_sft_early_stop_reason_patience():
    # improving then flat
    losses = [1.0, 0.5, 0.4, 0.4, 0.4, 0.4]
    assert sft_early_stop_reason(losses, patience=3, rel_eps=0.01) is not None
    assert sft_early_stop_reason(losses, patience=10, rel_eps=0.01) is None
    # still improving at end
    rising = [1.0, 0.8, 0.6, 0.5, 0.4]
    assert sft_early_stop_reason(rising, patience=3, rel_eps=0.01) is None


def test_run_sft_production_early_stops(tmp_path):
    """Toy CE plateaus quickly; production mode should not burn full budget."""
    run_dir = tmp_path / "es"
    ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text="hi"),)),
            Message(role="assistant", content=(TextPart(text="yo"),)),
        )
    )
    res = run_sft(
        endpoint="fake://",
        examples=[ex],
        steps=500,
        run_dir=str(run_dir),
        early_stop_mode="production",
        early_stop_patience=15,
        early_stop_rel_eps=0.01,
    )
    assert res.steps_run < 500
    assert res.early_stop_reason is not None
    assert "loss_plateau" in res.early_stop_reason
    events = [
        r
        for r in read_jsonl(run_dir / "metrics.jsonl")
        if r.get("type") == "event" and r.get("event") == "early_stop"
    ]
    assert events
    assert events[0].get("mode") == "production"


def test_run_sft_calibration_runs_full_budget():
    ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text="a"),)),
            Message(role="assistant", content=(TextPart(text="b"),)),
        )
    )
    res = run_sft(
        endpoint="fake://",
        examples=[ex],
        steps=25,
        early_stop_mode="calibration",
    )
    # calibration patience is huge → full steps
    assert res.steps_run == 25
    assert res.early_stop_reason is None


def test_run_sft_no_early_stop_flag():
    ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text="a"),)),
            Message(role="assistant", content=(TextPart(text="b"),)),
        )
    )
    res = run_sft(
        endpoint="fake://",
        examples=[ex],
        steps=30,
        early_stop=False,
        early_stop_mode="production",
        early_stop_patience=5,
    )
    assert res.steps_run == 30


def test_default_patience_constant():
    assert DEFAULT_SFT_EARLY_STOP_PATIENCE >= 20
