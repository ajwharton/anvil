"""Early-stop when GRPO signal is dead (ceiling / floor / collapse)."""

from __future__ import annotations

from anvil.observe.metrics import METRICS_FILENAME, read_jsonl
from anvil.recipes.grpo import (
    classify_dead_step,
    early_stop_reason,
    run_grpo,
)


def test_classify_dead_step_labels():
    assert classify_dead_step(reward_mean=1.0, group_reward_std_mean=0.0) == "ceiling"
    assert classify_dead_step(reward_mean=0.0, group_reward_std_mean=0.0) == "floor"
    assert classify_dead_step(reward_mean=0.5, group_reward_std_mean=0.0) == "collapsed"
    assert classify_dead_step(reward_mean=1.0, group_reward_std_mean=0.2) is None


def test_early_stop_reason_patience():
    assert early_stop_reason([None, "ceiling"], patience=3) is None
    assert early_stop_reason(["ceiling"] * 3, patience=3) == "ceiling_x3"
    assert early_stop_reason(["floor", "floor", "ceiling"], patience=3) is None
    assert early_stop_reason(["floor"] * 8, patience=8) == "floor_x8"


def test_run_grpo_early_stops_on_constant_reward(tmp_path):
    """Constant reward → zero group std → abandon after patience, not full steps."""
    run_dir = tmp_path / "early"
    res = run_grpo(
        endpoint="fake://",
        steps=50,
        group_size=4,
        run_dir=str(run_dir),
        reward_fn=lambda _t, _toks: 1.0,
        early_stop=True,
        early_stop_patience=5,
    )
    assert res.early_stop_reason == "ceiling_x5"
    assert res.steps_run == 5  # not 50
    events = [
        r
        for r in read_jsonl(run_dir / METRICS_FILENAME)
        if r.get("type") == "event" and r.get("event") == "early_stop"
    ]
    assert len(events) == 1
    assert events[0]["reason"] == "ceiling_x5"
    assert events[0]["step"] == 4


def test_run_grpo_no_early_stop_runs_full(tmp_path):
    res = run_grpo(
        endpoint="fake://",
        steps=6,
        group_size=4,
        run_dir=str(tmp_path / "full"),
        reward_fn=lambda _t, _toks: 1.0,
        early_stop=False,
    )
    assert res.early_stop_reason is None
    assert res.steps_run == 6
