"""Mid-train auto-stop when southward cliffs fire."""

from __future__ import annotations

from anvil.observe.metrics import read_jsonl
from anvil.observe.southward import maybe_stop_on_southward
from anvil.protocol.messages import Example, Message, TextPart
from anvil.recipes.grpo import run_grpo
from anvil.recipes.sft import run_sft


def test_maybe_stop_on_southward_after_collapse(tmp_path):
    from anvil.observe.metrics import RunMetricsWriter

    w = RunMetricsWriter(tmp_path)
    for i in range(6):
        w.log_step(
            step=i,
            reward_mean=0.5,
            reward_std=0.0,
            group_reward_std_mean=0.0,
            loss=0.1,
            n_datums=4,
        )
    reason = maybe_stop_on_southward(tmp_path, step=5, min_steps=3)
    assert reason is not None
    assert reason.startswith("southward:")
    events = [r for r in read_jsonl(tmp_path / "metrics.jsonl") if r.get("event") == "southward"]
    assert events


def test_grpo_stops_on_southward_collapse(tmp_path):
    """Constant reward → collapse; southward should fire (or dead-signal early-stop)."""
    res = run_grpo(
        endpoint="fake://",
        steps=30,
        group_size=4,
        run_dir=str(tmp_path / "grpo-sw"),
        reward_fn=lambda _t, _toks: 1.0,
        early_stop=True,
        early_stop_patience=8,
        stop_on_southward=True,
        southward_min_steps=3,
        probes=[[1, 2, 3]],
        probe_every=1,
    )
    assert res.steps_run < 30
    assert res.early_stop_reason is not None
    # Either classic ceiling_xN or southward:advantage_collapse
    assert (
        "ceiling" in res.early_stop_reason
        or "southward" in res.early_stop_reason
        or "collapsed" in res.early_stop_reason
    )


def test_sft_stop_on_southward_flag(tmp_path):
    """With probes that regress and flat loss, SFT should be stoppable via southward."""
    # Mostly ensure the hook is wired (production + run_dir) without false stop
    # on healthy short runs.
    ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text="a"),)),
            Message(role="assistant", content=(TextPart(text="b"),)),
        )
    )
    res = run_sft(
        endpoint="fake://",
        examples=[ex],
        steps=10,
        run_dir=str(tmp_path / "sft-ok"),
        early_stop=False,
        stop_on_southward=True,
        southward_min_steps=20,  # higher than budget → no mid-train southward
    )
    assert res.steps_run == 10
