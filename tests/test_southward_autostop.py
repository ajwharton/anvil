"""Mid-train auto-stop when southward cliffs fire."""

from __future__ import annotations

from anvil.observe.metrics import METRICS_FILENAME, RunMetricsWriter, read_jsonl
from anvil.observe.southward import maybe_stop_on_southward
from anvil.protocol.messages import Example, Message, TextPart
from anvil.recipes.dpo import PreferencePair, run_dpo
from anvil.recipes.grpo import run_grpo
from anvil.recipes.sft import run_sft


def test_maybe_stop_on_southward_after_collapse(tmp_path):
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
    assert "advantage_collapse" in reason
    events = [r for r in read_jsonl(tmp_path / "metrics.jsonl") if r.get("event") == "southward"]
    assert events


def test_maybe_stop_skips_before_min_steps(tmp_path):
    w = RunMetricsWriter(tmp_path)
    for i in range(3):
        w.log_step(
            step=i,
            reward_mean=0.5,
            reward_std=0.0,
            group_reward_std_mean=0.0,
            loss=0.1,
            n_datums=4,
        )
    assert maybe_stop_on_southward(tmp_path, step=1, min_steps=5) is None


def test_grpo_stops_on_southward_not_classic_early_stop(tmp_path):
    """Constant reward collapses advantage; with classic early-stop off, southward must fire."""
    res = run_grpo(
        endpoint="fake://",
        steps=30,
        group_size=4,
        run_dir=str(tmp_path / "grpo-sw"),
        reward_fn=lambda _t, _toks: 1.0,
        early_stop=False,  # isolate southward path
        stop_on_southward=True,
        southward_min_steps=3,
        probes=[[1, 2, 3]],
        probe_every=1,
    )
    assert res.steps_run < 30
    assert res.early_stop_reason is not None
    assert res.early_stop_reason.startswith("southward:")
    assert "advantage_collapse" in res.early_stop_reason
    events = [
        r
        for r in read_jsonl(tmp_path / "grpo-sw" / METRICS_FILENAME)
        if r.get("event") == "early_stop"
    ]
    assert events
    assert events[0].get("trigger") == "southward"


def test_sft_stop_on_southward_no_false_positive(tmp_path):
    """Healthy short run with min_steps above budget must not stop."""
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
    assert res.early_stop_reason is None


def test_sft_stops_when_preseeded_loss_flat_probes_down(tmp_path):
    """Mid-train hook sees disk cliffs (flat loss + probe regression) and aborts."""
    run_dir = tmp_path / "sft-cliff"
    w = RunMetricsWriter(run_dir)
    # Flat loss + falling probe rewards → loss_flat_probes_down cliff
    for i in range(10):
        w.log_sft_step(step=i, loss=0.5, n_datums=1, job="sft")
        w.log_probe(
            step=i,
            probe_idx=0,
            tokens=(1, 2),
            text="x",
            reward=1.0 - i * 0.08,
            target="x",
            job="sft",
        )
    reason = maybe_stop_on_southward(run_dir, step=9, min_steps=5)
    assert reason is not None
    assert reason.startswith("southward:")
    assert "loss_flat_probes_down" in reason or "probe_regression" in reason

    # Wire path: continue SFT with stop_on_southward; preseeded cliffs should abort early
    ex = Example(
        messages=(
            Message(role="user", content=(TextPart(text="q"),)),
            Message(role="assistant", content=(TextPart(text="a"),)),
        )
    )
    res = run_sft(
        endpoint="fake://",
        examples=[ex],
        steps=20,
        run_dir=str(run_dir),
        early_stop=False,
        stop_on_southward=True,
        southward_min_steps=1,
    )
    assert res.steps_run < 20
    assert res.early_stop_reason is not None
    assert res.early_stop_reason.startswith("southward:")


def test_dpo_stops_on_length_bias_spike(tmp_path):
    """Preferred much longer than rejected → length_bias cliff → auto-stop."""
    # Large positive length_bias: preferred tokens >> rejected
    long_pref = "answer " * 40
    pairs = [
        PreferencePair(prompt="hi", preferred=long_pref, rejected="no"),
        PreferencePair(prompt="yo", preferred=long_pref + "x", rejected="nope"),
    ]
    res = run_dpo(
        endpoint="fake://",
        pairs=pairs,
        steps=30,
        run_dir=str(tmp_path / "dpo-sw"),
        early_stop=False,
        stop_on_southward=True,
        southward_min_steps=5,
    )
    assert res.mean_length_bias is not None
    assert res.mean_length_bias >= 8.0  # detector threshold
    assert res.steps_run < 30
    assert res.early_stop_reason is not None
    assert res.early_stop_reason.startswith("southward:")
    assert "length_bias_spike" in res.early_stop_reason
