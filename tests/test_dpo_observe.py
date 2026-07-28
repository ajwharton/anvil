"""DPO preference recipe + metrics.jsonl observe SSOT."""

from __future__ import annotations

from anvil.observe.metrics import METRICS_FILENAME, read_jsonl
from anvil.recipes.dpo import PreferencePair, run_dpo


def test_run_dpo_emits_metrics_and_length_bias(tmp_path):
    run_dir = tmp_path / "dpo-run"
    pairs = [
        PreferencePair(prompt="hi", preferred="ok", rejected="a much longer bad answer here"),
        PreferencePair(prompt="2+2", preferred="4", rejected="maybe four or five"),
    ]
    res = run_dpo(
        endpoint="fake://",
        pairs=pairs,
        steps=5,
        run_dir=str(run_dir),
        early_stop=False,
    )
    assert res.steps_run == 5
    assert res.mean_length_bias is not None
    # preferred shorter than rejected → negative length_bias
    assert res.mean_length_bias < 0
    steps = read_jsonl(run_dir / METRICS_FILENAME)
    assert len(steps) == 5
    assert steps[0]["job"] == "dpo"
    assert steps[0]["n_pairs"] == 2
    assert "length_bias" in steps[0]
    assert steps[0]["loss"] is not None


def test_run_dpo_production_early_stop(tmp_path):
    res = run_dpo(
        endpoint="fake://",
        steps=200,
        run_dir=str(tmp_path / "dpo-es"),
        early_stop_mode="production",
        early_stop_patience=15,
        stop_on_southward=False,  # isolate plateau stop
    )
    assert res.steps_run < 200
    assert res.early_stop_reason is not None
    assert "plateau" in res.early_stop_reason
    events = [
        r
        for r in read_jsonl(tmp_path / "dpo-es" / METRICS_FILENAME)
        if r.get("event") == "early_stop"
    ]
    assert events
    assert events[0].get("job") == "dpo"


def test_run_dpo_held_out_probes(tmp_path):
    from anvil.observe.metrics import PROBES_FILENAME

    train = [
        PreferencePair(prompt="hi", preferred="ok", rejected="long bad answer text"),
    ]
    hold = [
        PreferencePair(prompt="2+2?", preferred="4", rejected="five is bigger"),
    ]
    res = run_dpo(
        endpoint="fake://",
        pairs=train,
        probes=hold,
        probe_every=1,
        steps=3,
        run_dir=str(tmp_path / "dpo-pr"),
        early_stop=False,
        stop_on_southward=False,
    )
    assert res.n_probe_records == 3
    probes = read_jsonl(tmp_path / "dpo-pr" / PROBES_FILENAME)
    assert len(probes) == 3
    assert probes[0]["job"] == "dpo"
    assert probes[0]["target"] == "4"
