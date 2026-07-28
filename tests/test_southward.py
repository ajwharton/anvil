"""Southward-turn detectors."""

from __future__ import annotations

from anvil.observe.metrics import RunMetricsWriter, read_jsonl
from anvil.observe.southward import (
    scan_and_log,
    scan_records,
)


def test_advantage_collapse_flag():
    metrics = [
        {
            "type": "step",
            "step": i,
            "reward_mean": 0.5,
            "group_reward_std_mean": 0.0,
            "loss": 0.1,
        }
        for i in range(5)
    ]
    rep = scan_records(metrics, [])
    assert "advantage_collapse" in rep.names
    assert not rep.ok


def test_reward_up_probes_down():
    metrics = []
    probes = []
    for i in range(10):
        metrics.append(
            {
                "type": "step",
                "step": i,
                "reward_mean": 0.2 + i * 0.05,
                "group_reward_std_mean": 0.2,
                "loss": 0.1,
            }
        )
        probes.append({"type": "probe", "step": i, "probe_idx": 0, "reward": 1.0 - i * 0.08})
    rep = scan_records(metrics, probes)
    assert "reward_up_probes_down" in rep.names


def test_probe_regression():
    probes = [{"step": i, "reward": 1.0 if i < 5 else 0.1} for i in range(12)]
    rep = scan_records([{"type": "step", "step": 0, "loss": 1.0}], probes)
    assert "probe_regression" in rep.names


def test_length_bias_spike_dpo():
    metrics = [
        {"type": "step", "job": "dpo", "step": i, "loss": 0.5, "length_bias": 20.0}
        for i in range(8)
    ]
    rep = scan_records(metrics, [])
    assert "length_bias_spike" in rep.names


def test_loss_flat_probes_down():
    metrics = [
        {"type": "step", "job": "vlm_sft", "step": i, "loss": 0.02}
        for i in range(20)
    ]
    probes = [
        {"step": i, "reward": 1.0 if i < 6 else 0.0}
        for i in range(18)
    ]
    rep = scan_records(metrics, probes)
    assert "loss_flat_probes_down" in rep.names


def test_scan_and_log_writes_events(tmp_path):
    w = RunMetricsWriter(tmp_path)
    for i in range(5):
        w.log_step(
            step=i,
            reward_mean=0.5,
            reward_std=0.0,
            group_reward_std_mean=0.0,
            loss=0.1,
            n_datums=4,
        )
    rep = scan_and_log(tmp_path)
    assert not rep.ok
    events = [r for r in read_jsonl(tmp_path / "metrics.jsonl") if r.get("event") == "southward"]
    assert events
    assert events[0]["reason"] == "advantage_collapse"
