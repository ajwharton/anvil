"""P2.5 observability: metrics writer/reader, tripwire, run_grpo emission,
and the web /api/observe endpoints + SSE stream."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from anvil.observe.metrics import (
    METRICS_FILENAME,
    PROBES_FILENAME,
    RunMetricsWriter,
    advantage_collapsed,
    read_jsonl,
)
from anvil.recipes.grpo import run_grpo


def test_writer_roundtrip_and_tail(tmp_path):
    w = RunMetricsWriter(tmp_path / "run-x")
    w.log_step(
        step=0, reward_mean=0.5, reward_std=0.4, group_reward_std_mean=0.3,
        loss=-0.001, n_datums=8, fb_metrics={"mean_ratio": 1.0}, wall_time_s=0.1,
    )
    w.log_step(
        step=1, reward_mean=0.75, reward_std=0.2, group_reward_std_mean=0.1,
        loss=-0.002, n_datums=8,
    )
    path = tmp_path / "run-x" / METRICS_FILENAME
    recs = read_jsonl(path)
    assert len(recs) == 2
    assert recs[0]["schema_version"] == 1
    assert recs[0]["is_mean_ratio"] == 1.0
    assert recs[1]["is_mean_ratio"] is None
    assert [r["step"] for r in read_jsonl(path, tail=1)] == [1]


def test_probe_records(tmp_path):
    w = RunMetricsWriter(tmp_path)
    w.log_probe(step=0, probe_idx=2, tokens=[1, 2, 3], text="abc", reward=1.0)
    (rec,) = read_jsonl(tmp_path / PROBES_FILENAME)
    assert rec["probe_idx"] == 2
    assert rec["tokens"] == [1, 2, 3]
    assert rec["text"] == "abc"
    assert rec["reward"] == 1.0


def test_advantage_collapsed_tripwire():
    assert advantage_collapsed({"group_reward_std_mean": 0.0})
    assert not advantage_collapsed({"group_reward_std_mean": 0.5})
    assert not advantage_collapsed({})  # missing key -> no trip


def test_run_grpo_emits_metrics_and_probes(tmp_path):
    run_dir = tmp_path / "grpo-run"
    res = run_grpo(
        endpoint="fake://",
        steps=3,
        group_size=4,
        run_dir=str(run_dir),
        probes=[list(range(10, 18)), list(range(20, 28))],
        probe_every=1,
        detokenize=lambda toks: f"<{len(toks)} toks>",
    )
    assert res.steps_run == 3
    steps = read_jsonl(run_dir / METRICS_FILENAME)
    assert [s["step"] for s in steps] == [0, 1, 2]
    for s in steps:
        assert 0.0 <= s["reward_mean"] <= 1.0
        assert s["group_reward_std_mean"] >= 0.0
        assert s["n_datums"] > 0
        assert s["fb"]  # backend forward_backward metrics passed through
        assert s["wall_time_s"] is not None
    probes = read_jsonl(run_dir / PROBES_FILENAME)
    assert len(probes) == 3 * 2  # 3 steps x 2 probes
    assert {p["probe_idx"] for p in probes} == {0, 1}
    assert all(p["text"] is not None for p in probes)
    # probes are scored too — the reward-hacking signature needs both signals
    assert all(p["reward"] is not None for p in probes)


def test_run_grpo_constant_reward_shows_advantage_collapse(tmp_path):
    """A constant reward_fn zeroes within-group std — the tripwire sees it."""
    run_dir = tmp_path / "collapse"
    run_grpo(
        endpoint="fake://",
        steps=2,
        group_size=4,
        reward_fn=lambda _text, _toks: 1.0,
        run_dir=str(run_dir),
    )
    steps = read_jsonl(run_dir / METRICS_FILENAME)
    assert len(steps) == 2
    assert all(advantage_collapsed(s) for s in steps)


# --- web endpoints ----------------------------------------------------------


def _seed_run(root, run_id="r1"):
    w = RunMetricsWriter(root / run_id)
    w.log_step(
        step=0, reward_mean=0.5, reward_std=0.1, group_reward_std_mean=0.2,
        loss=-0.001, n_datums=4,
    )
    w.log_probe(step=0, probe_idx=0, tokens=[5, 6], text="hi", reward=1.0)


def test_observe_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_OBSERVE_ROOT", str(tmp_path))
    _seed_run(tmp_path)
    from anvil.web.app import create_app

    client = TestClient(create_app())

    r = client.get("/api/observe")
    assert r.status_code == 200
    assert r.json()["runs"] == ["r1"]

    r = client.get("/api/observe/r1/metrics")
    assert r.status_code == 200
    (rec,) = r.json()["metrics"]
    assert rec["step"] == 0
    assert rec["reward_mean"] == 0.5

    r = client.get("/api/observe/r1/probes")
    assert r.status_code == 200
    assert r.json()["probes"][0]["text"] == "hi"

    assert client.get("/api/observe/nope/metrics").status_code == 404
    assert client.get("/api/observe/..%2F..%2Fetc/metrics").status_code in (400, 404, 422)

    r = client.get("/observe/r1")
    assert r.status_code == 200
    assert "anvil observe" in r.text
    assert "ADVANTAGE COLLAPSED" in r.text  # tripwire banner wired into the page


def test_observe_metrics_stream(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_OBSERVE_ROOT", str(tmp_path))
    _seed_run(tmp_path)
    from anvil.web.app import create_app

    client = TestClient(create_app())
    with client.stream("GET", "/api/observe/r1/metrics/stream") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = r.iter_lines()
        first = next(lines)
        assert first.startswith("data: ")
        payload = json.loads(first[len("data: "):])
        assert payload["step"] == 0
        assert payload["group_reward_std_mean"] == 0.2
