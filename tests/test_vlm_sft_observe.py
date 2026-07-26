"""P3.6 / 3.C: run_vlm_sft writes metrics.jsonl; /observe serves SFT runs."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from anvil.observe.metrics import METRICS_FILENAME, read_jsonl
from anvil.recipes.vlm_sft import run_vlm_sft


def test_run_vlm_sft_emits_metrics_jsonl(tmp_path):
    run_dir = tmp_path / "vlm-obs"
    res = run_vlm_sft(
        endpoint="fake://",
        steps=3,
        fetch_remote=False,
        run_dir=str(run_dir),
    )
    assert res.steps_run == 3
    assert res.run_dir == str(run_dir)
    steps = read_jsonl(run_dir / METRICS_FILENAME)
    assert [s["step"] for s in steps] == [0, 1, 2]
    for s in steps:
        assert s["schema_version"] == 1
        assert s["type"] == "step"
        assert s["job"] == "vlm_sft"
        assert isinstance(s["loss"], float)
        assert s["n_datums"] >= 1
        assert s["n_image_refs"] >= 1  # toy_vlm_examples has ImagePart
        assert s["wall_time_s"] is not None
        assert s["wall_time_s"] >= 0.0
        # no GRPO-only fields required
        assert "reward_mean" not in s


def test_run_vlm_sft_without_run_dir_is_silent(tmp_path):
    res = run_vlm_sft(endpoint="fake://", steps=1, fetch_remote=False)
    assert res.steps_run == 1
    assert res.run_dir is None
    assert not list(tmp_path.glob("**/metrics.jsonl"))


def test_observe_serves_vlm_sft_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_OBSERVE_ROOT", str(tmp_path))
    run_id = "vlm-sft-demo"
    run_vlm_sft(
        endpoint="fake://",
        steps=2,
        fetch_remote=False,
        run_dir=str(tmp_path / run_id),
    )
    from anvil.web.app import create_app

    client = TestClient(create_app())

    r = client.get("/api/observe")
    assert r.status_code == 200
    body = r.json()
    assert any(x["run_id"] == run_id for x in body["runs"])

    r = client.get(f"/api/observe/{run_id}/metrics")
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    assert len(metrics) == 2
    assert metrics[0]["job"] == "vlm_sft"
    assert metrics[0]["n_image_refs"] >= 1

    r = client.get(f"/observe/{run_id}")
    assert r.status_code == 200
    assert "anvil observe" in r.text
    # SFT chart path: loss series + no crash on missing reward
    assert "isSft" in r.text or "loss / step" in r.text or "job" in r.text
    assert "n_image_refs" in r.text

    r = client.get("/observe")
    assert r.status_code == 200
    assert "live runs" in r.text
    assert run_id in r.text
    assert "vlm_sft" in r.text or "loss=" in r.text

    with client.stream(
        "GET", f"/api/observe/{run_id}/metrics/stream?once=true"
    ) as stream:
        assert stream.status_code == 200
        data_lines = [ln for ln in stream.iter_lines() if ln.startswith("data: ")]
        assert data_lines
        payload = json.loads(data_lines[0][len("data: "):])
        assert payload["job"] == "vlm_sft"
        assert "loss" in payload
