"""Anvil web control-plane API (fake backend)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from anvil.web.app import create_app
from anvil.web import state as state_mod


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_FAKE_ROOT", str(tmp_path / "fake"))
    monkeypatch.setenv("ANVIL_EXPORT_ROOT", str(tmp_path / "exports"))
    monkeypatch.setenv("ANVIL_MODELS_ROOT", str(tmp_path / "models"))
    (tmp_path / "models" / "Qwen2.5-VL-3B-Instruct").mkdir(parents=True)
    (tmp_path / "models" / "Qwen2.5-VL-3B-Instruct" / "config.json").write_text("{}", encoding="utf-8")
    state_mod._STORE = None
    app = create_app()
    with TestClient(app) as c:
        yield c
    state_mod._STORE = None


def test_health_and_defaults(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    d = client.get("/api/defaults").json()
    assert "cross_entropy" in d["loss_choices"]
    assert d["knobs"]["rank"] == 32


def test_create_train_export_flow(client):
    r = client.post(
        "/api/runs",
        json={
            "name": "smoke",
            "knobs": {
                "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
                "rank": 8,
                "max_steps": 20,
                "batch_size": 2,
                "seq_len": 16,
            },
        },
    )
    assert r.status_code == 200
    run = r.json()
    rid = run["run_id"]
    assert run["status"] == "created"

    r = client.post(f"/api/runs/{rid}/train", json={"steps": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["step"] == 3
    assert body["last_loss"] is not None
    assert len(body["history"]) == 3

    r = client.post(f"/api/runs/{rid}/sample", json={})
    assert r.status_code == 200
    assert r.json()["n_tokens"] > 0

    r = client.post(f"/api/runs/{rid}/export", json={"format": "peft"})
    assert r.status_code == 200
    assert "path" in r.json()

    ov = client.get("/api/overview").json()
    assert ov["total_runs"] >= 1
    assert any(m["name"] == "Qwen2.5-VL-3B-Instruct" for m in ov["models"])


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Anvil" in r.text
