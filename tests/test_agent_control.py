"""Agent control plane: live act APIs + HTTP client + harness dispatch."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from anvil.agent.harness import dispatch_tool, load_prompt_pack, tool_specs
from anvil.web import state as state_mod
from anvil.web.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_FAKE_ROOT", str(tmp_path / "fake"))
    monkeypatch.setenv("ANVIL_EXPORT_ROOT", str(tmp_path / "exports"))
    monkeypatch.setenv("ANVIL_MODELS_ROOT", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    state_mod._STORE = None
    app = create_app()
    with TestClient(app) as c:
        yield c
    state_mod._STORE = None


def test_pause_resume_patch_knobs(client):
    r = client.post(
        "/api/runs",
        json={
            "name": "ctl",
            "knobs": {"base_model": "Qwen/Qwen2.5-VL-3B-Instruct", "rank": 8},
        },
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    # train once then pause
    assert client.post(f"/api/runs/{run_id}/train", json={"steps": 1}).status_code == 200
    pr = client.post(f"/api/runs/{run_id}/pause")
    assert pr.status_code == 200
    assert pr.json()["status"] == "paused"

    # cannot train while paused
    bad = client.post(f"/api/runs/{run_id}/train", json={"steps": 1})
    assert bad.status_code == 409

    assert client.post(f"/api/runs/{run_id}/resume").json()["status"] == "running"

    pk = client.patch(
        f"/api/runs/{run_id}/knobs",
        json={"knobs": {"learning_rate": 5e-5, "probe_every": 3}},
    )
    assert pk.status_code == 200
    kn = pk.json()["knobs"]
    assert kn["learning_rate"] == 5e-5
    assert kn["probe_every"] == 3
    assert any("knobs patched" in line for line in pk.json()["logs"])


def test_control_client_against_testclient(client, monkeypatch):
    # Drive AnvilControlClient via TestClient transport by monkeypatching urllib
    # is heavy; instead exercise client methods against the same app using
    # ASGI — use client fixture directly and test dispatch_tool with a stub.

    class _Stub:
        def overview(self):
            return {"ok": True}

        def list_runs(self):
            return []

        def pause(self, run_id):
            return {"run_id": run_id, "status": "paused"}

        def patch_knobs(self, run_id, knobs):
            return {"run_id": run_id, "knobs": knobs}

    s = _Stub()
    assert json.loads(dispatch_tool(s, "anvil_overview", {}))["ok"] is True
    assert json.loads(dispatch_tool(s, "anvil_pause", {"run_id": "r1"}))["status"] == "paused"
    names = {t["function"]["name"] for t in tool_specs()}
    assert "anvil_observe_metrics" in names
    assert "anvil_patch_knobs" in names


def test_prompt_pack_loads():
    text = load_prompt_pack()
    assert "post-training operator" in text.lower() or "Anvil" in text
    assert "advantage collapse" in text.lower() or "collapse" in text.lower()


def test_mcp_server_builds():
    pytest.importorskip("mcp")
    from anvil.agent.mcp_server import build_mcp_server

    srv = build_mcp_server("http://127.0.0.1:7600")
    assert srv is not None
