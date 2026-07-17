"""`anvil serve` + RemoteBackend — HTTP transport for the four verbs.

Runs against the fake backend (no torch); the wire codec and client/server
symmetry are what's under test. FastAPI TestClient stands in for the network.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from anvil.backends.fake import FakeBackend  # noqa: E402
from anvil.client.remote import RemoteBackend, RemoteBackendError  # noqa: E402
from anvil.client.service import resolve_backend  # noqa: E402
from anvil.protocol.types import (  # noqa: E402
    AdamParams,
    Datum,
    ExportFormat,
    ModelInput,
    SamplingParams,
)
from anvil.serve.app import create_app  # noqa: E402

TOKENS = list(range(10, 30))


def _datum() -> Datum:
    return Datum(
        model_input=ModelInput.from_ints(TOKENS[:-1]),
        loss_fn_inputs={
            "target_tokens": TOKENS[1:],
            "weights": [0.0] * 5 + [1.0] * (len(TOKENS) - 6),
        },
    )


def _client(tmp_path) -> TestClient:
    return TestClient(create_app(FakeBackend(root=str(tmp_path))))


def test_health(tmp_path):
    c = _client(tmp_path)
    r = c.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_full_verb_roundtrip_http(tmp_path):
    c = _client(tmp_path)

    cfg = {
        "base_model": "toy/TinyLM-0.1B",
        "lora": {"rank": 8, "alpha": None, "dropout": 0.0, "targets": {}},
        "modalities": ["text"],
    }
    adapter_id = c.post("/v1/sessions", json={"config": cfg}).json()["adapter_id"]
    assert adapter_id

    fb = c.post(
        f"/v1/sessions/{adapter_id}/forward_backward",
        json={
            "data": [
                {
                    "model_input": {"chunks": [{"kind": "text", "tokens": TOKENS[:-1]}]},
                    "loss_fn_inputs": {
                        "target_tokens": TOKENS[1:],
                        "weights": [1.0] * (len(TOKENS) - 1),
                    },
                }
            ],
            "loss_fn": "cross_entropy",
        },
    ).json()
    assert fb["loss"] >= 0.0
    assert fb["metrics"]["n_examples"] == 1.0

    step = c.post(
        f"/v1/sessions/{adapter_id}/optim_step",
        json={"adam": {"learning_rate": 1e-3}},
    ).json()
    assert step["step"] == 1

    ref = c.post(f"/v1/sessions/{adapter_id}/save_state", json={"name": "s1"}).json()
    assert ref["name"] == "s1"
    ok = c.post(f"/v1/sessions/{adapter_id}/load_state", json={"ref": ref}).json()
    assert ok["ok"] is True

    snap = c.post(
        f"/v1/sessions/{adapter_id}/snapshot_for_sample", json={"name": "snap1"}
    ).json()
    assert snap["name"] == "snap1"

    sample = c.post(
        "/v1/sample",
        json={
            "base_model": "toy/TinyLM-0.1B",
            "adapter_id": adapter_id,
            "prompt": {"chunks": [{"kind": "text", "tokens": TOKENS[:5]}]},
            "sampling_params": {"max_tokens": 8, "temperature": 1.0, "seed": 0},
            "num_samples": 2,
        },
    ).json()
    assert len(sample["sequences"]) == 2
    assert len(sample["sequences"][0]["tokens"]) == 8

    lps = c.post(
        "/v1/compute_logprobs",
        json={
            "base_model": "toy/TinyLM-0.1B",
            "adapter_id": None,
            "prompt": {"chunks": [{"kind": "text", "tokens": TOKENS[:5]}]},
        },
    ).json()
    assert len(lps["logprobs"]) == 5

    out = c.post(
        f"/v1/sessions/{adapter_id}/export",
        json={"format": "peft", "path": str(tmp_path / "adapter")},
    ).json()
    assert out["format"] == "peft"
    assert (tmp_path / "adapter" / "adapter_config.json").is_file()


def _tc_transport(c: TestClient):
    def transport(method, path, body):
        resp = c.request(method, path, json=body)
        if resp.status_code >= 400:
            raise RemoteBackendError(resp.status_code, resp.json())
        return resp.json()

    return transport


def test_remote_backend_symmetry(tmp_path):
    """RemoteBackend over the app must behave like the in-process backend."""
    c = _client(tmp_path)
    backend = RemoteBackend("http://testserver", transport=_tc_transport(c))

    from anvil.protocol.types import LoraConfig, TrainConfig

    adapter_id = backend.create_lora_session(
        TrainConfig(base_model="toy/TinyLM-0.1B", lora=LoraConfig(rank=8))
    )
    fb = backend.forward_backward(adapter_id, [_datum()], "cross_entropy")
    assert fb.loss >= 0.0

    step = backend.optim_step(adapter_id, AdamParams(learning_rate=1e-3))
    assert step.step == 1

    ref = backend.save_state(adapter_id, "s1")
    backend.load_state(adapter_id, ref)
    snap = backend.snapshot_for_sample(adapter_id, "snap1")
    assert snap.name == "snap1"

    sample = backend.sample(
        base_model="toy/TinyLM-0.1B",
        adapter_id=adapter_id,
        prompt=ModelInput.from_ints(TOKENS[:5]),
        sampling_params=SamplingParams(max_tokens=8, temperature=1.0, seed=0),
        num_samples=2,
    )
    assert len(sample.sequences) == 2
    assert len(sample.sequences[0].tokens) == 8

    lps = backend.compute_logprobs(
        base_model="toy/TinyLM-0.1B",
        adapter_id=None,
        prompt=ModelInput.from_ints(TOKENS[:5]),
    )
    assert len(lps) == 5

    result = backend.export_adapter(adapter_id, ExportFormat.PEFT, str(tmp_path / "a2"))
    assert result.format == ExportFormat.PEFT


def test_resolve_backend_http_scheme():
    b = resolve_backend("http://forge.local:8741")
    assert isinstance(b, RemoteBackend)
    assert b.base_url == "http://forge.local:8741"


def test_error_mapping_unknown_adapter(tmp_path):
    c = _client(tmp_path)
    r = c.post(
        "/v1/sessions/nope/forward_backward",
        json={"data": [], "loss_fn": "cross_entropy"},
    )
    assert r.status_code == 404
    assert r.json()["error"]


def test_auth_token_required(tmp_path):
    c = TestClient(create_app(FakeBackend(root=str(tmp_path)), token="secret"))
    assert c.get("/v1/health").status_code == 401
    r = c.get("/v1/health", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
