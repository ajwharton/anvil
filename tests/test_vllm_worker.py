"""Tests for the vLLM sample worker, using a fake ``vllm`` module.

No GPU or vllm install needed: the worker lazily imports ``vllm`` inside its
constructor, so tests inject a stub into ``sys.modules`` and assert on the
calls the worker makes (param mapping, LoRA hot-swap ids, logprob plumbing).
"""

from __future__ import annotations

import sys
import types

import pytest
from fastapi.testclient import TestClient

from anvil.protocol.types import (
    AdapterId,
    EncodedTextChunk,
    ModelInput,
    SamplingParams,
)
from anvil.workers.sample import VLLMSampleBackend


class _FakeLogprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class _FakeOutput:
    def __init__(self) -> None:
        self.token_ids = [11, 12]
        self.logprobs = [{11: _FakeLogprob(-0.5)}, {12: _FakeLogprob(-1.5)}]
        self.finish_reason = "stop"


class _FakeRequestOutput:
    def __init__(self, n: int, include_prompt_lps: bool) -> None:
        self.outputs = [_FakeOutput() for _ in range(n)]
        # Mirrors vLLM: one entry per prompt token (first is None), PLUS the
        # known quirk of a trailing entry for the first sampled continuation
        # token — the worker must trim it back to prompt length.
        self.prompt_logprobs = (
            [None, {8: _FakeLogprob(-2.0)}, {11: _FakeLogprob(-0.5)}]
            if include_prompt_lps
            else None
        )


class _FakeLLM:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.generate_calls: list[tuple] = []

    def generate(self, prompts, sampling_params, lora_request=None):
        self.generate_calls.append((prompts, sampling_params, lora_request))
        return [
            _FakeRequestOutput(
                sampling_params.n, sampling_params.prompt_logprobs is not None
            )
            for _ in prompts
        ]


@pytest.fixture()
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    mod = types.ModuleType("vllm")
    mod.LLM = _FakeLLM
    mod.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    mod.TokensPrompt = dict
    monkeypatch.setitem(sys.modules, "vllm", mod)
    # LoRARequest is NOT top-level in real vllm; the worker deep-imports it.
    lora_mod = types.ModuleType("vllm.lora.request")
    lora_mod.LoRARequest = (
        lambda lora_name, lora_int_id, lora_path: types.SimpleNamespace(
            lora_name=lora_name, lora_int_id=lora_int_id, lora_path=lora_path
        )
    )
    monkeypatch.setitem(sys.modules, "vllm.lora", types.ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_mod)
    return mod


def _worker() -> VLLMSampleBackend:
    return VLLMSampleBackend(model="test-model")


def _prompt() -> ModelInput:
    return ModelInput(chunks=(EncodedTextChunk(tokens=(7, 8)),))


def _sample(w: VLLMSampleBackend, **overrides):
    kwargs = {
        "base_model": "test-model",
        "adapter_id": None,
        "prompt": _prompt(),
        "sampling_params": SamplingParams(),
        "num_samples": 1,
    }
    kwargs.update(overrides)
    return w.sample(**kwargs)


def test_sample_maps_params_and_returns_true_policy_logprobs(fake_vllm):
    w = _worker()
    params = SamplingParams(
        max_tokens=5, temperature=0.7, top_p=0.9, top_k=10, stop=("END",), seed=3
    )
    result = _sample(w, sampling_params=params, num_samples=2)

    prompts, sp, lora_req = w._llm.generate_calls[-1]
    assert list(prompts[0]["prompt_token_ids"]) == [7, 8]
    assert (sp.n, sp.max_tokens, sp.temperature, sp.top_p, sp.top_k, sp.seed) == (
        2,
        5,
        0.7,
        0.9,
        10,
        3,
    )
    assert sp.stop == ["END"]
    assert sp.logprobs == 1  # RL needs sampled-token logprobs even if not asked
    assert lora_req is None

    assert len(result.sequences) == 2
    seq = result.sequences[0]
    assert seq.tokens == (11, 12)
    assert seq.logprobs == (-0.5, -1.5)
    assert seq.stop_reason == "stop"
    assert result.prompt_logprobs is None


def test_top_k_none_maps_to_disabled(fake_vllm):
    w = _worker()
    _sample(w)  # SamplingParams() default top_k is None
    _, sp, _ = w._llm.generate_calls[-1]
    assert sp.top_k == 0  # vLLM 0.25 "disabled" sentinel


def test_sample_includes_prompt_logprobs_when_asked(fake_vllm):
    w = _worker()
    result = _sample(w, include_prompt_logprobs=True)
    assert result.prompt_logprobs == (None, -2.0)


def test_compute_logprobs_scores_prompt_only(fake_vllm):
    w = _worker()
    lps = w.compute_logprobs(
        base_model="test-model", adapter_id=None, prompt=_prompt()
    )
    assert lps == [None, -2.0]
    _, sp, _ = w._llm.generate_calls[-1]
    assert sp.max_tokens == 1
    assert sp.prompt_logprobs == 1


def test_load_snapshot_hot_swap_bumps_lora_id(tmp_path, fake_vllm):
    w = _worker()
    aid = AdapterId("default")

    snap1 = tmp_path / "snap1"
    snap1.mkdir()
    w.load_snapshot(aid, str(snap1))
    _sample(w, adapter_id=aid)
    first = w._llm.generate_calls[-1][2]
    assert (first.lora_name, first.lora_int_id, first.lora_path) == (
        "default",
        1,
        str(snap1),
    )

    # Re-pushing the same adapter_id must NOT reuse the id, or vLLM would
    # serve the stale cached adapter.
    snap2 = tmp_path / "snap2"
    snap2.mkdir()
    w.load_snapshot(aid, str(snap2))
    _sample(w, adapter_id=aid)
    second = w._llm.generate_calls[-1][2]
    assert second.lora_int_id == 2
    assert second.lora_path == str(snap2)


def test_sample_with_unknown_adapter_raises_key_error(fake_vllm):
    w = _worker()
    with pytest.raises(KeyError, match="load_snapshot"):
        _sample(w, adapter_id=AdapterId("missing"))


def test_load_snapshot_missing_dir_raises(tmp_path, fake_vllm):
    w = _worker()
    with pytest.raises(FileNotFoundError):
        w.load_snapshot(AdapterId("x"), str(tmp_path / "nope"))


def test_training_verbs_raise_not_implemented(fake_vllm):
    w = _worker()
    aid = AdapterId("a")
    with pytest.raises(NotImplementedError, match="sample worker"):
        w.forward_backward(aid, [], "importance_sampling")
    with pytest.raises(NotImplementedError, match="sample worker"):
        w.optim_step(aid, None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="sample worker"):
        w.export_adapter(aid, "peft", "/tmp/x")  # type: ignore[arg-type]


def test_load_snapshot_route_registered_for_workers(tmp_path, fake_vllm):
    from anvil.serve.app import create_app

    client = TestClient(create_app(_worker()))
    snap = tmp_path / "s"
    snap.mkdir()
    r = client.post("/v1/adapters/default/load_snapshot", json={"path": str(snap)})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "adapter_id": "default", "path": str(snap)}


def test_load_snapshot_route_absent_for_plain_backends():
    """Backends without SnapshotLoader must not register the hot-swap route.

    FakeBackend *does* implement load_snapshot (Tier-1 test double); use a
    minimal train-only stub here.
    """
    from anvil.serve.app import create_app

    class _NoSnapshot:
        name = "no-snap"

        def create_lora_session(self, config):  # noqa: ANN001
            raise NotImplementedError

        def forward_backward(self, *a, **k):  # noqa: ANN001
            raise NotImplementedError

        def optim_step(self, *a, **k):  # noqa: ANN001
            raise NotImplementedError

        def save_state(self, *a, **k):  # noqa: ANN001
            raise NotImplementedError

        def load_state(self, *a, **k):  # noqa: ANN001
            raise NotImplementedError

        def snapshot_for_sample(self, *a, **k):  # noqa: ANN001
            raise NotImplementedError

        def sample(self, *a, **k):  # noqa: ANN001
            raise NotImplementedError

        def compute_logprobs(self, *a, **k):  # noqa: ANN001
            raise NotImplementedError

        def export_adapter(self, *a, **k):  # noqa: ANN001
            raise NotImplementedError

    client = TestClient(create_app(_NoSnapshot()))  # type: ignore[arg-type]
    r = client.post("/v1/adapters/default/load_snapshot", json={"path": "/tmp/x"})
    assert r.status_code == 404


def test_sample_over_http_roundtrip(fake_vllm):
    from anvil.serve.app import create_app

    client = TestClient(create_app(_worker()))
    r = client.post(
        "/v1/sample",
        json={
            "base_model": "test-model",
            "adapter_id": None,
            "prompt": {"chunks": [{"kind": "text", "tokens": [7, 8]}]},
            "sampling_params": {"max_tokens": 4},
            "num_samples": 1,
        },
    )
    assert r.status_code == 200, r.text
    seq = r.json()["sequences"][0]
    assert seq["tokens"] == [11, 12]
    assert seq["logprobs"] == [-0.5, -1.5]
    assert seq["stop_reason"] == "stop"
