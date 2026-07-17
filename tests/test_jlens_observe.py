"""J1: jlens.jsonl schema, writer, signals — no real jacobian-lens package."""

from __future__ import annotations

from anvil.observe.jlens import (
    JLENS_FILENAME,
    build_jlens_record,
    compute_signals,
    intermediate_order_score,
    jlens_order_collapsed,
    layer_tops_from_slice,
)
from anvil.observe.metrics import RunMetricsWriter, read_jsonl


def _fake_slice() -> dict:
    """Compact layer→position→top tokens (as a real apply would emit)."""
    return {
        "4": {"-1": [{"tok": "3", "rank": 1}, {"tok": "the", "rank": 2}]},
        "8": {"-1": [{"tok": "7", "rank": 1}, {"tok": "sum", "rank": 3}]},
        "12": {"-1": [{"tok": "14", "rank": 1}, {"tok": "answer", "rank": 5}]},
        "16": {"-1": [{"tok": "14", "rank": 1}, {"tok": ".", "rank": 2}]},
    }


def test_layer_tops_from_slice():
    tops = layer_tops_from_slice(_fake_slice())
    assert tops[4][0] == "3"
    assert tops[12][0] == "14"


def test_compute_signals_order_and_answer():
    stages = (("3",), ("7", "sum"), ("14",))
    sig = compute_signals(slice_=_fake_slice(), stages=stages, answer="14")
    assert sig["stage_layers"] == [4, 8, 12]
    assert sig["intermediate_order_score"] == 1.0
    assert sig["answer_token_min_rank"] == 1


def test_build_and_write_roundtrip(tmp_path):
    w = RunMetricsWriter(tmp_path / "run-j1")
    rec = w.log_jlens(
        step=3,
        probe_idx=0,
        prompt_preview="First add 3 and 4…",
        completion_preview="14",
        slice_=_fake_slice(),
        stages=(("3",), ("7",), ("14",)),
        answer="14",
        lens_id="test-lens",
        adapter_id="adapter-x",
        wall_time_s=0.5,
        top_k=5,
    )
    assert rec["type"] == "jlens"
    assert rec["schema_version"] == 1
    assert rec["step"] == 3
    assert rec["signals"]["intermediate_order_score"] == 1.0
    assert rec["lens_id"] == "test-lens"
    path = tmp_path / "run-j1" / JLENS_FILENAME
    assert path.is_file()
    (loaded,) = read_jsonl(path)
    assert loaded["probe_idx"] == 0
    assert loaded["slice"]["12"]["-1"][0]["tok"] == "14"
    assert not jlens_order_collapsed(loaded)


def test_jlens_order_collapsed_tripwire():
    rec = build_jlens_record(
        step=0,
        signals={"intermediate_order_score": 0.2, "answer_token_min_rank": 1},
    )
    assert jlens_order_collapsed(rec)
    assert not jlens_order_collapsed(
        build_jlens_record(step=0, signals={"intermediate_order_score": 0.9})
    )
    assert not jlens_order_collapsed(build_jlens_record(step=0, signals={}))


def test_observe_jlens_api(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_OBSERVE_ROOT", str(tmp_path))
    w = RunMetricsWriter(tmp_path / "r-jlens")
    w.log_step(
        step=0,
        reward_mean=0.5,
        reward_std=0.1,
        group_reward_std_mean=0.2,
        loss=0.1,
        n_datums=2,
    )
    w.log_jlens(
        step=0,
        probe_idx=1,
        slice_=_fake_slice(),
        stages=(("3",), ("14",)),
        answer="14",
    )
    from anvil.web.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    r = client.get("/api/observe/r-jlens/jlens")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == "r-jlens"
    assert len(body["jlens"]) == 1
    assert body["jlens"][0]["probe_idx"] == 1
    assert body["jlens"][0]["signals"]["answer_token_min_rank"] == 1


def test_intermediate_order_shared_with_spike_semantics():
    assert intermediate_order_score([4, 8, 12]) == 1.0
    assert intermediate_order_score([12, 4]) == 0.0
