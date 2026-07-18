"""Unit tests for J0 jlens spike scoring helpers (no torch / jlens required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "jlens_spike.py"


def _load():
    import sys

    spec = importlib.util.spec_from_file_location("jlens_spike", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # dataclasses needs the module registered before @dataclass runs
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def spike():
    return _load()


def test_order_score_monotonic(spike):
    # stages light at layers 4, 8, 12 → perfect order
    assert spike.intermediate_order_score([4, 8, 12]) == 1.0


def test_order_score_inversion(spike):
    # answer-ish early, intermediate late
    assert spike.intermediate_order_score([12, 4, 8]) == 0.5  # one of two pairs ok


def test_order_score_insufficient_hits(spike):
    assert spike.intermediate_order_score([None, 5, None]) is None
    assert spike.intermediate_order_score([3]) is None


def test_earliest_stage_layers(spike):
    layer_tops = {
        2: ["the", "a"],
        6: ["3", "x"],
        10: ["7", "sum"],
        14: ["14", "done"],
    }
    stages = (("3",), ("7", "sum"), ("14",))
    assert spike.earliest_stage_layers(layer_tops, stages) == [6, 10, 14]


def test_gate_go(spike):
    results = [
        {"intermediate_order_score": 0.8, "answer_min_rank": 2},
        {"intermediate_order_score": 0.7, "answer_min_rank": 1},
        {"intermediate_order_score": 0.9, "answer_min_rank": None},
    ]
    g = spike._gate_decision(results)
    assert g["go"] is True
    assert g["mean_intermediate_order_score"] == pytest.approx(0.8)


def test_gate_nogo_low_order(spike):
    results = [
        {"intermediate_order_score": 0.2, "answer_min_rank": 1},
        {"intermediate_order_score": 0.3, "answer_min_rank": 1},
    ]
    assert spike._gate_decision(results)["go"] is False


def test_check_command_runs_without_gpu(spike, capsys):
    # May return 1 if jlens not installed — still must not crash
    rc = spike.main(["check", "--model-path", "/nonexistent/model"])
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "jlens_spike check" in out


def test_position_order_score(spike):
    pos_tops = {
        0: ["3", "x"],
        1: ["7", "sum"],
        2: ["14", "done"],
    }
    score, stage_pos = spike.position_order_score(pos_tops, (("3",), ("7",), ("14",)))
    assert stage_pos == [0, 1, 2]
    assert score == 1.0


def test_gate_uses_primary_order(spike):
    results = [
        {"primary_order_score": 0.9, "answer_min_rank": 1},
        {"primary_order_score": 0.8, "answer_min_rank": 2},
    ]
    assert spike._gate_decision(results)["go"] is True


def test_digitseq_hit_layers(spike):
    # answer "14" as digit sequence: "1" at pos 5, "4" at pos 6
    layer_pos_tops = {
        20: {5: ["1", "2"], 6: ["4", "8"]},
        23: {5: ["2", "1"], 6: ["4", "0"]},
        26: {5: ["1"], 6: ["7"]},  # second digit missing at pos 6
    }
    assert spike.digitseq_hit_layers(layer_pos_tops, 5, "14") == [20, 23]


def test_digitseq_hit_layers_single_digit(spike):
    layer_pos_tops = {10: {3: ["7", "x"]}, 12: {3: ["a"]}}
    assert spike.digitseq_hit_layers(layer_pos_tops, 3, "7") == [10]


def test_digitseq_hit_layers_missing_position(spike):
    layer_pos_tops = {23: {5: ["1"]}}  # pos 6 absent
    assert spike.digitseq_hit_layers(layer_pos_tops, 5, "14") == []


def test_digitseq_strips_whitespace(spike):
    layer_pos_tops = {23: {5: [" 1", "2"], 6: [" 4"]}}
    assert spike.digitseq_hit_layers(layer_pos_tops, 5, "14") == [23]


def test_solve_order_score(spike):
    assert spike.solve_order_score([23, 24], [23, 26]) == 1.0
    assert spike.solve_order_score([25, 26], [23, 24]) == 0.0
    assert spike.solve_order_score([], [23]) is None
    assert spike.solve_order_score([23], []) is None


def test_primary_order_picks_up_solve_score(spike):
    # regression: solve records have no pooled/position order — the primary
    # must come from solve_order_score, not get clobbered to None
    rec = {
        "intermediate_order_score": None,
        "position_order_score": None,
        "solve_order_score": 1.0,
    }
    assert spike._primary_order(rec) == 1.0
    assert spike._primary_order({"intermediate_order_score": None}) is None


def test_gate_digitseq_answer_hits(spike):
    # v3 solve records: answer hit via digit sequence, order via solve_order_score
    results = [
        {
            "primary_order_score": 1.0,
            "solve_order_score": 1.0,
            "answer_digitseq_hit": True,
            "answer_min_rank": None,
            "intermediate_order_score": None,
        },
        {
            "primary_order_score": 1.0,
            "solve_order_score": 1.0,
            "answer_digitseq_hit": True,
            "answer_min_rank": None,
            "intermediate_order_score": None,
        },
        {
            "primary_order_score": 0.0,
            "solve_order_score": 0.0,
            "answer_digitseq_hit": True,
            "answer_min_rank": None,
            "intermediate_order_score": None,
        },
    ]
    g = spike._gate_decision(results)
    assert g["n_answer_in_topk"] == 3
    assert g["go"] is True  # mean order 2/3 ≥ 0.6, hits 3 ≥ 2


def test_new_probes_have_v3_fields(spike):
    by_id = {p.id: p for p in spike.DEFAULT_PROBES}
    assert len(spike.DEFAULT_PROBES) == 6
    for pid in ("add_then_mul", "sub_chain", "double_plus", "mul_34", "sub_25", "dbl_22"):
        p = by_id[pid]
        assert p.inter, pid
        assert p.solve_problem, pid
    # v3 guards the "all answers start with 1" digit-prior artifact
    assert any(not p.answer.startswith("1") for p in spike.DEFAULT_PROBES)


def test_write_j1_records_bridge(spike, tmp_path):
    """Solve results land in jlens.jsonl in the J1 schema (endpoint/tripwire-ready)."""
    pytest.importorskip("anvil.observe.jlens")
    results = [
        {
            "protocol": "solve",
            "probe_id": "add_then_mul",
            "text_preview": "Problem: …",
            "generated_continuation": "Step 1: 3 + 4 = 7\nStep 2: 7 * 2 = 14\nAnswer: 14\n",
            "answer": "14",
            "emitted_answer": "14",
            "answer_correct": True,
            "ans_hit_layers": [23, 24, 25, 26],
            "inter_hit_layers": [23, 24],
            "answer_digitseq_hit": True,
            "solve_order_score": 1.0,
            "sanity_top1_agreement": 0.9,
            "positions": [80, 81, 82],
            "wall_time_s": 1.0,
        },
        {"protocol": "last_prompt", "probe_id": "x"},  # skipped
        {"protocol": "solve", "probe_id": "y", "error": "boom"},  # skipped
    ]
    n = spike._write_j1_records(results, tmp_path, "/lenses/qwen2.5-1.5b-instruct/jacobian_lens.pt", 8)
    assert n == 1
    from anvil.observe.jlens import jlens_order_collapsed
    from anvil.observe.metrics import read_jsonl

    (rec,) = read_jsonl(tmp_path / "jlens.jsonl")
    assert rec["type"] == "jlens"
    assert rec["signals"]["answer_digitseq_hit"] is True
    assert rec["signals"]["intermediate_order_score"] == 1.0
    assert rec["lens_id"] == "qwen2.5-1.5b-instruct"
    assert rec["protocol"] == "solve"
    assert not jlens_order_collapsed(rec)
