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
