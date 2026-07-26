"""End-to-end: grpo_observe_demo.py writes metrics under observe root (fake://)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "grpo_observe_demo.py"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return env


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_grpo_observe_demo_fake_writes_metrics_and_probes(tmp_path):
    observe = tmp_path / "observe"
    run_id = "grpo-test-fake"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--endpoint",
        "fake://",
        "--steps",
        "3",
        "--group-size",
        "4",
        "--observe-root",
        str(observe),
        "--run-id",
        run_id,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

    run_dir = observe / run_id
    metrics = run_dir / "metrics.jsonl"
    probes = run_dir / "probes.jsonl"
    assert metrics.is_file()
    assert probes.is_file()

    steps = [
        json.loads(line)
        for line in metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(steps) == 3
    for s in steps:
        assert s["type"] == "step"
        assert "reward_mean" in s
        assert "group_reward_std_mean" in s
        assert "loss" in s

    probe_recs = [
        json.loads(line)
        for line in probes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(probe_recs) >= 3  # probe_every=1, ≥1 probe ids
    assert all(p.get("text") is not None for p in probe_recs)


def test_grpo_observe_demo_rejects_bad_run_id(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--endpoint",
            "fake://",
            "--steps",
            "1",
            "--observe-root",
            str(tmp_path),
            "--run-id",
            "../evil",
        ],
        cwd=str(ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode != 0
