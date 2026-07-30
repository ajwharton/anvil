"""Edge FPS measure (dry-run) + multi-cycle agent dogfood."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from anvil.backends.jetson import JetsonSampleConfig, measure_sample_fps


def test_measure_sample_fps_dry_run():
    stats = measure_sample_fps(
        config=JetsonSampleConfig(dry_run=True, model="smolvlm-256m"),
        n=3,
        warmup=1,
    )
    assert stats["n"] == 3
    assert stats["fps_mean"] >= 0
    assert "storage_note" in stats
    assert stats["url"] == "dry_run"


def test_j30_edge_fps_smoke_script():
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "j30_edge_fps_smoke.py"), "--dry-run", "--n", "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["dry_run"] is True
    assert data["n"] == 2


def test_agent_dogfood_multi_cycle(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "dog"
    r = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "agent_dogfood.py"),
            "--job",
            "sft",
            "--steps",
            "2",
            "--cycles",
            "2",
            "--run-dir",
            str(run_dir),
            "--endpoint",
            "fake://",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    data = json.loads(r.stdout)
    assert data["cycles"] >= 1
    assert (run_dir / "decisions.jsonl").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
