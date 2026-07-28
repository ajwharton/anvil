"""Lab smoke runner — quick profile is CI-safe (fake:// only)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "lab_smokes.py"


def test_lab_smokes_list():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--list"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    assert "fake_early_stop" in proc.stdout
    assert "nightly" in proc.stdout


def test_lab_smokes_quick_profile(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    report_dir = tmp_path / "reports"
    observe = tmp_path / "observe"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--profile",
            "quick",
            "--endpoint",
            "fake://",
            "--observe-root",
            str(observe),
            "--report-dir",
            str(report_dir),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    latest = report_dir / "latest.json"
    assert latest.is_file()
    report = json.loads(latest.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["failed"] == 0
    names = {r["name"] for r in report["results"]}
    # Expert-v1 quick profile: early-stop + DPO observe/probes + meta + southward + queue + expert path
    for required in (
        "fake_early_stop",
        "fake_sft_early_stop",
        "fake_dpo_observe",
        "fake_meta_exec",
        "fake_meta_live_runners",
        "fake_southward",
        "fake_rl_queue",
        "fake_expert_v0",
    ):
        assert required in names, f"missing smoke {required} in {sorted(names)}"
