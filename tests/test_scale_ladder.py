"""Expert-v2 scale ladder + throughput defaults."""

from __future__ import annotations

from anvil.recipes.scale_ladder import (
    build_ladder_plan,
    exercise_ladder,
    exercise_rung,
    get_rung,
    list_rungs,
)
from anvil.recipes.throughput import list_throughput_profiles, throughput_defaults


def test_rungs_and_throughput_defaults():
    ids = [r.id for r in list_rungs()]
    assert ids == ["1k", "5k", "50k"]
    assert get_rung("1000").max_rows == 1000
    assert get_rung("50k+").max_rows == 50_000
    thr = throughput_defaults(shape="dense_vlm", pattern="vlm_sft")
    assert thr.batch_size == 1
    assert thr.checkpoint_every >= 1
    assert len(list_throughput_profiles()) >= 4


def test_exercise_rung_demo_1k(tmp_path):
    r = exercise_rung(
        get_rung("1k"),
        demo=True,
        work_dir=tmp_path / "r1k",
        endpoint="fake://",
        train=True,
    )
    assert r.ok
    assert r.rows_converted == 32
    assert r.steps_run >= 1
    assert r.checkpoint_path is not None


def test_exercise_full_demo_ladder(tmp_path):
    plan = build_ladder_plan(rungs=["all"], demo=True)
    results = exercise_ladder(plan, work_dir=tmp_path / "ladder", endpoint="fake://")
    assert len(results) == 3
    assert all(r.ok for r in results)
    assert [r.rung.id for r in results] == ["1k", "5k", "50k"]
    assert results[0].rows_converted < results[2].rows_converted


def test_scale_ladder_script_demo(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "scale_ladder.py"),
            "--demo",
            "--rung",
            "1k",
            "--work-dir",
            str(tmp_path / "cli"),
            "--endpoint",
            "fake://",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "cli" / "ladder_report.json").is_file()
