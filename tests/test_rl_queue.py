"""Multi-stage RL recipe queue — advance / halt policy."""

from __future__ import annotations

from pathlib import Path

from anvil.observe.metrics import read_jsonl
from anvil.recipes.rl_queue import (
    RLQueueRecipe,
    RLStage,
    load_rl_queue_recipe,
    recipe_from_hard_bank,
    run_rl_queue,
    should_advance,
)


def test_should_advance_ceiling_and_floor():
    adv, halt = should_advance(
        "ceiling_x8",
        hit_budget=False,
        advance_on=("ceiling", "collapsed"),
        stop_queue_on=("floor",),
        advance_on_budget=True,
    )
    assert adv and not halt

    adv, halt = should_advance(
        "floor_x8",
        hit_budget=False,
        advance_on=("ceiling",),
        stop_queue_on=("floor",),
        advance_on_budget=True,
    )
    assert not adv and halt

    adv, halt = should_advance(
        None,
        hit_budget=True,
        advance_on=("ceiling",),
        stop_queue_on=("floor",),
        advance_on_budget=True,
    )
    assert adv and not halt


def test_load_arith_curriculum_json():
    root = Path(__file__).resolve().parents[1]
    recipe = load_rl_queue_recipe(root / "recipes" / "arith_curriculum_v1.json")
    assert recipe.id == "arith-curriculum-v1"
    assert len(recipe.stages) >= 3
    assert recipe.stages[0].gold == "127"
    assert "ceiling" in recipe.advance_on


def test_recipe_from_hard_bank():
    r = recipe_from_hard_bank(max_steps=10)
    assert len(r.stages) == 3
    assert r.stages[1].gold == "90"


def test_queue_advances_on_ceiling(tmp_path):
    recipe = RLQueueRecipe(
        id="ceil-queue",
        name="ceil",
        stages=(
            RLStage(id="a", prompt="p1", gold="always_one", max_steps=40),
            RLStage(id="b", prompt="p2", gold="always_one", max_steps=40),
        ),
        early_stop_patience=3,
        advance_on=("ceiling",),
        stop_queue_on=("floor",),
    )
    result = run_rl_queue(
        recipe,
        base_model="toy/lm",
        endpoint="fake://",
        observe_root=tmp_path,
        run_prefix="tq",
        fake_prompts=True,
    )
    assert result.stages_run == 2
    assert result.stages[0].result.early_stop_reason == "ceiling_x3"
    assert result.stages[0].result.steps_run == 3
    assert result.stages[0].advanced is True
    assert result.stages[1].result.early_stop_reason == "ceiling_x3"
    # same adapter carried
    assert result.stages[0].result.adapter_id == result.stages[1].result.adapter_id
    # queue events
    qdir = tmp_path / "tq-queue"
    events = read_jsonl(qdir / "metrics.jsonl")
    kinds = [e.get("event") for e in events]
    assert "stage_start" in kinds and "stage_end" in kinds


def test_queue_halts_on_floor(tmp_path):
    recipe = RLQueueRecipe(
        id="floor-queue",
        name="floor",
        stages=(
            RLStage(id="bad", prompt="p", gold="always_zero", max_steps=40),
            RLStage(id="good", prompt="p2", gold="always_one", max_steps=40),
        ),
        early_stop_patience=3,
        advance_on=("ceiling",),
        stop_queue_on=("floor",),
    )
    result = run_rl_queue(
        recipe,
        base_model="toy/lm",
        endpoint="fake://",
        observe_root=tmp_path,
        run_prefix="fq",
        fake_prompts=True,
    )
    assert result.stages_run == 1
    assert result.stages[0].result.early_stop_reason == "floor_x3"
    assert result.stages[0].queue_halted is True
    assert result.stages[0].advanced is False


def test_script_fake_queue_smoke(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = {**dict(**__import__("os").environ), "PYTHONPATH": str(root)}
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "grpo_recipe_queue.py"),
            "--endpoint",
            "fake://",
            "--recipe-builtin",
            "hard-bank",
            "--observe-root",
            str(tmp_path),
            "--run-prefix",
            "script-q",
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "stages_completed=" in proc.stdout
