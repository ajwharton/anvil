"""Meta-recipe executor."""

from __future__ import annotations

from anvil.observe.metrics import read_jsonl
from anvil.recipes.meta import MetaEdge, MetaRecipe, MetaStage
from anvil.recipes.meta_exec import StageRunResult, run_meta_recipe


def test_executor_advances_on_signal(tmp_path):
    meta = MetaRecipe(
        id="ladder",
        title="L",
        stages=[
            MetaStage(id="sft", recipe_id="vlm_sft_edge", pattern="vlm_sft"),
            MetaStage(id="export", recipe_id="vlm_sft_edge", pattern="vlm_sft"),
        ],
        edges=[
            MetaEdge(
                on="early_stop:loss_plateau*",
                from_stage="sft",
                to_stage="export",
            )
        ],
    )
    calls: list[str] = []

    def runner(stage: MetaStage, *, step_index: int) -> StageRunResult:
        calls.append(stage.id)
        if stage.id == "sft":
            return StageRunResult(
                signal="early_stop:loss_plateau_patience_40",
                metrics={"steps": 69},
            )
        return StageRunResult(signal="export_done", metrics={"ok": True})

    res = run_meta_recipe(meta, runner, run_dir=tmp_path / "meta-run")
    assert calls == ["sft", "export"]
    assert res.stages_run == 2
    assert res.outcomes[0].advanced is True
    assert res.stopped_reason in {"export_done", "complete"}
    events = read_jsonl(tmp_path / "meta-run" / "metrics.jsonl")
    kinds = [e.get("event") for e in events if e.get("type") == "event"]
    assert "stage_start" in kinds
    assert "stage_end" in kinds


def test_executor_halts():
    meta = MetaRecipe(
        id="h",
        title="H",
        stages=[
            MetaStage(id="a", recipe_id="x"),
            MetaStage(id="b", recipe_id="y"),
        ],
    )

    def runner(stage: MetaStage, *, step_index: int) -> StageRunResult:
        return StageRunResult(signal="floor", halted=True)

    res = run_meta_recipe(meta, runner)
    assert res.stages_run == 1
    assert res.stopped_reason == "floor"
