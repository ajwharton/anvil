"""Default live meta-recipe stage runners (SFT/GRPO/DPO/export)."""

from __future__ import annotations

from anvil.observe.metrics import read_jsonl
from anvil.recipes.meta import MetaEdge, MetaRecipe, MetaStage, example_vlm_ladder
from anvil.recipes.meta_runners import (
    DefaultRunnerConfig,
    normalize_pattern,
    run_meta_with_defaults,
    signal_from_early_stop,
)


def test_normalize_and_signal_helpers():
    assert normalize_pattern(MetaStage(id="x", recipe_id="vlm_sft_edge")) == "vlm_sft"
    assert normalize_pattern(MetaStage(id="g", recipe_id="foo", pattern="rl_verifiable")) == (
        "rl_verifiable"
    )
    assert signal_from_early_stop("loss_plateau_patience_12").startswith("early_stop:")
    assert signal_from_early_stop("southward:advantage_collapse") == (
        "early_stop:southward:advantage_collapse"
    )
    assert signal_from_early_stop(None) == "complete"


def test_default_runners_sft_then_grpo(tmp_path):
    meta = MetaRecipe(
        id="sft-grpo",
        title="SFT then GRPO",
        stages=[
            MetaStage(id="sft", recipe_id="sft_chat", pattern="sft_chat"),
            MetaStage(id="grpo", recipe_id="rl", pattern="rl_verifiable"),
        ],
        edges=[MetaEdge(on="early_stop:*", from_stage="sft", to_stage="grpo")],
    )
    cfg = DefaultRunnerConfig(
        endpoint="fake://",
        run_dir=tmp_path / "meta-live",
        sft_steps=40,
        grpo_steps=20,
        early_stop_patience=12,
        grpo_patience=5,
        stop_on_southward=False,
    )
    res = run_meta_with_defaults(meta, config=cfg)
    assert res.stages_run == 2
    assert res.outcomes[0].advanced is True
    assert res.outcomes[0].result.metrics["steps_run"] < 40
    assert "early_stop" in (res.outcomes[0].result.signal or "")
    # stage metrics dirs
    assert (tmp_path / "meta-live" / "sft" / "metrics.jsonl").is_file()
    assert (tmp_path / "meta-live" / "grpo" / "metrics.jsonl").is_file()
    events = read_jsonl(tmp_path / "meta-live" / "metrics.jsonl")
    kinds = [e.get("event") for e in events]
    assert "stage_start" in kinds and "stage_end" in kinds


def test_default_runners_vlm_ladder_with_export(tmp_path):
    meta = example_vlm_ladder()
    assert meta.stages[1].pattern == "export"
    cfg = DefaultRunnerConfig(
        endpoint="fake://",
        base_model="Qwen/Qwen2.5-VL-3B-Instruct",
        run_dir=tmp_path / "vlm-ladder",
        vlm_steps=40,
        early_stop_patience=12,
        stop_on_southward=False,
        export_root=tmp_path / "vlm-ladder" / "peft-out",
    )
    res = run_meta_with_defaults(meta, config=cfg)
    assert res.stages_run == 2
    assert res.outcomes[0].advanced is True
    assert res.outcomes[1].result.signal == "export_done"
    assert res.outcomes[1].result.metrics.get("export_path")


def test_default_runners_dpo_stage(tmp_path):
    meta = MetaRecipe(
        id="dpo-only",
        title="DPO",
        stages=[MetaStage(id="pref", recipe_id="dpo", pattern="preference_dpo")],
    )
    cfg = DefaultRunnerConfig(
        endpoint="fake://",
        run_dir=tmp_path / "dpo-meta",
        dpo_steps=40,
        early_stop_patience=12,
        stop_on_southward=False,
    )
    res = run_meta_with_defaults(meta, config=cfg)
    assert res.stages_run == 1
    assert res.outcomes[0].result.metrics["pattern"] == "preference_dpo"
    assert res.outcomes[0].result.metrics["steps_run"] < 40


def test_default_runners_sft_then_dpo_shares_adapter(tmp_path):
    """SFT → DPO must continue the SAME LoRA adapter (shared client), not a fresh one."""
    meta = MetaRecipe(
        id="sft-dpo",
        title="SFT then DPO",
        stages=[
            MetaStage(id="sft", recipe_id="sft_chat", pattern="sft_chat"),
            MetaStage(id="pref", recipe_id="dpo", pattern="preference_dpo"),
        ],
        edges=[MetaEdge(on="early_stop:*", from_stage="sft", to_stage="pref")],
    )
    cfg = DefaultRunnerConfig(
        endpoint="fake://",
        run_dir=tmp_path / "meta-sft-dpo",
        sft_steps=40,
        dpo_steps=40,
        early_stop_patience=12,
        stop_on_southward=False,
    )
    res = run_meta_with_defaults(meta, config=cfg)
    assert res.stages_run == 2
    sft_metrics = res.outcomes[0].result.metrics
    dpo_metrics = res.outcomes[1].result.metrics
    assert sft_metrics["pattern"] == "sft_chat"
    assert dpo_metrics["pattern"] == "preference_dpo"
    # The whole point of the shared client: one adapter across both stages.
    assert dpo_metrics["adapter_id"] == sft_metrics["adapter_id"]
