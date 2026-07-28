"""Meta-recipe skeleton + next_stage."""

from __future__ import annotations

from anvil.recipes.meta import (
    MetaEdge,
    MetaRecipe,
    MetaStage,
    example_vlm_ladder,
    get_meta_recipe,
    list_meta_recipes,
    next_stage,
    save_meta_recipe,
)


def test_next_stage_sequential_and_edge(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_RECIPE_BOOK", str(tmp_path))
    meta = MetaRecipe(
        id="ladder-a",
        title="A",
        stages=[
            MetaStage(id="s0", recipe_id="r0"),
            MetaStage(id="s1", recipe_id="r1"),
            MetaStage(id="s2", recipe_id="r2"),
        ],
        edges=[
            MetaEdge(on="early_stop:loss_plateau*", from_stage="s0", to_stage="s2"),
        ],
    )
    assert next_stage(meta, current_stage_id="s0").id == "s1"
    assert (
        next_stage(
            meta, current_stage_id="s0", signal="early_stop:loss_plateau_patience_40"
        ).id
        == "s2"
    )
    assert next_stage(meta, current_stage_id="s2") is None


def test_save_list_get_meta(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_RECIPE_BOOK", str(tmp_path))
    m = example_vlm_ladder()
    path = save_meta_recipe(m)
    assert path.is_file()
    got = get_meta_recipe("vlm-sft-then-export")
    assert got is not None
    assert got.stages[0].recipe_id == "vlm_sft_edge"
    assert len(list_meta_recipes()) == 1
