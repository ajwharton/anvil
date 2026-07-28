"""Personal recipe book (P.Recipes v0)."""

from __future__ import annotations

import json

from anvil.observe.metrics import RunMetricsWriter
from anvil.recipes.book import BookRecipe, RecipeBook, promote_from_run


def test_book_save_list_get(tmp_path):
    book = RecipeBook(tmp_path)
    r = BookRecipe(
        id="vlm-tabletop-v1",
        title="Tabletop VLM",
        pattern="vlm_sft",
        family="Qwen2.5-VL",
        knobs={"rank": 16},
        stop_policy={"mode": "production", "patience": 40},
        notes="lab prior",
        tags=["robotics"],
    )
    path = book.save(r)
    assert path.is_file()
    got = book.get("vlm-tabletop-v1")
    assert got is not None
    assert got.knobs["rank"] == 16
    assert got.family == "Qwen2.5-VL"
    listed = book.list()
    assert len(listed) == 1
    pref = book.prefer(pattern="vlm_sft", family="Qwen2.5")
    assert len(pref) == 1


def test_promote_from_run_reads_metrics(tmp_path):
    run_dir = tmp_path / "run"
    w = RunMetricsWriter(run_dir)
    w.log_sft_step(step=0, loss=1.0, n_datums=2, n_image_refs=1, job="vlm_sft")
    w.log_sft_step(step=1, loss=0.5, n_datums=2, n_image_refs=1, job="vlm_sft")
    w.log_event(step=1, event="early_stop", reason="loss_plateau_patience_40", mode="production")
    book = RecipeBook(tmp_path / "book")
    rec = promote_from_run(
        recipe_id="from-run",
        run_dir=run_dir,
        run_id="run",
        pattern="vlm_sft",
        family="Qwen2.5-VL",
        knobs={"rank": 8},
        book=book,
        early_stop_reason="loss_plateau_patience_40",
    )
    assert rec.source_run_id == "run"
    assert "loss_first" in rec.notes or "observe_summary" in rec.notes
    loaded = json.loads((book.root / "from-run.json").read_text())
    assert loaded["stop_policy"]["last_early_stop_reason"] == "loss_plateau_patience_40"


def test_bad_recipe_id_rejected(tmp_path):
    book = RecipeBook(tmp_path)
    try:
        book.save(BookRecipe(id="../evil", title="x"))
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "bad recipe id" in str(e)
