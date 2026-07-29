"""Org packs + experience → production patience (Expert-v2)."""

from __future__ import annotations

from pathlib import Path

from anvil.recipes.book import (
    BookRecipe,
    RecipeBook,
    export_org_pack,
    install_org_pack,
    promote_from_run,
)
from anvil.recipes.experience import (
    aggregate_patience,
    collect_experience_from_book,
    parse_patience_from_reason,
    patience_prior_for_model,
    sample_from_book_recipe,
)
from anvil.recipes.profiles import suggest_for_model


def test_parse_patience_from_reason():
    assert parse_patience_from_reason("loss_plateau_patience_40") == 40
    assert parse_patience_from_reason("ceiling_x8") == 8
    assert parse_patience_from_reason(None) is None


def test_promote_stores_structured_experience(tmp_path):
    from anvil.observe.metrics import RunMetricsWriter

    run = tmp_path / "run"
    w = RunMetricsWriter(run)
    w.log_sft_step(step=0, loss=1.0, n_datums=1, job="vlm_sft")
    w.log_sft_step(step=1, loss=0.4, n_datums=1, job="vlm_sft")
    w.log_event(step=1, event="early_stop", reason="loss_plateau_patience_32")
    book = RecipeBook(tmp_path / "book")
    rec = promote_from_run(
        recipe_id="exp1",
        run_dir=run,
        pattern="vlm_sft",
        family="Qwen2.5-VL",
        book=book,
        early_stop_reason="loss_plateau_patience_32",
    )
    assert rec.stop_policy["patience"] == 32
    assert rec.stop_policy["experience"]["patience"] == 32
    sample = sample_from_book_recipe(rec)
    assert sample is not None
    assert sample.patience == 32


def test_aggregate_patience_median(tmp_path):
    book = RecipeBook(tmp_path)
    for i, pat in enumerate((20, 30, 40)):
        book.save(
            BookRecipe(
                id=f"r{i}",
                title=f"r{i}",
                pattern="vlm_sft",
                family="Qwen2.5-VL",
                stop_policy={
                    "mode": "production",
                    "patience": pat,
                    "experience": {"patience": pat},
                },
            )
        )
    # calibration must be ignored
    book.save(
        BookRecipe(
            id="cal",
            title="cal",
            pattern="vlm_sft",
            family="Qwen2.5-VL",
            stop_policy={"mode": "calibration", "patience": 999},
        )
    )
    samples = collect_experience_from_book(book)
    prior = aggregate_patience(
        samples, family="Qwen2.5-VL", pattern="vlm_sft", atlas_fallback=40
    )
    assert prior.source == "experience"
    assert prior.suggested_patience == 30
    assert prior.n_samples == 3


def test_org_pack_install_and_env_search(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    export_org_pack(
        pack,
        book=RecipeBook(tmp_path / "src"),
        name="empty",
    )
    # seed source book then export
    src = RecipeBook(tmp_path / "src2")
    src.save(
        BookRecipe(
            id="org-a",
            title="A",
            pattern="vlm_sft",
            family="Qwen2.5-VL",
            stop_policy={"mode": "production", "patience": 28},
        )
    )
    pack2 = tmp_path / "pack2"
    export_org_pack(pack2, book=src, name="demo", recipe_ids=["org-a"])
    dest = RecipeBook(tmp_path / "dest")
    installed = install_org_pack(pack2, book=dest)
    assert len(installed) == 1
    assert installed[0].id == "org-a"
    assert "org_pack" in installed[0].tags

    # env search without install
    monkeypatch.setenv("ANVIL_RECIPE_BOOK", str(tmp_path / "personal"))
    monkeypatch.setenv("ANVIL_ORG_RECIPE_PACK", str(pack2))
    (tmp_path / "personal").mkdir()
    merged = RecipeBook()
    ids = {r.id for r in merged.list()}
    assert "org-a" in ids


def test_suggest_includes_experience_priors(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_RECIPE_BOOK", str(tmp_path))
    book = RecipeBook(tmp_path)
    book.save(
        BookRecipe(
            id="lab-prior",
            title="Lab",
            pattern="vlm_sft",
            family="Qwen2.5-VL",
            stop_policy={
                "mode": "production",
                "patience": 33,
                "experience": {"patience": 33},
            },
        )
    )
    s = suggest_for_model("Qwen/Qwen2.5-VL-3B-Instruct", include_personal_book=True)
    assert s["experience_priors"] is not None
    pat = s["experience_priors"]["patience"]
    assert pat["suggested_patience"] == 33
    assert pat["source"] == "experience"


def test_demo_org_pack_in_repo(tmp_path):
    root = Path(__file__).resolve().parents[1] / "packs" / "demo-org-qwen-vl"
    assert (root / "manifest.json").is_file()
    b = RecipeBook(tmp_path)
    rows = install_org_pack(root, book=b)
    assert {r.id for r in rows} >= {"org-vlm-tabletop-v1", "org-vlm-edge-short-v1"}
    prior = patience_prior_for_model(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        pattern="vlm_sft",
        book=b,
        atlas_fallback=40,
    )
    assert prior.source == "experience"
    assert prior.suggested_patience == 30  # median of 25 & 35
