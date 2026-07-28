"""suggest_for_model prefers personal book when family matches."""

from __future__ import annotations

from anvil.recipes.book import BookRecipe, RecipeBook
from anvil.recipes.profiles import infer_model_family, suggest_for_model


def test_infer_model_family_qwen_vl():
    assert infer_model_family("Qwen/Qwen2.5-VL-3B-Instruct") == "Qwen2.5-VL"
    assert infer_model_family("/mnt/data/models/Qwen2.5-VL-3B-Instruct") == "Qwen2.5-VL"


def test_suggest_prefixes_personal_book(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_RECIPE_BOOK", str(tmp_path))
    book = RecipeBook(tmp_path)
    book.save(
        BookRecipe(
            id="vlm-lerobot-pusht-v1",
            title="LeRobot pusht prior",
            pattern="vlm_sft",
            family="Qwen2.5-VL",
            knobs={"rank": 16},
            stop_policy={"mode": "production", "patience": 40},
            notes="from overnight dogfood",
        )
    )
    s = suggest_for_model("Qwen/Qwen2.5-VL-3B-Instruct", include_personal_book=True)
    assert s["family"] == "Qwen2.5-VL"
    assert s["personal_book"]
    assert s["personal_book"][0]["recipe_id"] == "vlm-lerobot-pusht-v1"
    assert s["recipes"][0]["source"] == "personal_book"
    assert s["default_recipe_id"] == "vlm-lerobot-pusht-v1"
    # catalog still present
    assert any(c.get("source") == "catalog" for c in s["recipes"])
    assert s["catalog"]


def test_suggest_can_skip_personal_book(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_RECIPE_BOOK", str(tmp_path))
    RecipeBook(tmp_path).save(
        BookRecipe(id="x", title="x", pattern="vlm_sft", family="Qwen2.5-VL")
    )
    s = suggest_for_model(
        "Qwen/Qwen2.5-VL-3B-Instruct", include_personal_book=False
    )
    assert s["personal_book"] == []
    assert all(c.get("source") == "catalog" for c in s["recipes"])
