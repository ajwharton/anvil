"""Architecture-aware recipes — catalog + gates + model cards + personal book."""

from anvil.recipes.book import (
    BookRecipe,
    RecipeBook,
    default_book_root,
    promote_from_run,
)
from anvil.recipes.catalog import (
    GateLevel,
    GateResult,
    RecipeSpec,
    default_recipe_id_for_shape,
    gate_recipe,
    get_recipe,
    list_recipes,
    recipes_for_shape,
)
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
from anvil.recipes.model_card import ModelCardFacts, inspect_base_model
from anvil.recipes.profiles import (
    JobPattern,
    ModelShape,
    PatternSpec,
    RecipePlan,
    infer_model_family,
    infer_shape,
    list_patterns,
    plan_recipe,
    suggest_for_model,
)

__all__ = [
    "BookRecipe",
    "GateLevel",
    "GateResult",
    "JobPattern",
    "MetaEdge",
    "MetaRecipe",
    "MetaStage",
    "ModelCardFacts",
    "ModelShape",
    "PatternSpec",
    "RecipeBook",
    "RecipePlan",
    "RecipeSpec",
    "default_book_root",
    "default_recipe_id_for_shape",
    "example_vlm_ladder",
    "gate_recipe",
    "get_meta_recipe",
    "get_recipe",
    "infer_model_family",
    "infer_shape",
    "inspect_base_model",
    "list_meta_recipes",
    "list_patterns",
    "list_recipes",
    "next_stage",
    "plan_recipe",
    "promote_from_run",
    "recipes_for_shape",
    "save_meta_recipe",
    "suggest_for_model",
]
