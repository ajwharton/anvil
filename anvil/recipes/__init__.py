"""Architecture-aware recipes — catalog + gates + model cards."""

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
from anvil.recipes.model_card import ModelCardFacts, inspect_base_model
from anvil.recipes.profiles import (
    JobPattern,
    ModelShape,
    PatternSpec,
    RecipePlan,
    infer_shape,
    list_patterns,
    plan_recipe,
    suggest_for_model,
)

__all__ = [
    "GateLevel",
    "GateResult",
    "JobPattern",
    "ModelCardFacts",
    "ModelShape",
    "PatternSpec",
    "RecipePlan",
    "RecipeSpec",
    "default_recipe_id_for_shape",
    "gate_recipe",
    "get_recipe",
    "infer_shape",
    "inspect_base_model",
    "list_patterns",
    "list_recipes",
    "plan_recipe",
    "recipes_for_shape",
    "suggest_for_model",
]
