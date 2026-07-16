"""Bounded recipe catalog — sensible defaults with explicit gates.

People may stretch past these boundaries; Anvil should still name the boundary
and say when a combo is *recommended*, *stretch*, or *blocked* for v0.

15 recipes cover dense LM, dense VLM, edge student, and MoE shapes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence

from anvil.recipes.profiles import JobPattern, ModelShape


class GateLevel(str, Enum):
    """How hard the boundary is."""

    RECOMMENDED = "recommended"  # default happy path
    STRETCH = "stretch"  # allowed with warnings; expert territory
    BLOCKED = "blocked"  # refuse unless force=True


@dataclass(frozen=True, slots=True)
class GateResult:
    level: GateLevel
    recipe_id: str
    shape: ModelShape
    reasons: tuple[str, ...] = ()
    stretch_reasons: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.level != GateLevel.BLOCKED

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        d["shape"] = self.shape.value
        d["ok"] = self.ok
        return d


@dataclass(frozen=True, slots=True)
class RecipeSpec:
    """One productized recipe with architecture boundaries."""

    id: str
    title: str
    summary: str
    pattern: JobPattern
    # Architecture boundaries
    shapes_recommended: tuple[ModelShape, ...]
    shapes_stretch: tuple[ModelShape, ...] = ()
    # None = no hard block beyond empty
    shapes_blocked: tuple[ModelShape, ...] = ()
    # Size gates (billions of params); None = unrestricted
    max_param_b_recommended: float | None = None
    min_param_b_recommended: float | None = None
    max_param_b_hard: float | None = None  # above this → blocked
    # Capability gates
    needs_vision: bool = False
    needs_reward: bool = False
    needs_preference_pairs: bool = False
    allows_vision_encoder_lora: bool = False  # default freeze
    # Knob bounds (gates on overrides)
    rank_min: int = 4
    rank_max_recommended: int = 64
    rank_max_hard: int = 256
    lr_max_recommended: float = 5e-4
    # Defaults that override pattern-level derivation when set
    default_export: str | None = None
    default_rank: int | None = None
    default_lr: float | None = None
    group: str = "train"  # train | rl | preference | edge | eval
    tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Catalog (14 recipes) — order is UI display order within groups
# ---------------------------------------------------------------------------

RECIPES: tuple[RecipeSpec, ...] = (
    # --- dense language ---
    RecipeSpec(
        id="sft_chat_dense",
        title="Chat SFT · dense LM",
        summary="Standard instruction/chat SFT on dense text models (Phi, Qwen text, Llama-class).",
        pattern=JobPattern.SFT_CHAT,
        shapes_recommended=(ModelShape.DENSE_LM,),
        shapes_stretch=(ModelShape.UNKNOWN, ModelShape.EDGE_STUDENT),
        shapes_blocked=(),
        max_param_b_recommended=14.0,
        max_param_b_hard=80.0,
        default_rank=32,
        group="train",
        tags=("sft", "dense", "text"),
    ),
    RecipeSpec(
        id="sft_chat_moe",
        title="Chat SFT · MoE",
        summary="Conservative LoRA SFT on mixture-of-experts bases. Lower rank/LR; validate on a slice first.",
        pattern=JobPattern.SFT_CHAT,
        shapes_recommended=(ModelShape.MOE_LM,),
        shapes_stretch=(ModelShape.DENSE_LM,),  # using MoE recipe on dense is odd but ok
        shapes_blocked=(ModelShape.DENSE_VLM, ModelShape.EDGE_STUDENT),
        max_param_b_recommended=40.0,
        max_param_b_hard=120.0,
        default_rank=16,
        default_lr=5e-5,
        rank_max_recommended=32,
        group="train",
        tags=("sft", "moe", "text"),
    ),
    # --- vision-language ---
    RecipeSpec(
        id="vlm_sft_lab",
        title="VLM instruction SFT · lab",
        summary="Multimodal SFT for lab-sized VLMs (~7B). Freeze vision encoder; LoRA LM + projector.",
        pattern=JobPattern.VLM_SFT,
        shapes_recommended=(ModelShape.DENSE_VLM,),
        shapes_stretch=(ModelShape.EDGE_STUDENT, ModelShape.UNKNOWN),
        shapes_blocked=(ModelShape.MOE_LM,),
        needs_vision=True,
        min_param_b_recommended=4.0,
        max_param_b_recommended=14.0,
        max_param_b_hard=40.0,
        default_rank=32,
        default_export="peft",
        group="train",
        tags=("sft", "vlm", "lab"),
    ),
    RecipeSpec(
        id="vlm_sft_edge",
        title="VLM instruction SFT · edge student",
        summary="Edge-sized VLM (e.g. Qwen2.5-VL-3B). Smaller rank, ONNX-leaning export, Jetson path.",
        pattern=JobPattern.VLM_SFT,
        shapes_recommended=(ModelShape.EDGE_STUDENT,),
        shapes_stretch=(ModelShape.DENSE_VLM,),
        shapes_blocked=(ModelShape.MOE_LM, ModelShape.DENSE_LM),
        needs_vision=True,
        max_param_b_recommended=4.5,
        max_param_b_hard=8.0,
        default_rank=16,
        default_lr=2e-4,
        default_export="onnx",
        group="edge",
        tags=("sft", "vlm", "edge", "jetson"),
    ),
    RecipeSpec(
        id="vlm_classifier",
        title="VLM classifier / rubric",
        summary="Short graded answers over frames (grasp ok? lane clear?). Low seq budget.",
        pattern=JobPattern.VLM_CLASSIFIER,
        shapes_recommended=(ModelShape.EDGE_STUDENT, ModelShape.DENSE_VLM),
        shapes_stretch=(ModelShape.UNKNOWN,),
        shapes_blocked=(ModelShape.MOE_LM, ModelShape.DENSE_LM),
        needs_vision=True,
        max_param_b_recommended=14.0,
        default_rank=16,
        group="train",
        tags=("sft", "vlm", "classifier"),
    ),
    RecipeSpec(
        id="vlm_encoder_lora",
        title="VLM + vision-encoder LoRA (stretch)",
        summary="Opens LoRA on the vision encoder. Use only when projector+LM adapters plateau.",
        pattern=JobPattern.VLM_SFT,
        shapes_recommended=(),  # nothing is "recommended" — always stretch
        shapes_stretch=(ModelShape.DENSE_VLM, ModelShape.EDGE_STUDENT),
        shapes_blocked=(ModelShape.DENSE_LM, ModelShape.MOE_LM),
        needs_vision=True,
        allows_vision_encoder_lora=True,
        max_param_b_recommended=8.0,
        max_param_b_hard=20.0,
        default_rank=16,
        rank_max_recommended=32,
        group="train",
        tags=("sft", "vlm", "stretch", "encoder"),
    ),
    # --- RL ---
    RecipeSpec(
        id="rl_verifiable_dense",
        title="On-policy verifiable RL · dense",
        summary="Sample → verifiable reward → group-relative / IS loss. Math, code, exact-match.",
        pattern=JobPattern.RL_VERIFIABLE,
        shapes_recommended=(ModelShape.DENSE_LM, ModelShape.EDGE_STUDENT),
        shapes_stretch=(ModelShape.UNKNOWN,),
        shapes_blocked=(),
        needs_reward=True,
        max_param_b_recommended=14.0,
        max_param_b_hard=40.0,
        default_rank=16,
        default_lr=5e-5,
        lr_max_recommended=1e-4,
        group="rl",
        tags=("rl", "grpo", "dense"),
    ),
    RecipeSpec(
        id="rl_verifiable_vlm",
        title="On-policy verifiable RL · VLM",
        summary="Vision-aware on-policy RL (UI agents, visual puzzles with checkable rewards).",
        pattern=JobPattern.RL_VERIFIABLE,
        shapes_recommended=(ModelShape.DENSE_VLM, ModelShape.EDGE_STUDENT),
        shapes_stretch=(ModelShape.UNKNOWN,),
        shapes_blocked=(ModelShape.MOE_LM,),
        needs_vision=True,
        needs_reward=True,
        max_param_b_recommended=14.0,
        default_rank=16,
        default_lr=5e-5,
        group="rl",
        tags=("rl", "vlm", "grpo"),
    ),
    RecipeSpec(
        id="rl_verifiable_moe",
        title="On-policy verifiable RL · MoE (stretch)",
        summary="RL on MoE bases. High systems cost; start tiny, expect expert-routing quirks.",
        pattern=JobPattern.RL_VERIFIABLE,
        shapes_recommended=(),
        shapes_stretch=(ModelShape.MOE_LM,),
        shapes_blocked=(ModelShape.EDGE_STUDENT,),
        needs_reward=True,
        max_param_b_recommended=40.0,
        max_param_b_hard=120.0,
        default_rank=8,
        default_lr=3e-5,
        rank_max_recommended=16,
        group="rl",
        tags=("rl", "moe", "stretch"),
    ),
    # --- preference ---
    RecipeSpec(
        id="preference_dpo_dense",
        title="Preference DPO · dense",
        summary="Paired preferred/rejected completions. No online reward model.",
        pattern=JobPattern.PREFERENCE_DPO,
        shapes_recommended=(ModelShape.DENSE_LM,),
        shapes_stretch=(ModelShape.EDGE_STUDENT, ModelShape.MOE_LM, ModelShape.UNKNOWN),
        shapes_blocked=(),
        needs_preference_pairs=True,
        max_param_b_recommended=14.0,
        default_rank=16,
        default_lr=5e-5,
        group="preference",
        tags=("dpo", "dense"),
    ),
    RecipeSpec(
        id="preference_dpo_vlm",
        title="Preference DPO · VLM",
        summary="Multimodal preference pairs (better caption / safer answer over image).",
        pattern=JobPattern.PREFERENCE_DPO,
        shapes_recommended=(ModelShape.DENSE_VLM, ModelShape.EDGE_STUDENT),
        shapes_stretch=(ModelShape.UNKNOWN,),
        shapes_blocked=(ModelShape.DENSE_LM,),  # use dense DPO instead
        needs_vision=True,
        needs_preference_pairs=True,
        max_param_b_recommended=14.0,
        default_rank=16,
        default_lr=5e-5,
        group="preference",
        tags=("dpo", "vlm"),
    ),
    # --- edge / robot ---
    RecipeSpec(
        id="robot_offline_edge",
        title="Robot offline · edge loop",
        summary="Offline trajectories with image refs → SFT/offline RL; export toward Jetson.",
        pattern=JobPattern.ROBOT_OFFLINE,
        shapes_recommended=(ModelShape.EDGE_STUDENT,),
        shapes_stretch=(ModelShape.DENSE_VLM,),
        shapes_blocked=(ModelShape.MOE_LM, ModelShape.DENSE_LM),
        needs_vision=True,
        max_param_b_recommended=4.5,
        max_param_b_hard=10.0,
        default_rank=16,
        default_export="onnx",
        group="edge",
        tags=("robot", "edge", "offline"),
    ),
    RecipeSpec(
        id="distill_to_edge",
        title="Distill lab teacher → edge student",
        summary="Teacher (lab VLM/LM) produces targets; train small student for FPS/power.",
        pattern=JobPattern.VLM_SFT,  # student side is SFT-shaped; teacher is external
        shapes_recommended=(ModelShape.EDGE_STUDENT,),
        shapes_stretch=(ModelShape.DENSE_LM,),  # text distill student
        shapes_blocked=(ModelShape.MOE_LM, ModelShape.DENSE_VLM),  # student must be small
        max_param_b_recommended=4.5,
        max_param_b_hard=8.0,
        default_rank=16,
        default_export="onnx",
        group="edge",
        tags=("distill", "edge", "jetson"),
    ),
    # --- agent / eval ---
    RecipeSpec(
        id="tool_agent_sft",
        title="Tool / agent SFT",
        summary="Tool-call and multi-step agent traces (text; VLM if base is multimodal).",
        pattern=JobPattern.SFT_CHAT,
        shapes_recommended=(ModelShape.DENSE_LM, ModelShape.DENSE_VLM, ModelShape.EDGE_STUDENT),
        shapes_stretch=(ModelShape.MOE_LM, ModelShape.UNKNOWN),
        shapes_blocked=(),
        max_param_b_recommended=32.0,
        default_rank=32,
        group="train",
        tags=("sft", "tools", "agent"),
    ),
    RecipeSpec(
        id="eval_sample_only",
        title="Eval / sample only",
        summary="No training — sample under base (+ optional adapter) for baselines and gates.",
        pattern=JobPattern.SFT_CHAT,  # pattern unused for train
        shapes_recommended=(
            ModelShape.DENSE_LM,
            ModelShape.DENSE_VLM,
            ModelShape.EDGE_STUDENT,
            ModelShape.MOE_LM,
            ModelShape.UNKNOWN,
        ),
        shapes_stretch=(),
        shapes_blocked=(),
        group="eval",
        tags=("eval", "sample"),
    ),
)

_BY_ID: dict[str, RecipeSpec] = {r.id: r for r in RECIPES}


def get_recipe(recipe_id: str) -> RecipeSpec:
    try:
        return _BY_ID[recipe_id]
    except KeyError as e:
        known = ", ".join(sorted(_BY_ID))
        raise KeyError(f"unknown recipe {recipe_id!r}; known: {known}") from e


def list_recipes(*, group: str | None = None) -> list[dict[str, Any]]:
    out = []
    for r in RECIPES:
        if group and r.group != group:
            continue
        out.append(_spec_public(r))
    return out


def _spec_public(r: RecipeSpec) -> dict[str, Any]:
    return {
        "id": r.id,
        "title": r.title,
        "summary": r.summary,
        "pattern": r.pattern.value,
        "group": r.group,
        "tags": list(r.tags),
        "shapes_recommended": [s.value for s in r.shapes_recommended],
        "shapes_stretch": [s.value for s in r.shapes_stretch],
        "shapes_blocked": [s.value for s in r.shapes_blocked],
        "needs_vision": r.needs_vision,
        "needs_reward": r.needs_reward,
        "needs_preference_pairs": r.needs_preference_pairs,
        "allows_vision_encoder_lora": r.allows_vision_encoder_lora,
        "max_param_b_recommended": r.max_param_b_recommended,
        "min_param_b_recommended": r.min_param_b_recommended,
        "max_param_b_hard": r.max_param_b_hard,
        "rank_max_recommended": r.rank_max_recommended,
        "default_rank": r.default_rank,
        "default_lr": r.default_lr,
        "default_export": r.default_export,
    }


def gate_recipe(
    recipe_id: str,
    *,
    shape: ModelShape | str,
    param_count: int | None = None,
    has_vision: bool | None = None,
    rank: int | None = None,
    learning_rate: float | None = None,
    vision_encoder_lora: bool | None = None,
) -> GateResult:
    """Evaluate whether this recipe is sensible for the model shape/size."""
    spec = get_recipe(recipe_id)
    if isinstance(shape, str):
        shape = ModelShape(shape)

    blocked: list[str] = []
    stretch: list[str] = []
    reasons: list[str] = []

    param_b = (param_count / 1e9) if param_count else None

    # Shape matrix
    if shape in spec.shapes_blocked:
        blocked.append(f"shape {shape.value} is blocked for recipe {spec.id}")
    elif shape in spec.shapes_recommended:
        reasons.append(f"shape {shape.value} is recommended for {spec.id}")
    elif shape in spec.shapes_stretch:
        stretch.append(f"shape {shape.value} is stretch for {spec.id}")
    elif not spec.shapes_recommended and shape in spec.shapes_stretch:
        stretch.append(f"recipe {spec.id} is always stretch; shape {shape.value} allowed")
    elif not spec.shapes_recommended and not spec.shapes_stretch:
        # eval-only style: all shapes recommended listed fully
        reasons.append(f"shape {shape.value} accepted")
    else:
        # not listed anywhere
        if spec.shapes_recommended or spec.shapes_stretch:
            stretch.append(
                f"shape {shape.value} not in recommended/stretch lists for {spec.id} — treat as stretch"
            )

    # Vision capability
    if spec.needs_vision and has_vision is False:
        blocked.append("recipe needs vision but model has_vision=False")
    if spec.needs_vision and shape == ModelShape.DENSE_LM:
        blocked.append("vision recipe on dense_lm shape")

    # Param size
    if param_b is not None:
        if spec.max_param_b_hard is not None and param_b > spec.max_param_b_hard:
            blocked.append(
                f"params≈{param_b:.1f}B exceeds hard max {spec.max_param_b_hard}B for {spec.id}"
            )
        if (
            spec.max_param_b_recommended is not None
            and param_b > spec.max_param_b_recommended
        ):
            stretch.append(
                f"params≈{param_b:.1f}B above recommended max {spec.max_param_b_recommended}B"
            )
        if (
            spec.min_param_b_recommended is not None
            and param_b < spec.min_param_b_recommended
        ):
            stretch.append(
                f"params≈{param_b:.1f}B below recommended min {spec.min_param_b_recommended}B "
                f"(consider edge recipe)"
            )

    # Rank / LR overrides
    if rank is not None:
        if rank < spec.rank_min:
            blocked.append(f"rank {rank} < minimum {spec.rank_min}")
        if rank > spec.rank_max_hard:
            blocked.append(f"rank {rank} > hard max {spec.rank_max_hard}")
        elif rank > spec.rank_max_recommended:
            stretch.append(f"rank {rank} > recommended max {spec.rank_max_recommended}")

    if learning_rate is not None and learning_rate > spec.lr_max_recommended:
        stretch.append(
            f"lr {learning_rate} > recommended max {spec.lr_max_recommended}"
        )

    # Encoder LoRA
    if vision_encoder_lora and not spec.allows_vision_encoder_lora:
        stretch.append(
            "vision_encoder LoRA enabled but recipe defaults to freeze — stretch"
        )
    if (
        vision_encoder_lora is False
        and spec.allows_vision_encoder_lora
        and recipe_id == "vlm_encoder_lora"
    ):
        stretch.append("vlm_encoder_lora recipe usually sets vision_encoder_lora=True")

    # Always-stretch recipes (empty recommended)
    if not spec.shapes_recommended and spec.shapes_stretch and shape in spec.shapes_stretch:
        if not any("always stretch" in s for s in stretch):
            stretch.append(f"recipe {spec.id} is expert/stretch by design")

    if blocked:
        level = GateLevel.BLOCKED
    elif stretch:
        level = GateLevel.STRETCH
    else:
        level = GateLevel.RECOMMENDED

    return GateResult(
        level=level,
        recipe_id=spec.id,
        shape=shape,
        reasons=tuple(reasons),
        stretch_reasons=tuple(stretch),
        blocked_reasons=tuple(blocked),
    )


def recipes_for_shape(
    shape: ModelShape | str,
    *,
    param_count: int | None = None,
    has_vision: bool | None = None,
    include_blocked: bool = False,
) -> list[dict[str, Any]]:
    """All catalog recipes with gate level for this shape (UI matrix)."""
    if isinstance(shape, str):
        shape = ModelShape(shape)
    rows = []
    for r in RECIPES:
        g = gate_recipe(
            r.id,
            shape=shape,
            param_count=param_count,
            has_vision=has_vision,
        )
        if g.level == GateLevel.BLOCKED and not include_blocked:
            continue
        rows.append({**_spec_public(r), "gate": g.to_public()})
    # recommended first, then stretch
    order = {GateLevel.RECOMMENDED: 0, GateLevel.STRETCH: 1, GateLevel.BLOCKED: 2}
    rows.sort(key=lambda x: (order[GateLevel(x["gate"]["level"])], x["id"]))
    return rows


def default_recipe_id_for_shape(shape: ModelShape | str) -> str:
    """Sensible default recipe when user only picks a base model."""
    if isinstance(shape, str):
        shape = ModelShape(shape)
    mapping = {
        ModelShape.EDGE_STUDENT: "vlm_sft_edge",
        ModelShape.DENSE_VLM: "vlm_sft_lab",
        ModelShape.MOE_LM: "sft_chat_moe",
        ModelShape.DENSE_LM: "sft_chat_dense",
        ModelShape.UNKNOWN: "sft_chat_dense",
    }
    return mapping[shape]


def apply_recipe_defaults(spec: RecipeSpec, knobs: dict[str, Any]) -> dict[str, Any]:
    """Merge recipe default rank/lr/export into knob dict."""
    out = dict(knobs)
    if spec.default_rank is not None:
        out["rank"] = spec.default_rank
    if spec.default_lr is not None:
        out["learning_rate"] = spec.default_lr
    if spec.default_export is not None:
        out["export_hint"] = spec.default_export
    if spec.allows_vision_encoder_lora:
        out["vision_encoder_lora"] = True
    elif spec.needs_vision:
        out.setdefault("vision_encoder_lora", False)
    return out
