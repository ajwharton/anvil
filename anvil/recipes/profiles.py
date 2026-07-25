"""Architecture profiles — the hard part is *not* knobs.

Knobs are a small, finite surface (rank, lr, loss, freeze masks, …).
What Anvil should own is: given a **model shape** (dense LM, dense VLM,
edge student, …) and a **job pattern** (SFT, on-policy RL, preference,
robot offline), emit a defensible default recipe.

Profiles are deliberate opinions, not autotune magic. Override when you
know better; the default should already be runnable on lab hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from anvil.recipes.families import lookup_family


class ModelShape(str, Enum):
    """Coarse architecture class — drives freeze policy and recipe fit."""

    DENSE_LM = "dense_lm"
    DENSE_VLM = "dense_vlm"
    MOE_LM = "moe_lm"
    EDGE_STUDENT = "edge_student"  # small dense (often VLM) aimed at Jetson
    UNKNOWN = "unknown"


class JobPattern(str, Enum):
    """What you are trying to teach — not which optimizer flag."""

    SFT_CHAT = "sft_chat"
    VLM_SFT = "vlm_sft"
    VLM_CLASSIFIER = "vlm_classifier"
    RL_VERIFIABLE = "rl_verifiable"  # GRPO / IS on exact-match rewards
    PREFERENCE_DPO = "preference_dpo"
    ROBOT_OFFLINE = "robot_offline"  # trajectory SFT / offline RL later


@dataclass(frozen=True, slots=True)
class LoraShape:
    rank: int = 32
    alpha: int | None = None
    language: bool = True
    vision_encoder: bool = False
    mm_projector: bool = True

    def effective_alpha(self) -> int:
        return self.alpha if self.alpha is not None else 2 * self.rank


@dataclass(frozen=True, slots=True)
class RecipePlan:
    """Resolved training plan: pattern + architecture → concrete knobs + rationale."""

    pattern: JobPattern
    shape: ModelShape
    base_model: str
    loss_fn: str
    learning_rate: float
    lora: LoraShape
    modalities: tuple[str, ...]
    max_steps: int
    batch_size: int
    seq_len: int
    temperature: float
    max_tokens: int
    export_hint: str  # peft | merged_hf | gguf | onnx | trt
    title: str
    rationale: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()
    next_patterns: tuple[str, ...] = ()
    peft_target_modules: tuple[str, ...] = ()
    sources: tuple[dict[str, str], ...] = ()
    card_evidence: tuple[str, ...] = ()
    shape_confidence: str = "low"
    recipe_id: str | None = None
    gate: dict[str, Any] | None = None

    def as_knobs(self) -> dict[str, Any]:
        """Flatten into RunKnobs-compatible dict (UI / ServiceClient)."""
        return {
            "base_model": self.base_model,
            "rank": self.lora.rank,
            "alpha": self.lora.alpha,
            "learning_rate": self.learning_rate,
            "loss_fn": self.loss_fn,
            "modalities": list(self.modalities),
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "language_lora": self.lora.language,
            "vision_encoder_lora": self.lora.vision_encoder,
            "mm_projector_lora": self.lora.mm_projector,
            "peft_target_modules": list(self.peft_target_modules),
            "recipe_id": self.recipe_id,
        }

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        d["pattern"] = self.pattern.value
        d["shape"] = self.shape.value
        d["knobs"] = self.as_knobs()
        return d


# --- architecture inference -------------------------------------------------

_VLM_MARKERS = ("-vl", "vl-", "vision", "vlm", "qwen2.5-vl", "qwen2-vl", "internvl")
_MOE_MARKERS = ("moe", "a3b", "a22b", "mixtral", "deepseek-v")
_EDGE_MARKERS = ("0.5b", "1.5b", "1b", "2b", "3b")  # size cues, not perfect


def infer_shape(base_model: str) -> ModelShape:
    """Best-effort shape from HF id / path basename. Prefer explicit override."""
    s = base_model.lower().replace("_", "-")
    name = s.rsplit("/", 1)[-1]

    is_vlm = any(m in s for m in _VLM_MARKERS)
    is_moe = any(m in s for m in _MOE_MARKERS)
    is_small = any(m in name for m in _EDGE_MARKERS) and not any(
        x in name for x in ("7b", "8b", "13b", "14b", "32b", "70b", "72b")
    )

    if is_moe and not is_vlm:
        return ModelShape.MOE_LM
    if is_vlm and is_small:
        return ModelShape.EDGE_STUDENT
    if is_vlm:
        return ModelShape.DENSE_VLM
    if is_small:
        return ModelShape.DENSE_LM
    if "qwen" in s or "phi" in s or "llama" in s or "gemma" in s:
        return ModelShape.DENSE_LM
    return ModelShape.UNKNOWN


def shapes_compatible(shape: ModelShape, pattern: JobPattern) -> bool:
    if pattern in {JobPattern.VLM_SFT, JobPattern.VLM_CLASSIFIER, JobPattern.ROBOT_OFFLINE}:
        return shape in {ModelShape.DENSE_VLM, ModelShape.EDGE_STUDENT, ModelShape.UNKNOWN}
    if pattern == JobPattern.SFT_CHAT:
        return shape != ModelShape.MOE_LM or True  # MoE SFT allowed but cautioned
    return True


# --- pattern catalog -------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PatternSpec:
    pattern: JobPattern
    title: str
    summary: str
    default_loss: str
    needs_vision: bool = False
    needs_reward: bool = False


PATTERNS: dict[JobPattern, PatternSpec] = {
    JobPattern.SFT_CHAT: PatternSpec(
        pattern=JobPattern.SFT_CHAT,
        title="Chat SFT",
        summary="Imitate assistant turns on text (or text-primary) data.",
        default_loss="cross_entropy",
    ),
    JobPattern.VLM_SFT: PatternSpec(
        pattern=JobPattern.VLM_SFT,
        title="VLM instruction SFT",
        summary="Multimodal SFT; freeze heavy vision encoder by default.",
        default_loss="cross_entropy",
        needs_vision=True,
    ),
    JobPattern.VLM_CLASSIFIER: PatternSpec(
        pattern=JobPattern.VLM_CLASSIFIER,
        title="VLM classifier / rubric",
        summary="Short graded answers over frames (grasp ok? lane clear?).",
        default_loss="cross_entropy",
        needs_vision=True,
    ),
    JobPattern.RL_VERIFIABLE: PatternSpec(
        pattern=JobPattern.RL_VERIFIABLE,
        title="On-policy verifiable RL",
        summary="Sample → score (exact match / unit test) → IS or PPO-style loss.",
        default_loss="importance_sampling",
        needs_reward=True,
    ),
    JobPattern.PREFERENCE_DPO: PatternSpec(
        pattern=JobPattern.PREFERENCE_DPO,
        title="Preference (DPO)",
        summary="Paired preferred/rejected completions; no online reward model required.",
        default_loss="dpo",
    ),
    JobPattern.ROBOT_OFFLINE: PatternSpec(
        pattern=JobPattern.ROBOT_OFFLINE,
        title="Robot offline (edge loop)",
        summary="Trajectories with image refs → SFT/offline RL; export toward Jetson.",
        default_loss="cross_entropy",
        needs_vision=True,
    ),
}


def list_patterns() -> list[dict[str, Any]]:
    return [
        {
            "id": p.pattern.value,
            "title": p.title,
            "summary": p.summary,
            "default_loss": p.default_loss,
            "needs_vision": p.needs_vision,
            "needs_reward": p.needs_reward,
        }
        for p in PATTERNS.values()
    ]


def plan_recipe(
    *,
    base_model: str,
    pattern: JobPattern | str | None = None,
    recipe_id: str | None = None,
    shape: ModelShape | str | None = None,
    overrides: dict[str, Any] | None = None,
    card: Any | None = None,
    use_card: bool = True,
    fetch_remote: bool = False,
    force: bool = False,
    record_override: bool = True,
) -> RecipePlan:
    """Derive a sensible plan from catalog recipe and/or job pattern.

    Prefer ``recipe_id`` from the bounded catalog (gates applied). Pattern-only
    remains supported for low-level use. ``force=True`` allows blocked gates;
    every forced pass through a *blocked* gate is written to the control-plane
    audit trail (anvil.control.audit) unless ``record_override=False`` — used
    by preview/enumeration paths that force gates only to display them.
    """
    from anvil.recipes.catalog import (
        GateLevel,
        apply_recipe_defaults,
        default_recipe_id_for_shape,
        gate_recipe,
        get_recipe,
    )
    from anvil.recipes.research import research_notes, sources_for_pattern

    card_evidence: tuple[str, ...] = ()
    shape_confidence = "low"
    peft_targets: tuple[str, ...] = ()
    resolved_base = base_model
    param_count = None
    has_vision = None
    recipe_spec = None

    if card is None and use_card:
        try:
            from anvil.recipes.model_card import inspect_base_model

            card = inspect_base_model(base_model, fetch_remote=fetch_remote)
        except Exception:
            card = None

    if card is not None:
        # Prefer on-disk snapshot for loading (AutoProcessor/from_pretrained).
        # repo_id alone is not a valid HF id when it is only a folder basename.
        resolved_base = (
            getattr(card, "local_path", None)
            or getattr(card, "repo_id", None)
            or base_model
        )
        if shape is None:
            shape = card.shape
        shape_confidence = getattr(card, "shape_confidence", "medium")
        card_evidence = tuple(getattr(card, "evidence", ()) or ())
        peft_targets = tuple(getattr(card, "peft_target_modules", ()) or ())
        param_count = getattr(card, "param_count", None)
        has_vision = getattr(card, "has_vision", None)
        if param_count:
            card_evidence = card_evidence + (f"card param_count≈{param_count / 1e9:.2f}B",)

    if shape is None:
        shape = infer_shape(resolved_base)
        shape_confidence = "low"
    elif isinstance(shape, str):
        shape = ModelShape(shape)

    # Resolve catalog recipe
    if recipe_id is None and pattern is None:
        recipe_id = default_recipe_id_for_shape(shape)
    if recipe_id is not None:
        recipe_spec = get_recipe(recipe_id)
        pattern = recipe_spec.pattern
    if isinstance(pattern, str):
        pattern = JobPattern(pattern)
    if pattern is None:
        raise ValueError("pattern or recipe_id required")

    # Apply recipe-level default knobs before derive/overrides
    pre_overrides: dict[str, Any] = {}
    if recipe_spec is not None:
        pre_overrides = apply_recipe_defaults(recipe_spec, {})
    if overrides:
        pre_overrides.update({k: v for k, v in overrides.items() if v is not None})

    # Gate
    gate_pub = None
    if recipe_spec is not None:
        g = gate_recipe(
            recipe_spec.id,
            shape=shape,
            param_count=param_count,
            has_vision=has_vision,
            rank=pre_overrides.get("rank"),
            learning_rate=pre_overrides.get("learning_rate"),
            vision_encoder_lora=pre_overrides.get("vision_encoder_lora"),
        )
        gate_pub = g.to_public()
        if g.level == GateLevel.BLOCKED:
            if not force:
                raise ValueError(
                    f"recipe {recipe_spec.id!r} blocked for shape={shape.value}: "
                    + "; ".join(g.blocked_reasons)
                    + " (pass force=True to stretch past the gate)"
                )
            if record_override:
                from anvil.control.audit import default_log, gate_override_event

                default_log().record(
                    gate_override_event(
                        recipe_id=recipe_spec.id,
                        base_model=resolved_base,
                        shape=shape.value,
                        blocked_reasons=g.blocked_reasons,
                        stretch_reasons=g.stretch_reasons,
                    )
                )

    if not peft_targets:
        peft_targets = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )

    pspec = PATTERNS[pattern]
    lora, lr, steps, batch, seq, mods, export, rationale, cautions = _derive(
        shape, pattern, resolved_base
    )

    # Recipe defaults win over shape derivation where specified
    if recipe_spec is not None:
        if recipe_spec.default_rank is not None:
            lora = LoraShape(
                rank=recipe_spec.default_rank,
                alpha=lora.alpha,
                language=lora.language,
                vision_encoder=recipe_spec.allows_vision_encoder_lora,
                mm_projector=lora.mm_projector if pspec.needs_vision else False,
            )
        if recipe_spec.default_lr is not None:
            lr = recipe_spec.default_lr
        if recipe_spec.default_export is not None:
            export = recipe_spec.default_export
        if recipe_spec.allows_vision_encoder_lora:
            lora = LoraShape(
                rank=lora.rank,
                alpha=lora.alpha,
                language=True,
                vision_encoder=True,
                mm_projector=True,
            )
        title = f"{recipe_spec.title}"
        rationale = (f"catalog recipe={recipe_spec.id}",) + rationale
        if gate_pub and gate_pub.get("level") == "stretch":
            cautions = cautions + tuple(gate_pub.get("stretch_reasons") or ())
        if gate_pub and gate_pub.get("level") == "blocked":
            cautions = cautions + ("FORCE: blocked gate overridden",) + tuple(
                gate_pub.get("blocked_reasons") or ()
            )
    else:
        title = f"{pspec.title} · {shape.value}"

    # Per-family training knowledge (families.py): correct LoRA targets per
    # architecture, LR caps, template footguns. Family beats card-derived
    # generic targets; explicit user overrides (below) beat family.
    fam = lookup_family(
        resolved_base,
        model_type=getattr(card, "model_type", None) if card is not None else None,
        architectures=tuple(getattr(card, "architectures", ()) or ())
        if card is not None
        else (),
    )
    if fam is not None:
        peft_targets = fam.lora_targets
        rationale = (f"family={fam.id} ({fam.display})",) + rationale
        if fam.template_notes:
            rationale = rationale + (f"template: {fam.template_notes}",)
        ce_pattern = pattern in {
            JobPattern.SFT_CHAT,
            JobPattern.VLM_SFT,
            JobPattern.VLM_CLASSIFIER,
            JobPattern.ROBOT_OFFLINE,
        }
        if ce_pattern and fam.sft_notes:
            rationale = rationale + (fam.sft_notes,)
        if pattern in {JobPattern.RL_VERIFIABLE, JobPattern.PREFERENCE_DPO} and fam.rl_notes:
            rationale = rationale + (fam.rl_notes,)
        cautions = cautions + fam.cautions
        if ce_pattern and fam.sft_lr_cap is not None and lr > fam.sft_lr_cap:
            cautions = cautions + (
                f"lr {lr:g} capped at family max {fam.sft_lr_cap:g} for {fam.id}",
            )
            lr = fam.sft_lr_cap

    rationale = rationale + tuple(research_notes(pattern.value)[:2])
    if card is not None and getattr(card, "model_type", None):
        rationale = (
            f"HF model_type={card.model_type!r} arch={list(getattr(card, 'architectures', ())[:1])}",
        ) + rationale
    if card is not None and getattr(card, "max_position_embeddings", None):
        ctx = int(card.max_position_embeddings)
        if seq > 2048 and ctx >= 8192:
            cautions = cautions + (
                f"card max_position_embeddings={ctx}; recipe seq_len={seq} keeps train context modest",
            )

    sources = tuple(sources_for_pattern(pattern.value))

    # User overrides after recipe defaults
    if pre_overrides:
        if "rank" in pre_overrides:
            lora = LoraShape(
                rank=int(pre_overrides["rank"]),
                alpha=pre_overrides.get("alpha", lora.alpha),
                language=bool(pre_overrides.get("language_lora", lora.language)),
                vision_encoder=bool(
                    pre_overrides.get("vision_encoder_lora", lora.vision_encoder)
                ),
                mm_projector=bool(
                    pre_overrides.get("mm_projector_lora", lora.mm_projector)
                ),
            )
        if "learning_rate" in pre_overrides:
            lr = float(pre_overrides["learning_rate"])
        if "max_steps" in pre_overrides:
            steps = int(pre_overrides["max_steps"])
        if "batch_size" in pre_overrides:
            batch = int(pre_overrides["batch_size"])
        if "seq_len" in pre_overrides:
            seq = int(pre_overrides["seq_len"])
        if "modalities" in pre_overrides:
            mods = tuple(pre_overrides["modalities"])
        if "loss_fn" in pre_overrides:
            loss_override = pre_overrides["loss_fn"]
        else:
            loss_override = None
        if "temperature" in pre_overrides:
            temperature = float(pre_overrides["temperature"])
        else:
            # RL rollouts need exploration; SFT/pref sampling stays near-greedy
            temperature = 1.0 if pattern == JobPattern.RL_VERIFIABLE else 0.2
        if "max_tokens" in pre_overrides:
            max_tokens = int(pre_overrides["max_tokens"])
        else:
            max_tokens = 64 if pattern != JobPattern.VLM_CLASSIFIER else 16
        if "export_hint" in pre_overrides:
            export = pre_overrides["export_hint"]
        if "peft_target_modules" in pre_overrides:
            peft_targets = tuple(pre_overrides["peft_target_modules"])
        if overrides:
            rationale = rationale + ("user overrides applied",)
    else:
        loss_override = None
        temperature = 1.0 if pattern == JobPattern.RL_VERIFIABLE else 0.2
        max_tokens = 64 if pattern != JobPattern.VLM_CLASSIFIER else 16

    loss_fn = loss_override if loss_override else pspec.default_loss

    return RecipePlan(
        pattern=pattern,
        shape=shape,
        base_model=resolved_base if not pre_overrides.get("base_model") else pre_overrides["base_model"],
        loss_fn=loss_fn,
        learning_rate=lr,
        lora=lora,
        modalities=mods,
        max_steps=steps,
        batch_size=batch,
        seq_len=seq,
        temperature=temperature,
        max_tokens=max_tokens,
        export_hint=export,
        title=title,
        rationale=rationale,
        cautions=cautions,
        next_patterns=_next_patterns(pattern, shape),
        peft_target_modules=peft_targets,
        sources=sources,
        card_evidence=card_evidence,
        shape_confidence=shape_confidence,
        recipe_id=recipe_spec.id if recipe_spec else None,
        gate=gate_pub,
    )


def _derive(
    shape: ModelShape,
    pattern: JobPattern,
    base_model: str,
) -> tuple[
    LoraShape,
    float,
    int,
    int,
    int,
    tuple[str, ...],
    str,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Return lora, lr, max_steps, batch, seq, modalities, export, rationale, cautions."""
    rationale: list[str] = []
    cautions: list[str] = []

    # Defaults by shape
    if shape == ModelShape.EDGE_STUDENT:
        lora = LoraShape(rank=16, language=True, vision_encoder=False, mm_projector=True)
        lr = 2e-4
        batch, seq = 4, 512
        export = "onnx"
        rationale.append("Edge student: smaller rank, higher LR, export path toward Jetson.")
    elif shape == ModelShape.DENSE_VLM:
        lora = LoraShape(rank=32, language=True, vision_encoder=False, mm_projector=True)
        lr = 1e-4
        batch, seq = 2, 1024
        export = "peft"
        rationale.append("Dense VLM: LoRA language+projector; freeze vision encoder until data proves need.")
    elif shape == ModelShape.MOE_LM:
        lora = LoraShape(rank=16, language=True, vision_encoder=False, mm_projector=False)
        lr = 5e-5
        batch, seq = 1, 1024
        export = "peft"
        rationale.append("MoE: conservative rank/LR; expert routing is not fully modeled in v0.")
        cautions.append("MoE LoRA is research-grade here — validate on a tiny subset first.")
    else:  # dense_lm / unknown
        lora = LoraShape(rank=32, language=True, vision_encoder=False, mm_projector=False)
        lr = 1e-4
        batch, seq = 4, 1024
        export = "peft"
        rationale.append("Dense LM: standard LoRA on language modules only.")

    # Pattern adjustments
    mods: tuple[str, ...]
    steps = 200

    if pattern in {JobPattern.VLM_SFT, JobPattern.VLM_CLASSIFIER, JobPattern.ROBOT_OFFLINE}:
        mods = ("text", "image")
        if shape == ModelShape.DENSE_LM:
            cautions.append(f"{pattern.value} wants vision; inferred shape is {shape.value} for {base_model!r}.")
        rationale.append("Freeze ladder: projector-only → LM+projector LoRA → encoder LoRA (~10× lower lr) only on plateau.")
        if pattern == JobPattern.VLM_CLASSIFIER:
            steps = 150
            seq = min(seq, 256)
            rationale.append("Classifier/rubric: short targets, lower seq budget.")
        if pattern == JobPattern.ROBOT_OFFLINE:
            steps = 300
            export = "onnx" if shape == ModelShape.EDGE_STUDENT else "peft"
            rationale.append("Robot offline: keep image refs in media store; same schema lab→edge.")
            cautions.append("Never actuate robot from raw sample without a supervisor.")
    elif pattern == JobPattern.RL_VERIFIABLE:
        mods = ("text",) if shape in {ModelShape.DENSE_LM, ModelShape.MOE_LM, ModelShape.UNKNOWN} else ("text", "image")
        steps = 500
        batch = max(1, batch // 2)
        lr = min(lr, 5e-5)
        rationale.append("On-policy RL: lower LR, more steps; sample must see current adapter.")
        rationale.append("GRPO practice: group 8-16 (lab) / 64 (paper); rollout temp ~1.0; KL 0-0.04; clip 0.2.")
        cautions.append("Reward is client-side; worker only sees named RL losses + advantages.")
        cautions.append("Generation dominates wall-clock (the Phase 2 vLLM split exists for this); cap completion length against reward hacking.")
    elif pattern == JobPattern.PREFERENCE_DPO:
        mods = ("text",)
        steps = 200
        lr = min(lr, 5e-5)
        rationale.append("DPO: paired data; modest LR to avoid collapsing the reference gap.")
        rationale.append("DPO practice: beta≈0.1 (0.05-0.3); LoRA lr 10-50× below SFT; 1 epoch; ref = base with adapter disabled (peft disable_adapter) — no second model in memory.")
        cautions.append("Length bias is the classic DPO hack — monitor completion length + holdout; NLL term (RPO) helps when chosen responses are long.")
    else:  # SFT_CHAT
        mods = ("text",)
        if shape in {ModelShape.DENSE_VLM, ModelShape.EDGE_STUDENT}:
            mods = ("text", "image")
            rationale.append("Base is VLM-class; chat SFT still allows image parts in messages.")
        steps = 200
        rationale.append("Chat SFT: CE on assistant tokens only (renderer weights).")
        rationale.append("LoRA practice: alpha=2·rank (≡ alpha=rank at ~2× lr); rsLoRA stabilizes r≥64.")
        rationale.append("Data: 1k curated > 100k noisy; 1-3 epochs with holdout eval each.")

    if not shapes_compatible(shape, pattern):
        cautions.append("Shape/pattern pairing is unusual — review freeze masks and modalities.")

    return lora, lr, steps, batch, seq, mods, export, tuple(rationale), tuple(cautions)


def _next_patterns(pattern: JobPattern, shape: ModelShape) -> tuple[str, ...]:
    if pattern == JobPattern.SFT_CHAT:
        return (JobPattern.RL_VERIFIABLE.value, JobPattern.PREFERENCE_DPO.value)
    if pattern == JobPattern.VLM_SFT:
        return (JobPattern.VLM_CLASSIFIER.value, JobPattern.ROBOT_OFFLINE.value)
    if pattern == JobPattern.VLM_CLASSIFIER:
        return (JobPattern.ROBOT_OFFLINE.value,)
    if pattern == JobPattern.ROBOT_OFFLINE and shape == ModelShape.EDGE_STUDENT:
        return ()
    if pattern == JobPattern.RL_VERIFIABLE:
        return (JobPattern.PREFERENCE_DPO.value,)
    return (JobPattern.SFT_CHAT.value,)


def suggest_for_model(
    base_model: str,
    *,
    fetch_remote: bool = False,
    include_blocked: bool = False,
) -> dict[str, Any]:
    """UI helper: card-derived shape + gated catalog recipes for a base model."""
    from anvil.recipes.catalog import (
        default_recipe_id_for_shape,
        recipes_for_shape,
    )

    card = None
    try:
        from anvil.recipes.model_card import inspect_base_model

        card = inspect_base_model(base_model, fetch_remote=fetch_remote)
        shape = card.shape
        param_count = card.param_count
        has_vision = card.has_vision
    except Exception:
        shape = infer_shape(base_model)
        param_count = None
        has_vision = shape in {ModelShape.DENSE_VLM, ModelShape.EDGE_STUDENT}

    gated = recipes_for_shape(
        shape,
        param_count=param_count,
        has_vision=has_vision,
        include_blocked=include_blocked,
    )
    cards: list[dict[str, Any]] = []
    for row in gated:
        try:
            plan = plan_recipe(
                base_model=base_model,
                recipe_id=row["id"],
                shape=shape,
                card=card,
                use_card=False,
                force=row["gate"]["level"] == "blocked",
                record_override=False,  # preview enumeration, not a real override
            )
            cards.append(
                {
                    "recipe_id": row["id"],
                    "pattern": row["pattern"],
                    "title": row["title"],
                    "summary": row["summary"],
                    "group": row["group"],
                    "gate": row["gate"],
                    "plan": plan.to_public(),
                }
            )
        except ValueError:
            # blocked without force — still show gate row without plan
            cards.append(
                {
                    "recipe_id": row["id"],
                    "pattern": row["pattern"],
                    "title": row["title"],
                    "summary": row["summary"],
                    "group": row["group"],
                    "gate": row["gate"],
                    "plan": None,
                }
            )

    out: dict[str, Any] = {
        "base_model": base_model,
        "shape": shape.value,
        "shape_label": _shape_label(shape),
        "default_recipe_id": default_recipe_id_for_shape(shape),
        "recipes": cards,
        "catalog_count": len(list_patterns()),  # pattern count; see list_recipes for full
    }
    if card is not None:
        out["card"] = card.to_public()
        out["shape_confidence"] = card.shape_confidence
    return out


def _shape_label(shape: ModelShape) -> str:
    return {
        ModelShape.DENSE_LM: "Dense language model",
        ModelShape.DENSE_VLM: "Dense vision-language model",
        ModelShape.MOE_LM: "Mixture-of-experts LM",
        ModelShape.EDGE_STUDENT: "Edge-sized student (Jetson path)",
        ModelShape.UNKNOWN: "Unknown shape — defaults are conservative",
    }[shape]
