"""Experience → production patience priors (Expert-v2 / P.Recipes).

Personal and org recipe books accumulate ``stop_policy`` and observe summaries.
This module turns those into a **suggested patience** for a family×pattern so
the next production run can inherit lab learnings without hard-coding atlas
constants forever.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Sequence

# plateau / early-stop reason embeds patience: loss_plateau_patience_40
_PATIENCE_IN_REASON = re.compile(
    r"patience[_\s-]?(\d+)|ceiling_x(\d+)|floor_x(\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ExperienceSample:
    """One production-tagged learning from a promoted recipe or experience file."""

    family: str | None
    pattern: str
    patience: int | None
    early_stop_reason: str | None = None
    steps_run: int | None = None
    loss_first: float | None = None
    loss_last: float | None = None
    source_recipe_id: str | None = None
    source_run_id: str | None = None
    production: bool = True
    notes: str = ""

    def to_public(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PatiencePrior:
    """Aggregated production patience for a family×pattern."""

    family: str | None
    pattern: str
    suggested_patience: int | None
    n_samples: int
    patience_values: tuple[int, ...] = ()
    source: str = "experience"  # experience | atlas_fallback
    notes: str = ""

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        d["patience_values"] = list(self.patience_values)
        return d


def parse_patience_from_reason(reason: str | None) -> int | None:
    """Extract integer patience from early-stop reason strings."""
    if not reason:
        return None
    m = _PATIENCE_IN_REASON.search(str(reason))
    if not m:
        return None
    for g in m.groups():
        if g is not None:
            return int(g)
    return None


def sample_from_book_recipe(recipe: Any) -> ExperienceSample | None:
    """Build an experience sample from a :class:`BookRecipe` if production-tagged."""
    policy = dict(getattr(recipe, "stop_policy", None) or {})
    mode = str(policy.get("mode") or "production").lower()
    if mode not in {"production", "prod"}:
        return None  # calibration runs must not shift production priors

    patience = policy.get("patience") or policy.get("early_stop_patience")
    if patience is not None:
        try:
            patience = int(patience)
        except (TypeError, ValueError):
            patience = None
    reason = policy.get("last_early_stop_reason")
    if patience is None:
        patience = parse_patience_from_reason(
            reason if isinstance(reason, str) else None
        )

    # Optional structured experience blob
    exp = policy.get("experience") if isinstance(policy.get("experience"), dict) else {}
    if patience is None and exp.get("patience") is not None:
        try:
            patience = int(exp["patience"])
        except (TypeError, ValueError):
            patience = None

    if patience is None and reason is None and not exp:
        # Still record if we have observe summary knobs only — skip empty
        if not (getattr(recipe, "notes", None) or "").strip():
            return None

    steps = exp.get("n_steps") or exp.get("steps_run")
    try:
        steps_i = int(steps) if steps is not None else None
    except (TypeError, ValueError):
        steps_i = None

    def _f(key: str) -> float | None:
        v = exp.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return ExperienceSample(
        family=getattr(recipe, "family", None),
        pattern=str(getattr(recipe, "pattern", None) or "sft_chat"),
        patience=patience,
        early_stop_reason=str(reason) if reason else None,
        steps_run=steps_i,
        loss_first=_f("loss_first"),
        loss_last=_f("loss_last"),
        source_recipe_id=getattr(recipe, "id", None),
        source_run_id=getattr(recipe, "source_run_id", None),
        production=True,
        notes=(getattr(recipe, "notes", None) or "")[:200],
    )


def collect_experience_from_book(book: Any) -> list[ExperienceSample]:
    """Scan a recipe book (and optional org pack roots) for production samples."""
    out: list[ExperienceSample] = []
    for rec in book.list():
        s = sample_from_book_recipe(rec)
        if s is not None:
            out.append(s)
    return out


def aggregate_patience(
    samples: Sequence[ExperienceSample],
    *,
    family: str | None,
    pattern: str,
    atlas_fallback: int | None = None,
    min_samples: int = 1,
) -> PatiencePrior:
    """Median production patience for matching family×pattern samples."""
    pat = pattern.lower().strip()
    fam = (family or "").lower().strip() or None
    matched: list[ExperienceSample] = []
    for s in samples:
        if not s.production:
            continue
        if s.pattern.lower().strip() != pat and pat not in s.pattern.lower():
            continue
        if fam:
            sf = (s.family or "").lower()
            if not sf or (fam not in sf and sf not in fam):
                continue
        if s.patience is not None and s.patience >= 1:
            matched.append(s)

    values = tuple(sorted(s.patience for s in matched if s.patience is not None))
    if len(values) >= min_samples:
        sug = int(median(values))
        return PatiencePrior(
            family=family,
            pattern=pattern,
            suggested_patience=sug,
            n_samples=len(values),
            patience_values=values,
            source="experience",
            notes=f"median of {len(values)} production book sample(s)",
        )
    return PatiencePrior(
        family=family,
        pattern=pattern,
        suggested_patience=atlas_fallback,
        n_samples=len(values),
        patience_values=values,
        source="atlas_fallback" if atlas_fallback is not None else "none",
        notes="insufficient production samples; atlas/default used"
        if atlas_fallback is not None
        else "no samples and no atlas fallback",
    )


def patience_prior_for_model(
    base_model: str,
    *,
    pattern: str = "vlm_sft",
    atlas_fallback: int | None = 40,
    book: Any | None = None,
) -> PatiencePrior:
    """End-to-end: infer family, collect book experience, aggregate patience."""
    from anvil.recipes.book import RecipeBook
    from anvil.recipes.profiles import infer_model_family
    from anvil.recipes.sft import DEFAULT_SFT_EARLY_STOP_PATIENCE

    fam = infer_model_family(base_model)
    b = book or RecipeBook()
    samples = collect_experience_from_book(b)
    fb = (
        atlas_fallback
        if atlas_fallback is not None
        else DEFAULT_SFT_EARLY_STOP_PATIENCE
    )
    return aggregate_patience(
        samples, family=fam, pattern=pattern, atlas_fallback=fb
    )


__all__ = [
    "ExperienceSample",
    "PatiencePrior",
    "aggregate_patience",
    "collect_experience_from_book",
    "parse_patience_from_reason",
    "patience_prior_for_model",
    "sample_from_book_recipe",
]
