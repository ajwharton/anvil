"""Meta-recipe skeleton — stage graphs and cliff→next edges (P.Recipes / Expert-v1).

A meta-recipe is policy *over* recipes: ordered stages plus optional edges when
a live signal fires (early-stop reason, advantage collapse, …). v0 is schema +
JSON load/save in the personal book root; execution still uses GRPO queue or
operator/agent choice.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from anvil.recipes.book import default_book_root

SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")


@dataclass
class MetaStage:
    """One stage in a ladder."""

    id: str
    recipe_id: str  # catalog id or personal book id
    source: str = "catalog"  # catalog | personal_book
    pattern: str | None = None
    notes: str = ""

    def to_public(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_public(cls, d: dict[str, Any]) -> MetaStage:
        return cls(
            id=str(d.get("id") or d.get("recipe_id") or "stage"),
            recipe_id=str(d["recipe_id"]),
            source=str(d.get("source") or "catalog"),
            pattern=None if d.get("pattern") is None else str(d["pattern"]),
            notes=str(d.get("notes") or ""),
        )


@dataclass
class MetaEdge:
    """Transition when a signal matches."""

    on: str  # e.g. early_stop:loss_plateau_*, advantage_collapse, manual
    from_stage: str
    to_stage: str
    notes: str = ""

    def to_public(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_public(cls, d: dict[str, Any]) -> MetaEdge:
        return cls(
            on=str(d["on"]),
            from_stage=str(d["from_stage"]),
            to_stage=str(d["to_stage"]),
            notes=str(d.get("notes") or ""),
        )


@dataclass
class MetaRecipe:
    """Named stage graph over recipes."""

    id: str
    title: str
    stages: list[MetaStage] = field(default_factory=list)
    edges: list[MetaEdge] = field(default_factory=list)
    family: str | None = None
    notes: str = ""
    created_ts: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "stages": [s.to_public() for s in self.stages],
            "edges": [e.to_public() for e in self.edges],
            "family": self.family,
            "notes": self.notes,
            "created_ts": self.created_ts,
            "schema_version": self.schema_version,
            "kind": "meta_recipe",
        }

    @classmethod
    def from_public(cls, d: dict[str, Any]) -> MetaRecipe:
        return cls(
            id=str(d["id"]),
            title=str(d.get("title") or d["id"]),
            stages=[MetaStage.from_public(s) for s in (d.get("stages") or [])],
            edges=[MetaEdge.from_public(e) for e in (d.get("edges") or [])],
            family=None if d.get("family") is None else str(d["family"]),
            notes=str(d.get("notes") or ""),
            created_ts=float(d.get("created_ts") or time.time()),
            schema_version=int(d.get("schema_version") or SCHEMA_VERSION),
        )


def meta_book_dir(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else default_book_root()
    d = base / "meta"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_meta_recipe(meta: MetaRecipe, *, root: str | Path | None = None) -> Path:
    if not _SAFE_ID.fullmatch(meta.id):
        raise ValueError(f"bad meta-recipe id: {meta.id!r}")
    path = meta_book_dir(root) / f"{meta.id}.json"
    path.write_text(
        json.dumps(meta.to_public(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def get_meta_recipe(meta_id: str, *, root: str | Path | None = None) -> MetaRecipe | None:
    if not _SAFE_ID.fullmatch(meta_id):
        raise ValueError(f"bad meta-recipe id: {meta_id!r}")
    path = meta_book_dir(root) / f"{meta_id}.json"
    if not path.is_file():
        return None
    return MetaRecipe.from_public(json.loads(path.read_text(encoding="utf-8")))


def list_meta_recipes(*, root: str | Path | None = None) -> list[MetaRecipe]:
    out: list[MetaRecipe] = []
    for p in sorted(meta_book_dir(root).glob("*.json")):
        try:
            out.append(MetaRecipe.from_public(json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out


def next_stage(
    meta: MetaRecipe,
    *,
    current_stage_id: str,
    signal: str | None = None,
) -> MetaStage | None:
    """Resolve next stage: matching edge on ``signal``, else sequential order."""
    if signal:
        for e in meta.edges:
            if e.from_stage != current_stage_id:
                continue
            if e.on == signal or e.on.endswith("*") and signal.startswith(e.on[:-1]):
                for s in meta.stages:
                    if s.id == e.to_stage:
                        return s
    ids = [s.id for s in meta.stages]
    if current_stage_id not in ids:
        return meta.stages[0] if meta.stages else None
    i = ids.index(current_stage_id)
    if i + 1 < len(meta.stages):
        return meta.stages[i + 1]
    return None


def example_vlm_ladder(*, family: str = "Qwen2.5-VL") -> MetaRecipe:
    """Shipped example meta-recipe (not auto-executed)."""
    return MetaRecipe(
        id="vlm-sft-then-export",
        title="VLM SFT then export",
        family=family,
        stages=[
            MetaStage(
                id="sft",
                recipe_id="vlm_sft_edge",
                source="catalog",
                pattern="vlm_sft",
                notes="Production early-stop on loss plateau",
            ),
            MetaStage(
                id="export",
                recipe_id="vlm_sft_edge",
                source="catalog",
                pattern="vlm_sft",
                notes="Export PEFT after early-stop (operator/agent act)",
            ),
        ],
        edges=[
            MetaEdge(
                on="early_stop:loss_plateau*",
                from_stage="sft",
                to_stage="export",
                notes="Dogfood: stop training when plateau fires",
            )
        ],
        notes="Skeleton for agent/MCP; GRPO queue remains the executed multi-stage path today.",
    )


__all__ = [
    "MetaEdge",
    "MetaRecipe",
    "MetaStage",
    "example_vlm_ladder",
    "get_meta_recipe",
    "list_meta_recipes",
    "meta_book_dir",
    "next_stage",
    "save_meta_recipe",
]
