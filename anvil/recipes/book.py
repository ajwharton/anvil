"""Personal recipe book — operator-owned learnings (P.Recipes v0).

Shipped atlas lives in ``catalog.py``. This module stores **local** recipes
under ``ANVIL_RECIPE_BOOK`` (default ``~/.anvil/recipes``): promote a finished
run's knobs/stop policy into a versioned JSON file the next plan can prefer.

Sovereign: nothing is uploaded; the forge owns its book.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")


def default_book_root() -> Path:
    env = os.environ.get("ANVIL_RECIPE_BOOK")
    if env:
        return Path(env)
    return Path.home() / ".anvil" / "recipes"


@dataclass
class BookRecipe:
    """One entry in a personal (or org) recipe book."""

    id: str
    title: str
    pattern: str = "vlm_sft"
    family: str | None = None
    job: str | None = None
    knobs: dict[str, Any] = field(default_factory=dict)
    stop_policy: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    source_run_id: str | None = None
    source_run_dir: str | None = None
    tags: list[str] = field(default_factory=list)
    created_ts: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION

    def to_public(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_public(cls, d: dict[str, Any]) -> BookRecipe:
        return cls(
            id=str(d["id"]),
            title=str(d.get("title") or d["id"]),
            pattern=str(d.get("pattern") or "vlm_sft"),
            family=None if d.get("family") is None else str(d["family"]),
            job=None if d.get("job") is None else str(d["job"]),
            knobs=dict(d.get("knobs") or {}),
            stop_policy=dict(d.get("stop_policy") or {}),
            notes=str(d.get("notes") or ""),
            source_run_id=None if d.get("source_run_id") is None else str(d["source_run_id"]),
            source_run_dir=None if d.get("source_run_dir") is None else str(d["source_run_dir"]),
            tags=[str(t) for t in (d.get("tags") or [])],
            created_ts=float(d.get("created_ts") or time.time()),
            schema_version=int(d.get("schema_version") or SCHEMA_VERSION),
        )


class RecipeBook:
    """Filesystem-backed personal recipe library."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_book_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, recipe_id: str) -> Path:
        if not _SAFE_ID.fullmatch(recipe_id):
            raise ValueError(f"bad recipe id: {recipe_id!r}")
        return self.root / f"{recipe_id}.json"

    def save(self, recipe: BookRecipe) -> Path:
        if not _SAFE_ID.fullmatch(recipe.id):
            raise ValueError(f"bad recipe id: {recipe.id!r}")
        path = self._path(recipe.id)
        path.write_text(
            json.dumps(recipe.to_public(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def get(self, recipe_id: str) -> BookRecipe | None:
        path = self._path(recipe_id)
        if not path.is_file():
            return None
        return BookRecipe.from_public(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[BookRecipe]:
        out: list[BookRecipe] = []
        for p in sorted(self.root.glob("*.json")):
            try:
                out.append(BookRecipe.from_public(json.loads(p.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return out

    def prefer(
        self,
        *,
        pattern: str | None = None,
        family: str | None = None,
    ) -> list[BookRecipe]:
        """Recipes matching pattern and/or family (substring, case-insensitive)."""
        rows = self.list()
        if pattern:
            p = pattern.lower()
            rows = [r for r in rows if r.pattern.lower() == p or p in r.pattern.lower()]
        if family:
            f = family.lower()
            rows = [
                r
                for r in rows
                if r.family and (r.family.lower() == f or f in r.family.lower())
            ]
        return rows


def _summarize_metrics(run_dir: str | Path | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    path = Path(run_dir) / "metrics.jsonl"
    if not path.is_file():
        return {}
    losses: list[float] = []
    last_job = None
    early = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") == "event" and rec.get("event") == "early_stop":
            early = rec.get("reason")
        if rec.get("loss") is not None and rec.get("type") in (None, "step"):
            losses.append(float(rec["loss"]))
            last_job = rec.get("job") or last_job
    out: dict[str, Any] = {"n_steps": len(losses), "job": last_job}
    if losses:
        out["loss_first"] = losses[0]
        out["loss_last"] = losses[-1]
    if early:
        out["early_stop_reason"] = early
    return out


def promote_from_run(
    *,
    recipe_id: str,
    title: str | None = None,
    run_dir: str | Path | None = None,
    run_id: str | None = None,
    pattern: str = "vlm_sft",
    family: str | None = None,
    job: str | None = None,
    knobs: dict[str, Any] | None = None,
    stop_policy: dict[str, Any] | None = None,
    notes: str = "",
    tags: Sequence[str] | None = None,
    book: RecipeBook | None = None,
    early_stop_reason: str | None = None,
) -> BookRecipe:
    """Create/update a personal recipe from a finished train run."""
    book = book or RecipeBook()
    summary = _summarize_metrics(run_dir)
    if early_stop_reason and "early_stop_reason" not in summary:
        summary["early_stop_reason"] = early_stop_reason
    note_bits = [notes.strip()] if notes.strip() else []
    if summary:
        note_bits.append(f"observe_summary={json.dumps(summary, sort_keys=True)}")
    policy = dict(stop_policy or {})
    if early_stop_reason and "last_early_stop_reason" not in policy:
        policy["last_early_stop_reason"] = early_stop_reason
    if "mode" not in policy:
        policy["mode"] = "production"
    rec = BookRecipe(
        id=recipe_id,
        title=title or recipe_id,
        pattern=pattern,
        family=family,
        job=job or summary.get("job"),
        knobs=dict(knobs or {}),
        stop_policy=policy,
        notes="\n".join(note_bits),
        source_run_id=run_id,
        source_run_dir=str(run_dir) if run_dir else None,
        tags=list(tags or []),
    )
    book.save(rec)
    return rec


__all__ = [
    "BookRecipe",
    "RecipeBook",
    "default_book_root",
    "promote_from_run",
]
