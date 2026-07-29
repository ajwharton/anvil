"""Personal / org recipe book — operator-owned learnings (P.Recipes).

Shipped atlas lives in ``catalog.py``. This module stores **local** recipes
under ``ANVIL_RECIPE_BOOK`` (default ``~/.anvil/recipes``): promote a finished
run's knobs/stop policy into a versioned JSON file the next plan can prefer.

**Org packs:** set ``ANVIL_ORG_RECIPE_PACK`` (or ``ANVIL_RECIPE_PACK``) to a
directory of recipe JSON files (optional ``manifest.json``). List/get merge
personal + org; **writes always go to the personal root**.

Sovereign: nothing is uploaded; the forge / org owns its book.
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


def org_pack_roots() -> list[Path]:
    """Extra read-only roots for org recipe packs (env)."""
    out: list[Path] = []
    for key in ("ANVIL_ORG_RECIPE_PACK", "ANVIL_RECIPE_PACK"):
        raw = os.environ.get(key)
        if not raw:
            continue
        p = Path(raw).expanduser()
        if p.is_dir() and p not in out:
            out.append(p)
    return out


def book_search_roots(*, personal: Path | None = None) -> list[Path]:
    """Personal write root first, then org pack roots."""
    roots: list[Path] = []
    pr = personal if personal is not None else default_book_root()
    roots.append(Path(pr))
    for o in org_pack_roots():
        if o.resolve() != Path(pr).resolve() and o not in roots:
            roots.append(o)
    return roots


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
    """Filesystem-backed personal recipe library (+ optional org pack search)."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        search_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else default_book_root()
        self.root.mkdir(parents=True, exist_ok=True)
        if search_roots is not None:
            self.search_roots = [Path(p) for p in search_roots]
        else:
            self.search_roots = book_search_roots(personal=self.root)

    def _path(self, recipe_id: str, *, root: Path | None = None) -> Path:
        if not _SAFE_ID.fullmatch(recipe_id):
            raise ValueError(f"bad recipe id: {recipe_id!r}")
        base = root if root is not None else self.root
        # org packs may nest under recipes/
        nested = base / "recipes" / f"{recipe_id}.json"
        if nested.is_file():
            return nested
        return base / f"{recipe_id}.json"

    def save(self, recipe: BookRecipe) -> Path:
        if not _SAFE_ID.fullmatch(recipe.id):
            raise ValueError(f"bad recipe id: {recipe.id!r}")
        path = self.root / f"{recipe.id}.json"
        path.write_text(
            json.dumps(recipe.to_public(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def get(self, recipe_id: str) -> BookRecipe | None:
        # Prefer personal root first
        for root in self.search_roots:
            for candidate in (
                root / f"{recipe_id}.json",
                root / "recipes" / f"{recipe_id}.json",
            ):
                if candidate.is_file():
                    try:
                        return BookRecipe.from_public(
                            json.loads(candidate.read_text(encoding="utf-8"))
                        )
                    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
        return None

    def list(self) -> list[BookRecipe]:
        """Union of personal + org pack recipes (personal wins on id conflict)."""
        by_id: dict[str, BookRecipe] = {}
        # Org first, then personal overwrites — personal wins
        for root in reversed(self.search_roots):
            paths = list(root.glob("*.json")) + list((root / "recipes").glob("*.json"))
            for p in sorted(paths):
                if p.name == "manifest.json":
                    continue
                try:
                    rec = BookRecipe.from_public(json.loads(p.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                by_id[rec.id] = rec
        return [by_id[k] for k in sorted(by_id)]

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
    patience: int | None = None,
) -> BookRecipe:
    """Create/update a personal recipe from a finished train run.

    Stores structured ``stop_policy.experience`` so
    :mod:`anvil.recipes.experience` can learn production patience.
    """
    from anvil.recipes.experience import parse_patience_from_reason

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
    # Structured experience for patience aggregation
    exp = dict(policy.get("experience") or {})
    if summary.get("n_steps") is not None:
        exp.setdefault("n_steps", summary["n_steps"])
    if summary.get("loss_first") is not None:
        exp.setdefault("loss_first", summary["loss_first"])
    if summary.get("loss_last") is not None:
        exp.setdefault("loss_last", summary["loss_last"])
    pat = patience
    if pat is None:
        pat = policy.get("patience") or policy.get("early_stop_patience")
    if pat is None:
        pat = parse_patience_from_reason(
            early_stop_reason or summary.get("early_stop_reason")
        )
    if pat is not None:
        try:
            pat_i = int(pat)
            policy["patience"] = pat_i
            exp["patience"] = pat_i
        except (TypeError, ValueError):
            pass
    if exp:
        policy["experience"] = exp

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


def install_org_pack(
    pack_dir: str | Path,
    *,
    book: RecipeBook | None = None,
    prefix: str | None = None,
    tag: str = "org_pack",
) -> list[BookRecipe]:
    """Copy recipes from an org pack directory into the personal book.

    Pack layout::

        pack/
          manifest.json          # optional {name, version, recipes: [ids]}
          *.json                 # BookRecipe files
          recipes/*.json         # alternate nest

    Personal copies get tag ``org_pack`` (and optional id prefix).
    """
    pack = Path(pack_dir)
    if not pack.is_dir():
        raise FileNotFoundError(f"org pack not found: {pack}")
    book = book or RecipeBook()
    manifest_ids: set[str] | None = None
    man = pack / "manifest.json"
    if man.is_file():
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
            if isinstance(data.get("recipes"), list):
                manifest_ids = {str(x) for x in data["recipes"]}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    paths = list(pack.glob("*.json")) + list((pack / "recipes").glob("*.json"))
    installed: list[BookRecipe] = []
    for p in sorted(paths):
        if p.name == "manifest.json":
            continue
        try:
            rec = BookRecipe.from_public(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if manifest_ids is not None and rec.id not in manifest_ids:
            continue
        new_id = f"{prefix}{rec.id}" if prefix else rec.id
        if not _SAFE_ID.fullmatch(new_id):
            continue
        tags = list(rec.tags)
        if tag and tag not in tags:
            tags.append(tag)
        out = BookRecipe(
            id=new_id,
            title=rec.title,
            pattern=rec.pattern,
            family=rec.family,
            job=rec.job,
            knobs=dict(rec.knobs),
            stop_policy=dict(rec.stop_policy),
            notes=rec.notes,
            source_run_id=rec.source_run_id,
            source_run_dir=rec.source_run_dir,
            tags=tags,
            created_ts=time.time(),
        )
        book.save(out)
        installed.append(out)
    return installed


def export_org_pack(
    dest: str | Path,
    *,
    book: RecipeBook | None = None,
    recipe_ids: Sequence[str] | None = None,
    name: str = "org-pack",
    version: str = "1",
) -> Path:
    """Write selected (or all) personal recipes into an org pack directory."""
    book = book or RecipeBook()
    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    recipes_dir = dest_p / "recipes"
    recipes_dir.mkdir(exist_ok=True)
    selected = list(book.list())
    if recipe_ids is not None:
        want = set(recipe_ids)
        selected = [r for r in selected if r.id in want]
    ids: list[str] = []
    for r in selected:
        path = recipes_dir / f"{r.id}.json"
        path.write_text(
            json.dumps(r.to_public(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        ids.append(r.id)
    manifest = {
        "name": name,
        "version": version,
        "schema_version": SCHEMA_VERSION,
        "recipes": ids,
        "created_ts": time.time(),
    }
    (dest_p / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return dest_p


__all__ = [
    "BookRecipe",
    "RecipeBook",
    "book_search_roots",
    "default_book_root",
    "export_org_pack",
    "install_org_pack",
    "org_pack_roots",
    "promote_from_run",
]
