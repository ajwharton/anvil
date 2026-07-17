"""J-Lens observe artifacts (Phase 2.5 / J1).

Debugger-plane only: append small residual-readout records to
``jlens.jsonl`` under a run dir. No dependency on ``anthropics/jacobian-lens``
— real apply lives in the forge spike / later worker; this module is the
stable **schema + tripwire helpers + writer**.

Record shape (schema_version 1)::

    {
      "type": "jlens",
      "step": 12,
      "probe_idx": 0,
      "prompt_preview": "…",
      "completion_preview": "…",   # optional
      "layers": [8, 12, 16],
      "positions": "last_prompt",
      "top_k": 5,
      "slice": {"12": {"-1": [{"tok": "14", "rank": 1}, …]}},
      "signals": {
        "answer_token_min_rank": 1,
        "intermediate_order_score": 0.82,
        "stage_layers": [4, 8, 12],
        "off_task_mass": null
      },
      "lens_id": "qwen2.5-1.5b-base-v0",
      "adapter_id": "adapter-…",
      "wall_time_s": 1.4
    }

Gate / scoring helpers mirror ``scripts/jlens_spike.py`` so product code and
the forge spike stay aligned.
"""

from __future__ import annotations

import re
import time
from typing import Any, Mapping, Sequence

from anvil.observe.metrics import _append_jsonl

JLENS_FILENAME = "jlens.jsonl"
JLENS_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Scoring (shared with forge spike semantics)
# ---------------------------------------------------------------------------


def _norm_tok(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())


def top_tokens_contain(top_strings: Sequence[str], candidates: Sequence[str]) -> bool:
    tops = {_norm_tok(t) for t in top_strings}
    for c in candidates:
        nc = _norm_tok(c)
        if not nc:
            continue
        if nc in tops:
            return True
        if any(nc in t or t in nc for t in tops if t):
            return True
    return False


def _normalize_layer_tops(
    layer_tops: Mapping[int | str, Sequence[str]],
) -> dict[int, list[str]]:
    return {int(k): list(v) for k, v in layer_tops.items()}


def earliest_stage_layers(
    layer_tops: Mapping[int | str, Sequence[str]],
    stages: Sequence[Sequence[str]],
) -> list[int | None]:
    """Earliest layer where each stage has a candidate in top-k tokens."""
    tops = _normalize_layer_tops(layer_tops)
    layers_sorted = sorted(tops.keys())
    out: list[int | None] = []
    for stage in stages:
        hit: int | None = None
        for L in layers_sorted:
            if top_tokens_contain(tops[L], stage):
                hit = L
                break
        out.append(hit)
    return out


def intermediate_order_score(stage_layers: Sequence[int | None]) -> float | None:
    """Fraction of consecutive *hit* stage pairs with non-decreasing layer index."""
    known = [(i, L) for i, L in enumerate(stage_layers) if L is not None]
    if len(known) < 2:
        return None
    ok = 0
    total = 0
    for (_, la), (_, lb) in zip(known, known[1:]):
        total += 1
        if lb >= la:
            ok += 1
    return ok / total if total else None


def answer_min_rank(
    layer_tops: Mapping[int | str, Sequence[str]], answer: str
) -> int | None:
    """Best (lowest) 1-based rank of ``answer`` across layers; None if absent."""
    best: int | None = None
    for tops in layer_tops.values():
        for i, t in enumerate(tops):
            if _norm_tok(answer) in _norm_tok(t) or _norm_tok(t) in _norm_tok(answer):
                rank = i + 1
                best = rank if best is None else min(best, rank)
                break
    return best


def layer_tops_from_slice(
    slice_: Mapping[str, Mapping[str, Sequence[Mapping[str, Any] | str]]],
    *,
    position: str = "-1",
) -> dict[int, list[str]]:
    """Extract layer → top token strings from a compact slice record."""
    out: dict[int, list[str]] = {}
    for layer_s, pos_map in slice_.items():
        L = int(layer_s)
        cells = pos_map.get(position) or pos_map.get(str(position))
        if cells is None and pos_map:
            # first position present
            cells = next(iter(pos_map.values()))
        if not cells:
            out[L] = []
            continue
        tops: list[str] = []
        for c in cells:
            if isinstance(c, str):
                tops.append(c)
            elif isinstance(c, Mapping):
                tops.append(str(c.get("tok", c.get("token", ""))))
            else:
                tops.append(str(c))
        out[L] = tops
    return out


def compute_signals(
    *,
    slice_: Mapping[str, Any] | None = None,
    layer_tops: Mapping[int | str, Sequence[str]] | None = None,
    stages: Sequence[Sequence[str]] | None = None,
    answer: str | None = None,
    position: str = "-1",
) -> dict[str, Any]:
    """Derive scalar tripwires from a slice or explicit layer tops."""
    tops: dict[int, list[str]]
    if layer_tops is not None:
        tops = {int(k): list(v) for k, v in layer_tops.items()}
    elif slice_ is not None:
        tops = layer_tops_from_slice(slice_, position=position)
    else:
        tops = {}

    stage_layers: list[int | None] | None = None
    order: float | None = None
    if stages is not None and tops:
        stage_layers = earliest_stage_layers(tops, stages)
        order = intermediate_order_score(stage_layers)

    ans_rank = answer_min_rank(tops, answer) if answer and tops else None
    return {
        "answer_token_min_rank": ans_rank,
        "intermediate_order_score": order,
        "stage_layers": stage_layers,
        "off_task_mass": None,  # reserved — needs full vocab mass, not top-k only
    }


def jlens_order_collapsed(
    record: Mapping[str, Any], *, min_order: float = 0.6
) -> bool:
    """Tripwire: intermediate order score present and below threshold."""
    signals = record.get("signals") if isinstance(record.get("signals"), Mapping) else record
    score = signals.get("intermediate_order_score") if isinstance(signals, Mapping) else None
    return score is not None and float(score) < min_order


# ---------------------------------------------------------------------------
# Record builder + write
# ---------------------------------------------------------------------------


def build_jlens_record(
    *,
    step: int,
    probe_idx: int | None = None,
    prompt_preview: str | None = None,
    completion_preview: str | None = None,
    layers: Sequence[int] | None = None,
    positions: str | Sequence[int] = "last_prompt",
    top_k: int = 5,
    slice_: Mapping[str, Any] | None = None,
    signals: Mapping[str, Any] | None = None,
    stages: Sequence[Sequence[str]] | None = None,
    answer: str | None = None,
    lens_id: str | None = None,
    adapter_id: str | None = None,
    wall_time_s: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one schema_version=1 jlens record (does not write)."""
    sl = dict(slice_ or {})
    if signals is None:
        sig = compute_signals(slice_=sl or None, stages=stages, answer=answer)
    else:
        sig = dict(signals)

    pos_field: Any
    if isinstance(positions, str):
        pos_field = positions
    else:
        pos_field = [int(p) for p in positions]

    layer_list = list(layers) if layers is not None else sorted(int(k) for k in sl.keys())

    record: dict[str, Any] = {
        "schema_version": JLENS_SCHEMA_VERSION,
        "type": "jlens",
        "ts": time.time(),
        "step": int(step),
        "probe_idx": None if probe_idx is None else int(probe_idx),
        "prompt_preview": prompt_preview,
        "completion_preview": completion_preview,
        "layers": layer_list,
        "positions": pos_field,
        "top_k": int(top_k),
        "slice": sl,
        "signals": sig,
        "lens_id": lens_id,
        "adapter_id": adapter_id,
        "wall_time_s": wall_time_s,
    }
    if extra:
        for k, v in extra.items():
            if k not in record:
                record[k] = v
    return record


def append_jlens_record(run_dir: str | Any, record: Mapping[str, Any]) -> dict[str, Any]:
    """Append a pre-built record to ``<run_dir>/jlens.jsonl``."""
    from pathlib import Path

    path = Path(run_dir) / JLENS_FILENAME
    rec = dict(record)
    if "schema_version" not in rec:
        rec["schema_version"] = JLENS_SCHEMA_VERSION
    if "type" not in rec:
        rec["type"] = "jlens"
    if "ts" not in rec:
        rec["ts"] = time.time()
    _append_jsonl(path, rec)
    return rec
