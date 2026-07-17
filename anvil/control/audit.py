"""Audit trail — the control plane's record of consequential actions.

Phase 2 starts small: every ``force=True`` past a **blocked** recipe gate is
logged with recipe, shape, and reasons. Phase 5 builds the multi-user audit
log on top of these events.

Events live in an append-only, process-local in-memory log with an optional
JSONL sink for durability. `anvil-web` exposes them at ``/api/audit``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AuditEvent:
    kind: str  # "gate_override" (more kinds land with later phases)
    at: str  # ISO-8601 UTC
    recipe_id: str
    base_model: str
    shape: str
    blocked_reasons: tuple[str, ...] = ()
    stretch_reasons: tuple[str, ...] = ()
    detail: str = ""

    def to_public(self) -> dict[str, Any]:
        return asdict(self)


def gate_override_event(
    *,
    recipe_id: str,
    base_model: str,
    shape: str,
    blocked_reasons: tuple[str, ...],
    stretch_reasons: tuple[str, ...],
) -> AuditEvent:
    return AuditEvent(
        kind="gate_override",
        at=datetime.now(timezone.utc).isoformat(),
        recipe_id=recipe_id,
        base_model=base_model,
        shape=shape,
        blocked_reasons=tuple(blocked_reasons),
        stretch_reasons=tuple(stretch_reasons),
    )


class AuditLog:
    """Append-only in-memory log with optional JSONL sink."""

    def __init__(self, jsonl_path: str | Path | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._sink = Path(jsonl_path) if jsonl_path is not None else None
        if self._sink is not None:
            self._sink.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: AuditEvent) -> None:
        self._events.append(event)
        if self._sink is not None:
            with self._sink.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_public()) + "\n")

    def events(self, *, kind: str | None = None) -> list[AuditEvent]:
        if kind is None:
            return list(self._events)
        return [e for e in self._events if e.kind == kind]

    def clear(self) -> None:
        """Drop in-memory events (the JSONL sink, if any, is untouched)."""
        self._events.clear()


_default_log = AuditLog()


def default_log() -> AuditLog:
    """Process-local default log — the one `anvil-web` serves at /api/audit."""
    return _default_log
