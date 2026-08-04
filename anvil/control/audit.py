"""Audit trail — the control plane's record of consequential actions.

Phase 2 starts small: every ``force=True`` past a **blocked** recipe gate is
logged with recipe, shape, and reasons. Phase 5 builds the multi-user audit
log on top of these events.

Events are append-only and persisted to a JSONL sink (default
``~/.anvil/audit.jsonl``, override with ``ANVIL_AUDIT_LOG``) so the trail
survives process restarts. `anvil-web` exposes them at ``/api/audit``.
"""

from __future__ import annotations

import json
import os
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

    @classmethod
    def from_public(cls, d: dict[str, Any]) -> AuditEvent:
        return cls(
            kind=str(d.get("kind") or "gate_override"),
            at=str(d.get("at") or ""),
            recipe_id=str(d.get("recipe_id") or ""),
            base_model=str(d.get("base_model") or ""),
            shape=str(d.get("shape") or ""),
            blocked_reasons=tuple(str(x) for x in (d.get("blocked_reasons") or ())),
            stretch_reasons=tuple(str(x) for x in (d.get("stretch_reasons") or ())),
            detail=str(d.get("detail") or ""),
        )


def default_audit_path() -> Path:
    env = os.environ.get("ANVIL_AUDIT_LOG")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".anvil" / "audit.jsonl"


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
    """Append-only audit log with a JSONL sink (default on)."""

    def __init__(self, jsonl_path: str | Path | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._sink = (
            Path(jsonl_path).expanduser()
            if jsonl_path is not None
            else default_audit_path()
        )
        self._sink.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        """Replay prior events from the sink so restarts keep the trail."""
        if not self._sink.is_file():
            return
        for line in self._sink.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self._events.append(AuditEvent.from_public(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    def record(self, event: AuditEvent) -> None:
        self._events.append(event)
        with self._sink.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_public()) + "\n")

    def events(self, *, kind: str | None = None) -> list[AuditEvent]:
        if kind is None:
            return list(self._events)
        return [e for e in self._events if e.kind == kind]

    def clear(self) -> None:
        """Drop in-memory events (the JSONL sink, if any, is untouched)."""
        self._events.clear()


_default_log: AuditLog | None = None


def default_log() -> AuditLog:
    """Process-local default log — the one `anvil-web` serves at /api/audit.

    Lazily constructed so ``ANVIL_AUDIT_LOG`` (or the home default) is read at
    first use, not at import time.
    """
    global _default_log
    if _default_log is None:
        _default_log = AuditLog()
    return _default_log


def reset_default_log() -> None:
    """Drop the cached default log (tests use this to isolate the sink)."""
    global _default_log
    _default_log = None
