"""Session / control-plane stubs (local-first, no auth yet)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from anvil.control.registry import AdapterRegistry
from anvil.protocol.types import SessionId


@dataclass
class Session:
    """Client-facing session handle. Multi-tenant auth lands in Phase 5."""

    session_id: SessionId
    endpoint: str
    registry: AdapterRegistry = field(default_factory=AdapterRegistry)

    @classmethod
    def create(cls, endpoint: str) -> Session:
        return cls(
            session_id=SessionId(value=f"sess-{uuid.uuid4().hex[:12]}"),
            endpoint=endpoint,
        )
