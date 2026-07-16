"""Adapter registry — maps adapter ids to base model + metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from anvil.protocol.types import AdapterId, TrainConfig


@dataclass
class AdapterRecord:
    adapter_id: AdapterId
    config: TrainConfig
    labels: dict[str, str] = field(default_factory=dict)


class AdapterRegistry:
    """In-process registry (Phase 0). Later: durable control-plane store."""

    def __init__(self) -> None:
        self._by_id: dict[str, AdapterRecord] = {}

    def register(self, adapter_id: AdapterId, config: TrainConfig, **labels: str) -> AdapterRecord:
        rec = AdapterRecord(adapter_id=adapter_id, config=config, labels=dict(labels))
        self._by_id[adapter_id.value] = rec
        return rec

    def get(self, adapter_id: AdapterId | str) -> AdapterRecord:
        key = adapter_id.value if isinstance(adapter_id, AdapterId) else adapter_id
        try:
            return self._by_id[key]
        except KeyError as e:
            raise KeyError(f"unknown adapter: {key}") from e

    def __contains__(self, adapter_id: AdapterId | str) -> bool:
        key = adapter_id.value if isinstance(adapter_id, AdapterId) else adapter_id
        return key in self._by_id

    def __iter__(self) -> Iterator[AdapterRecord]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)
