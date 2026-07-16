"""Renderer contract — same expansion for train and sample."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from anvil.protocol.messages import Example, Message
from anvil.protocol.types import Datum, ModelInput


@runtime_checkable
class Renderer(Protocol):
    """Expand chat/multimodal messages → model tokens (and optional Datum)."""

    name: str

    def render_messages(self, messages: Sequence[Message]) -> ModelInput: ...

    def render_example_for_sft(self, example: Example) -> Datum: ...
