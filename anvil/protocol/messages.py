"""Multimodal chat message schema (vision first-class).

Batches for high-level recipes use this schema; renderers expand to ModelInput.
Media blobs are referenced by content-addressed refs, not always inlined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class TextPart:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass(frozen=True, slots=True)
class ImagePart:
    """Image as a media-store ref (preferred)."""

    type: Literal["image"] = "image"
    ref: str = ""
    detail: str = "auto"


@dataclass(frozen=True, slots=True)
class ImageUrlPart:
    """Optional URL part for ingestion; store materializes to a ref."""

    type: Literal["image_url"] = "image_url"
    url: str = ""


ContentPart = TextPart | ImagePart | ImageUrlPart


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: tuple[ContentPart, ...] | str

    def parts(self) -> tuple[ContentPart, ...]:
        if isinstance(self.content, str):
            return (TextPart(text=self.content),)
        return tuple(self.content)


@dataclass(frozen=True, slots=True)
class Example:
    """One supervised or trajectory-bearing example."""

    messages: tuple[Message, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple):
            object.__setattr__(self, "messages", tuple(self.messages))
