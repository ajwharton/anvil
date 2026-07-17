"""Multimodal chat message schema (vision first-class).

Batches for high-level recipes use this schema; renderers expand to ModelInput.
Media blobs are referenced by content-addressed refs, not always inlined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class TextPart:
    type: Literal["text"] = "text"
    text: str = ""

    def to_public(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True, slots=True)
class ImagePart:
    """Image as a media-store ref (preferred)."""

    type: Literal["image"] = "image"
    ref: str = ""
    detail: str = "auto"

    def to_public(self) -> dict[str, Any]:
        return {"type": "image", "ref": self.ref, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ImageUrlPart:
    """Optional URL part for ingestion; store materializes to a ref."""

    type: Literal["image_url"] = "image_url"
    url: str = ""

    def to_public(self) -> dict[str, Any]:
        return {"type": "image_url", "url": self.url}


ContentPart = TextPart | ImagePart | ImageUrlPart


def content_part_from_public(d: Mapping[str, Any]) -> ContentPart:
    t = str(d.get("type", "text"))
    if t == "text":
        return TextPart(text=str(d.get("text", "")))
    if t == "image":
        return ImagePart(ref=str(d.get("ref", "")), detail=str(d.get("detail", "auto")))
    if t == "image_url":
        return ImageUrlPart(url=str(d.get("url", "")))
    raise ValueError(f"unknown content part type: {t!r}")


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: tuple[ContentPart, ...] | str

    def parts(self) -> tuple[ContentPart, ...]:
        if isinstance(self.content, str):
            return (TextPart(text=self.content),)
        return tuple(self.content)

    def to_public(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            content: Any = self.content
        else:
            content = [p.to_public() for p in self.content]
        return {"role": self.role, "content": content}

    @classmethod
    def from_public(cls, d: Mapping[str, Any]) -> Message:
        role = str(d["role"])  # type: ignore[arg-type]
        raw = d.get("content", "")
        if isinstance(raw, str):
            return cls(role=role, content=raw)  # type: ignore[arg-type]
        if isinstance(raw, Sequence):
            parts = tuple(content_part_from_public(p) for p in raw)  # type: ignore[arg-type]
            return cls(role=role, content=parts)  # type: ignore[arg-type]
        raise TypeError(f"message content must be str or list, got {type(raw)}")


@dataclass(frozen=True, slots=True)
class Example:
    """One supervised or trajectory-bearing example."""

    messages: tuple[Message, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple):
            object.__setattr__(self, "messages", tuple(self.messages))

    def to_public(self) -> dict[str, Any]:
        return {
            "messages": [m.to_public() for m in self.messages],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_public(cls, d: Mapping[str, Any]) -> Example:
        msgs = tuple(Message.from_public(m) for m in d.get("messages", []))
        meta = dict(d.get("meta") or {})
        return cls(messages=msgs, meta=meta)

    def image_refs(self) -> list[str]:
        refs: list[str] = []
        for m in self.messages:
            for p in m.parts():
                if isinstance(p, ImagePart) and p.ref:
                    refs.append(p.ref)
        return refs
