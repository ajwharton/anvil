"""Minimal text renderer for Phase 0 tests (no real tokenizer).

Uses a trivial whitespace / char-code encoding so golden tests need no HF deps.
Replace with model-specific chat templates in Phase 1.
"""

from __future__ import annotations

from typing import Sequence

from anvil.protocol.messages import Example, ImagePart, Message, TextPart
from anvil.protocol.types import Datum, EncodedTextChunk, ModelInput


class ToyTextRenderer:
    """Deterministic toy tokenizer: each char → codepoint int (clipped)."""

    name = "toy_text"

    def __init__(self, *, train_on_assistant_only: bool = True) -> None:
        self.train_on_assistant_only = train_on_assistant_only

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), 0x10FFFF) for c in text]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(chr(t) if t < 0x110000 else "?" for t in tokens)

    def render_messages(self, messages: Sequence[Message]) -> ModelInput:
        pieces: list[str] = []
        for m in messages:
            pieces.append(f"<|{m.role}|>\n")
            for part in m.parts():
                if isinstance(part, TextPart):
                    pieces.append(part.text)
                elif isinstance(part, ImagePart):
                    # Placeholder — real VLM renderer expands refs to image tokens
                    pieces.append(f"<|image:{part.ref}|>")
            pieces.append("\n")
        text = "".join(pieces)
        return ModelInput.from_ints(self.encode(text))

    def render_example_for_sft(self, example: Example) -> Datum:
        """Build Datum with CE weights: 0 on prompt/user, 1 on assistant tokens."""
        all_tokens: list[int] = []
        weights: list[float] = []

        for m in example.messages:
            prefix = self.encode(f"<|{m.role}|>\n")
            body_parts: list[int] = []
            for part in m.parts():
                if isinstance(part, TextPart):
                    body_parts.extend(self.encode(part.text))
                elif isinstance(part, ImagePart):
                    body_parts.extend(self.encode(f"<|image:{part.ref}|>"))
            suffix = self.encode("\n")
            chunk = prefix + body_parts + suffix
            w = 1.0 if (m.role == "assistant" and self.train_on_assistant_only) else 0.0
            if not self.train_on_assistant_only:
                w = 1.0
            all_tokens.extend(chunk)
            weights.extend([w] * len(chunk))

        if len(all_tokens) < 2:
            raise ValueError("example too short to form CE targets")

        # Standard causal LM: input[:-1] predicts target[1:]
        model_input = ModelInput(chunks=(EncodedTextChunk(tokens=tuple(all_tokens[:-1])),))
        target_tokens = all_tokens[1:]
        target_weights = weights[1:]
        return Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "weights": target_weights,
            },
        )
