"""Hugging Face chat-template renderer — real tokenizer, train/sample shared.

Phase 1 renderer. One renderer configuration must be used by both the train
worker and the sample worker: that is the train/sample consistency invariant
(docs/design.md §3). ``ToyTextRenderer`` proves control flow against the fake
backend; this renderer produces the tokens a real model actually sees.

Text-only. For images use :class:`~anvil.render.vlm.HFVLMRenderer` (Phase 3.1)
with a media store and multimodal processor.

Optional dependency: ``pip install anvil-train[hf]`` (transformers).
"""

from __future__ import annotations

from typing import Any, Sequence

from anvil.protocol.messages import Example, ImagePart, ImageUrlPart, Message
from anvil.protocol.types import Datum, ModelInput


class RendererConsistencyError(RuntimeError):
    """Chat template violated the train/sample prefix invariant.

    Raised when an incrementally-rendered prompt prefix is not a token-exact
    prefix of the full conversation render. Sampling with such a template
    would show the model different tokens than training did.
    """


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as e:  # pragma: no cover - depends on env
        raise ImportError(
            "HFChatRenderer requires transformers; "
            "install with: pip install anvil-train[hf]"
        ) from e
    version = tuple(int(p) for p in transformers.__version__.split(".")[:2])
    if version < (4, 44):
        raise ImportError(
            f"HFChatRenderer requires transformers>=4.44 "
            f"(found {transformers.__version__}); assistant-mask and "
            f"chat-template APIs differ on older versions"
        )
    try:
        import jinja2  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on env
        # transformers v5 does not declare jinja2 but apply_chat_template
        # needs it — fail here with an actionable message, not deep inside
        # transformers with a bare ImportError.
        raise ImportError(
            "apply_chat_template requires jinja2, which transformers does not "
            "declare; install with: pip install anvil-train[hf]"
        ) from e
    return transformers


class HFChatRenderer:
    """Render Example/Message batches through a model's real chat template.

    Parameters
    ----------
    tokenizer_or_model:
        HF repo id or local path for the tokenizer (usually the base model id).
    chat_template:
        Optional Jinja template override. Base models without an instruct
        template need one (e.g. a ChatML snippet) — training with template A
        and sampling with template B silently corrupts the loss.
    train_on_assistant_only:
        When True (default), CE weight is 1 only on assistant spans
        (content + end-of-turn), 0 on system/user/tool spans.
    """

    name = "hf_chat"

    def __init__(
        self,
        tokenizer_or_model: str,
        *,
        chat_template: str | None = None,
        train_on_assistant_only: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        transformers = _require_transformers()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_or_model, trust_remote_code=trust_remote_code
        )
        if chat_template is not None:
            self.tokenizer.chat_template = chat_template
        if self.tokenizer.chat_template is None:
            raise ValueError(
                f"{tokenizer_or_model!r} ships no chat template; pass "
                f"chat_template= (e.g. ChatML) so train and sample agree"
            )
        self.train_on_assistant_only = train_on_assistant_only

    # --- raw token helpers ---------------------------------------------------

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=False)

    # --- message rendering ---------------------------------------------------

    def _to_hf_messages(self, messages: Sequence[Message]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for m in messages:
            text_parts: list[str] = []
            for part in m.parts():
                if isinstance(part, (ImagePart, ImageUrlPart)):
                    raise NotImplementedError(
                        "HFChatRenderer is text-only; use HFVLMRenderer "
                        "(anvil.render.vlm) with a MediaStore for image parts"
                    )
                text_parts.append(part.text)
            out.append({"role": m.role, "content": "".join(text_parts)})
        return out

    def _render_ids(
        self, hf_messages: list[dict[str, str]], *, add_generation_prompt: bool = False
    ) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            hf_messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,  # plain list[int] (transformers v5 defaults to dict)
        )
        return [int(t) for t in ids]

    def render_messages(
        self, messages: Sequence[Message], *, add_generation_prompt: bool = False
    ) -> ModelInput:
        """Render a conversation (or sample-side prompt) to token ids."""
        return ModelInput.from_ints(
            self._render_ids(
                self._to_hf_messages(messages),
                add_generation_prompt=add_generation_prompt,
            )
        )

    def render_prompt(self, messages: Sequence[Message]) -> ModelInput:
        """Sample-side prompt: conversation + generation prompt."""
        return self.render_messages(messages, add_generation_prompt=True)

    # --- SFT datum -----------------------------------------------------------

    def render_example_for_sft(self, example: Example) -> Datum:
        """Build Datum with CE weights: 0 on non-assistant, 1 on assistant.

        Causal LM shift: ``model_input = ids[:-1]`` predicts ``ids[1:]``.
        Assistant spans come from the template's ``{% generation %}`` blocks
        when present, else from incremental prefix diffs (with a strict
        prefix-consistency check against the full render).
        """
        hf_msgs = self._to_hf_messages(example.messages)
        if not hf_msgs:
            raise ValueError("example has no messages")

        ids = self._render_ids(hf_msgs)
        if len(ids) < 2:
            raise ValueError("example too short to form CE targets")

        if not self.train_on_assistant_only:
            mask = [1.0] * len(ids)
        elif "{% generation %}" in (self.tokenizer.chat_template or ""):
            mask = self._native_assistant_mask(hf_msgs, ids)
        else:
            mask = self._incremental_assistant_mask(hf_msgs, ids)

        return Datum(
            model_input=ModelInput.from_ints(ids[:-1]),
            loss_fn_inputs={
                "target_tokens": ids[1:],
                "weights": mask[1:],
            },
        )

    def _native_assistant_mask(
        self, hf_msgs: list[dict[str, str]], ids: list[int]
    ) -> list[float]:
        enc = self.tokenizer.apply_chat_template(
            hf_msgs,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        raw = enc["assistant_masks"]
        mask = [float(v) for v in (raw[0] if raw and isinstance(raw[0], list) else raw)]
        if len(mask) != len(ids):  # pragma: no cover - defensive
            raise RendererConsistencyError(
                f"assistant mask length {len(mask)} != token length {len(ids)}"
            )
        return mask

    def _incremental_assistant_mask(
        self, hf_msgs: list[dict[str, str]], ids: list[int]
    ) -> list[float]:
        mask = [0.0] * len(ids)
        for i, msg in enumerate(hf_msgs):
            if msg["role"] != "assistant":
                continue
            prefix = self._render_ids(hf_msgs[:i], add_generation_prompt=True)
            through_i = self._render_ids(hf_msgs[: i + 1])
            self._assert_prefix(prefix, ids, ctx=f"prompt before message {i}")
            self._assert_prefix(through_i, ids, ctx=f"render through message {i}")
            if len(through_i) <= len(prefix):
                raise RendererConsistencyError(
                    f"assistant message {i} produced no tokens beyond the "
                    f"generation prompt — empty assistant turn?"
                )
            for j in range(len(prefix), len(through_i)):
                mask[j] = 1.0
        return mask

    @staticmethod
    def _assert_prefix(prefix: list[int], full: list[int], *, ctx: str) -> None:
        if full[: len(prefix)] != prefix:
            raise RendererConsistencyError(
                f"{ctx}: incremental render is not a token-exact prefix of the "
                f"full render; this chat template cannot guarantee train/sample "
                f"consistency (BPE merge across the boundary or non-prefix "
                f"template). Choose another template or fix the boundary."
            )
