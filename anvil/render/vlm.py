"""Processor-backed VLM renderer — media refs → image tokens (Phase 3.1).

Uses a Hugging Face multimodal ``Processor`` (e.g. Qwen2.5-VL
``AutoProcessor``) so train and sample expand the same image placeholders.
Images are loaded from a :class:`~anvil.media.store.MediaStore` via
``cas://`` refs (or ``file://`` / local paths for smoke ingest).

Optional deps: ``transformers``, ``Pillow`` (``pip install anvil-train[hf]``
plus pillow). The full VLM weights are only needed when constructing a real
processor — unit tests inject a fake processor.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Sequence

from anvil.media.store import MediaStore
from anvil.protocol.messages import Example, ImagePart, ImageUrlPart, Message, TextPart
from anvil.protocol.types import Datum, ModelInput
from anvil.render.hf import RendererConsistencyError, _require_transformers


def _require_pil() -> Any:
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "HFVLMRenderer requires Pillow; install with: pip install pillow"
        ) from e
    return Image


class HFVLMRenderer:
    """Render multimodal Examples through a HF vision-language processor.

    Parameters
    ----------
    processor_or_model:
        HF repo id / local path for ``AutoProcessor``, or an already-loaded
        processor (tests inject fakes).
    media_store:
        Resolves ``ImagePart.ref`` (``cas://…``) to image bytes.
    train_on_assistant_only:
        CE weights only on assistant spans when the template supports it.
    trust_remote_code:
        Forwarded to ``AutoProcessor.from_pretrained``.
    """

    name = "hf_vlm"

    def __init__(
        self,
        processor_or_model: str | Any,
        media_store: MediaStore,
        *,
        train_on_assistant_only: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        self.media_store = media_store
        self.train_on_assistant_only = train_on_assistant_only
        if isinstance(processor_or_model, str):
            transformers = _require_transformers()
            self.processor = transformers.AutoProcessor.from_pretrained(
                processor_or_model, trust_remote_code=trust_remote_code
            )
        else:
            self.processor = processor_or_model
        tok = getattr(self.processor, "tokenizer", None) or self.processor
        self.tokenizer = tok
        if getattr(self.tokenizer, "chat_template", None) is None and not hasattr(
            self.processor, "apply_chat_template"
        ):
            raise ValueError(
                "VLM processor/tokenizer has no chat template; train/sample "
                "consistency requires a shared template"
            )

    # --- image loading -------------------------------------------------------

    def load_image(self, part: ImagePart | ImageUrlPart) -> Any:
        """Return a PIL Image for a message part."""
        Image = _require_pil()
        if isinstance(part, ImagePart):
            ref = part.ref
            if ref.startswith("cas://"):
                data = self.media_store.get(ref)
                return Image.open(io.BytesIO(data)).convert("RGB")
            if ref.startswith("file://"):
                return Image.open(ref[7:]).convert("RGB")
            path = Path(ref)
            if path.is_file():
                return Image.open(path).convert("RGB")
            raise FileNotFoundError(f"image ref not found: {ref!r}")
        # ImageUrlPart
        url = part.url
        if url.startswith("file://"):
            return Image.open(url[7:]).convert("RGB")
        path = Path(url)
        if path.is_file():
            return Image.open(path).convert("RGB")
        raise ValueError(
            f"unsupported image_url {url!r}; use cas:// refs or local file paths"
        )

    def collect_images(self, messages: Sequence[Message]) -> list[Any]:
        """Ordered PIL images in conversation order (user/assistant content)."""
        images: list[Any] = []
        for m in messages:
            for part in m.parts():
                if isinstance(part, (ImagePart, ImageUrlPart)):
                    images.append(self.load_image(part))
        return images

    # --- HF message conversion -----------------------------------------------

    def _to_hf_messages(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """HF-style messages: content is str or list of {type, text|image}."""
        out: list[dict[str, Any]] = []
        for m in messages:
            parts = m.parts()
            has_image = any(isinstance(p, (ImagePart, ImageUrlPart)) for p in parts)
            if not has_image and len(parts) == 1 and isinstance(parts[0], TextPart):
                out.append({"role": m.role, "content": parts[0].text})
                continue
            content: list[dict[str, Any]] = []
            for part in parts:
                if isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, (ImagePart, ImageUrlPart)):
                    # Processor inserts image tokens; actual pixels passed separately.
                    content.append({"type": "image"})
                else:  # pragma: no cover
                    raise TypeError(f"unsupported part: {type(part)}")
            out.append({"role": m.role, "content": content})
        return out

    def _apply_template(
        self,
        hf_messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
    ) -> Any:
        apply = getattr(self.processor, "apply_chat_template", None)
        if apply is None:
            apply = self.tokenizer.apply_chat_template
        return apply(
            hf_messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=False if tokenize else False,
        )

    def _tokenize(
        self,
        messages: Sequence[Message],
        *,
        add_generation_prompt: bool = False,
    ) -> tuple[list[int], list[Any]]:
        """Return (input_ids, images) for a conversation."""
        hf_msgs = self._to_hf_messages(messages)
        images = self.collect_images(messages)
        # Prefer processor(text=template, images=...) when images present.
        text = self._apply_template(
            hf_msgs, add_generation_prompt=add_generation_prompt, tokenize=False
        )
        if images:
            enc = self.processor(
                text=text if isinstance(text, str) else text,
                images=images if len(images) != 1 else images[0],
                return_tensors=None,
            )
            ids = enc["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(t) for t in ids], images
        # Text-only path (still allowed on a VLM processor)
        if isinstance(text, str):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            return [int(t) for t in ids], []
        return [int(t) for t in text], []

    # public API matching HFChatRenderer --------------------------------------

    def encode(self, text: str) -> list[int]:
        return [int(t) for t in self.tokenizer.encode(text, add_special_tokens=False)]

    def decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=False)

    def render_messages(
        self, messages: Sequence[Message], *, add_generation_prompt: bool = False
    ) -> ModelInput:
        ids, _ = self._tokenize(messages, add_generation_prompt=add_generation_prompt)
        return ModelInput.from_ints(ids)

    def render_prompt(self, messages: Sequence[Message]) -> ModelInput:
        return self.render_messages(messages, add_generation_prompt=True)

    def render_example_for_sft(self, example: Example) -> Datum:
        """Build CE Datum; records image_refs for the train backend (P3.2)."""
        if not example.messages:
            raise ValueError("example has no messages")
        ids, _images = self._tokenize(example.messages, add_generation_prompt=False)
        if len(ids) < 2:
            raise ValueError("example too short to form CE targets")

        # Sample-side prompt must be a prefix of the full train render.
        # Build user-only prefix through last user turn for the check when
        # the final message is assistant (standard SFT).
        prompt_msgs = list(example.messages)
        if prompt_msgs and prompt_msgs[-1].role == "assistant":
            prompt_msgs = prompt_msgs[:-1]
        if prompt_msgs:
            prompt_ids, _ = self._tokenize(
                prompt_msgs, add_generation_prompt=True
            )
            self._assert_prefix(prompt_ids, ids, ctx="vlm sample prompt vs full SFT render")

        if not self.train_on_assistant_only:
            mask = [1.0] * len(ids)
        else:
            mask = self._assistant_mask(example.messages, ids)

        refs = example.image_refs()
        return Datum(
            model_input=ModelInput.from_ints(ids[:-1]),
            loss_fn_inputs={
                "target_tokens": ids[1:],
                "weights": mask[1:],
                "image_refs": list(refs),
            },
        )

    def _assistant_mask(
        self, messages: Sequence[Message], ids: list[int]
    ) -> list[float]:
        """Mark assistant token spans via incremental prefix renders."""
        hf_all = self._to_hf_messages(messages)
        mask = [0.0] * len(ids)
        # Walk message boundaries using tokenized prefixes of the same images.
        for i, msg in enumerate(messages):
            if msg.role != "assistant":
                continue
            prefix_msgs = list(messages[:i])
            through = list(messages[: i + 1])
            prefix_ids, _ = self._tokenize(prefix_msgs, add_generation_prompt=True)
            through_ids, _ = self._tokenize(through, add_generation_prompt=False)
            self._assert_prefix(prefix_ids, ids, ctx=f"vlm prompt before message {i}")
            self._assert_prefix(through_ids, ids, ctx=f"vlm render through message {i}")
            if len(through_ids) <= len(prefix_ids):
                raise RendererConsistencyError(
                    f"assistant message {i} produced no tokens beyond the generation prompt"
                )
            for j in range(len(prefix_ids), len(through_ids)):
                mask[j] = 1.0
        # Silence unused when empty assistants
        _ = hf_all
        return mask

    @staticmethod
    def _assert_prefix(prefix: list[int], full: list[int], *, ctx: str) -> None:
        if full[: len(prefix)] != prefix:
            raise RendererConsistencyError(
                f"{ctx}: incremental VLM render is not a token-exact prefix of "
                f"the full render; train/sample would diverge"
            )
