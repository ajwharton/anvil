"""HFVLMRenderer tests — fake processor (no multi‑GB VLM download)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("PIL")

from anvil.media import LocalMediaStore
from anvil.protocol.messages import Example, ImagePart, Message, TextPart
from anvil.render.hf import RendererConsistencyError
from anvil.render.vlm import HFVLMRenderer


def _png_bytes(w: int = 8, h: int = 8, rgb: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    """Minimal valid RGB PNG (no external deps beyond stdlib)."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), rgb).save(buf, format="PNG")
    return buf.getvalue()


class _FakeTok:
    def __init__(self) -> None:
        self.chat_template = "present"
        self._vocab: dict[str, int] = {"<img>": 1, "<|im_end|>": 2, "a": 10}
        self._next = 20

    def _id(self, piece: str) -> int:
        if piece not in self._vocab:
            self._vocab[piece] = self._next
            self._next += 1
        return self._vocab[piece]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Space-split + keep known specials
        out: list[int] = []
        for tok in text.replace("<img>", " <img> ").replace("<|im_end|>", " <|im_end|> ").split():
            if not tok:
                continue
            out.append(self._id(tok))
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, f"<{i}>") for i in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
    ) -> Any:
        parts: list[str] = []
        for m in messages:
            parts.append(f"<|{m['role']}|>")
            content = m["content"]
            if isinstance(content, str):
                parts.append(content)
            else:
                for c in content:
                    if c.get("type") == "image":
                        parts.append("<img>")
                    else:
                        parts.append(str(c.get("text", "")))
            parts.append("<|im_end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        text = " ".join(parts)
        if tokenize:
            return self.encode(text)
        return text


class _FakeProcessor:
    """Minimal processor: apply_chat_template + __call__(text, images)."""

    def __init__(self) -> None:
        self.tokenizer = _FakeTok()
        self.chat_template = self.tokenizer.chat_template

    def apply_chat_template(self, *a: Any, **k: Any) -> Any:
        return self.tokenizer.apply_chat_template(*a, **k)

    def __call__(
        self,
        text: str | None = None,
        images: Any = None,
        return_tensors: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Image count must match <img> placeholders for a realistic path
        n_img = 0
        if images is not None:
            n_img = 1 if not isinstance(images, list) else len(images)
        assert text is not None
        n_ph = text.count("<img>")
        if n_img and n_ph != n_img:
            # still encode — real processors are strict; we only check shape
            pass
        ids = self.tokenizer.encode(text)
        return {"input_ids": [ids]}


@pytest.fixture()
def store_with_image(tmp_path):
    store = LocalMediaStore(tmp_path / "cas")
    ref = store.put(_png_bytes(), suffix=".png")
    return store, ref


def test_vlm_sft_datum_includes_image_refs(store_with_image):
    store, ref = store_with_image
    r = HFVLMRenderer(_FakeProcessor(), store)
    ex = Example(
        messages=(
            Message(
                role="user",
                content=(TextPart(text="grasp?"), ImagePart(ref=ref)),
            ),
            Message(role="assistant", content="yes"),
        )
    )
    datum = r.render_example_for_sft(ex)
    ids = datum.model_input.token_ids()
    targets = datum.loss_fn_inputs["target_tokens"]
    weights = datum.loss_fn_inputs["weights"]
    assert len(ids) == len(targets) == len(weights)
    assert datum.loss_fn_inputs["image_refs"] == [ref]
    assert sum(weights) > 0  # some assistant tokens trained


def test_vlm_prompt_is_prefix_of_full_sft(store_with_image):
    store, ref = store_with_image
    r = HFVLMRenderer(_FakeProcessor(), store)
    messages = (
        Message(
            role="user",
            content=(TextPart(text="what color?"), ImagePart(ref=ref)),
        ),
        Message(role="assistant", content="blue"),
    )
    full = r.render_messages(messages).token_ids()
    prompt = r.render_prompt(messages[:-1]).token_ids()
    assert full[: len(prompt)] == prompt


def test_missing_cas_ref_raises(tmp_path):
    store = LocalMediaStore(tmp_path / "cas")
    r = HFVLMRenderer(_FakeProcessor(), store)
    ex = Example(
        messages=(
            Message(
                role="user",
                content=(
                    TextPart(text="x"),
                    ImagePart(ref="cas://sha256/" + "d" * 64 + ".png"),
                ),
            ),
            Message(role="assistant", content="y"),
        )
    )
    with pytest.raises(FileNotFoundError):
        r.render_example_for_sft(ex)


def test_text_only_still_works(store_with_image):
    store, _ = store_with_image
    r = HFVLMRenderer(_FakeProcessor(), store)
    ex = Example(
        messages=(
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        )
    )
    d = r.render_example_for_sft(ex)
    assert len(d.model_input.token_ids()) >= 1
    assert d.loss_fn_inputs.get("image_refs") == []


def test_broken_prefix_raises(store_with_image, monkeypatch):
    store, ref = store_with_image
    r = HFVLMRenderer(_FakeProcessor(), store)

    # Force second tokenize call to return a divergent sequence
    calls = {"n": 0}
    orig = r._tokenize

    def flaky(messages, add_generation_prompt=False):
        ids, imgs = orig(messages, add_generation_prompt=add_generation_prompt)
        calls["n"] += 1
        # Corrupt the sample-prompt render so it is not a prefix of the full SFT ids
        if add_generation_prompt and calls["n"] <= 3:
            return [99999] + ids, imgs
        return ids, imgs

    monkeypatch.setattr(r, "_tokenize", flaky)
    ex = Example(
        messages=(
            Message(
                role="user",
                content=(TextPart(text="q"), ImagePart(ref=ref)),
            ),
            Message(role="assistant", content="a"),
        )
    )
    with pytest.raises(RendererConsistencyError):
        r.render_example_for_sft(ex)
