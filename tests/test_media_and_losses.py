"""Media store + loss registry unit tests."""

from __future__ import annotations

import pytest

from anvil.losses import get_loss, list_losses
from anvil.media import LocalMediaStore
from anvil.protocol import ImagePart, ImageRefChunk, Message, ModelInput, TextPart
from anvil.protocol.messages import Example
from anvil.render import ToyTextRenderer


def test_local_media_store_roundtrip(tmp_path):
    store = LocalMediaStore(tmp_path / "cas")
    ref = store.put(b"\x89PNG-fake", suffix=".png")
    assert ref.startswith("cas://sha256/")
    assert store.exists(ref)
    assert store.get(ref) == b"\x89PNG-fake"
    # idempotent put
    ref2 = store.put(b"\x89PNG-fake", suffix=".png")
    assert ref2 == ref


def test_loss_registry_builtins():
    names = set(list_losses())
    assert "cross_entropy" in names
    assert "importance_sampling" in names
    assert "ppo" in names
    spec = get_loss("cross_entropy")
    assert "target_tokens" in spec.required_inputs
    with pytest.raises(KeyError):
        get_loss("not_a_real_loss")


def test_vision_message_renders_image_placeholder():
    ex = Example(
        messages=(
            Message(
                role="user",
                content=(
                    TextPart(text="grasp?"),
                    ImagePart(ref="cas://sha256/abc", detail="high"),
                ),
            ),
            Message(role="assistant", content="yes"),
        )
    )
    mi = ToyTextRenderer().render_messages(ex.messages)
    # Image ref appears as placeholder text tokens in toy renderer
    decoded_ish = ToyTextRenderer().decode(mi.token_ids())
    assert "image:cas://sha256/abc" in decoded_ish or "cas://sha256/abc" in decoded_ish


def test_model_input_chunks_mixed():
    mi = ModelInput.from_chunks(
        [
            ImageRefChunk(ref="cas://sha256/deadbeef"),
        ]
    )
    assert mi.token_ids() == []  # image-only: no text tokens until real renderer
