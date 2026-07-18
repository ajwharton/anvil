"""P3.2: LocalBackend image modality + freeze policy (CPU tiny LM smoke)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from anvil.backends.local import LocalBackend  # noqa: E402
from anvil.protocol.messages import Example, ImagePart, Message, TextPart  # noqa: E402
from anvil.protocol.types import AdamParams, LoraConfig, LoraTargets, TrainConfig  # noqa: E402
from anvil.render.text import ToyTextRenderer  # noqa: E402

TINY = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture()
def backend(tmp_path) -> LocalBackend:
    return LocalBackend(
        device="cpu",
        root=tmp_path / "local",
        target_modules=["c_attn", "c_proj", "c_fc"],
        allow_tiny_models=True,
    )


def test_image_modality_session_allowed(backend: LocalBackend):
    """Phase 3.2: image modality no longer raises NotImplementedError."""
    cfg = TrainConfig(
        base_model=TINY,
        lora=LoraConfig(
            rank=4,
            targets=LoraTargets(language=True, vision_encoder=False, mm_projector=True),
        ),
        modalities=("text", "image"),
    )
    aid = backend.create_lora_session(cfg)
    assert str(aid).startswith("adapter-")
    sess = backend._get(aid)
    assert "image" in sess.config.modalities


def test_ce_with_image_refs_metric(backend: LocalBackend):
    cfg = TrainConfig(
        base_model=TINY,
        lora=LoraConfig(rank=4),
        modalities=("text", "image"),
    )
    aid = backend.create_lora_session(cfg)
    r = ToyTextRenderer()
    ex = Example(
        messages=(
            Message(
                role="user",
                content=(
                    TextPart(text="grasp?"),
                    ImagePart(ref="cas://sha256/" + "a" * 64),
                ),
            ),
            Message(role="assistant", content="yes"),
        )
    )
    # Toy renderer embeds ref as text; stamp image_refs like HFVLMRenderer
    datum = r.render_example_for_sft(ex)
    datum.loss_fn_inputs["image_refs"] = [ex.image_refs()[0]]
    out = backend.forward_backward(aid, [datum], "cross_entropy")
    assert out.metrics.get("n_image_refs") == 1.0
    backend.optim_step(aid, AdamParams(learning_rate=1e-3))


def test_resolve_target_modules_language_default(tmp_path):
    # Fresh backend without forced target_modules (fixture forces c_attn for gpt2)
    bare = LocalBackend(device="cpu", root=tmp_path / "bare", allow_tiny_models=True)
    # language-only → peft architecture defaults
    assert (
        bare._resolve_target_modules(
            LoraTargets(language=True, vision_encoder=False, mm_projector=False)
        )
        is None
    )
    # + projector → explicit name list
    mods = bare._resolve_target_modules(
        LoraTargets(language=True, vision_encoder=False, mm_projector=True)
    )
    assert mods is not None
    assert "q_proj" in mods or "c_attn" in mods
    assert "merger" in mods or "mm_projector" in mods
