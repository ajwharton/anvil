"""HFChatRenderer tests — real tokenizer + the train/sample invariant.

Uses sshleifer/tiny-gpt2 (tiny download, no chat template) with ChatML-style
template overrides: one with {% generation %} markers (native mask path),
one without (incremental prefix-diff path). The two paths must agree.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")  # noqa: F401

from anvil.protocol.messages import Example, ImagePart, Message, TextPart
from anvil.render.hf import HFChatRenderer, RendererConsistencyError

TINY = "sshleifer/tiny-gpt2"

# Same rendered surface; GEN adds {% generation %} around assistant content.
CHATML_NOGEN = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
CHATML_GEN = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] + '<|im_end|>\n' }}{% endgeneration %}"
    "{% else %}{{ message['content'] + '<|im_end|>\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _example() -> Example:
    return Example(
        messages=(
            Message(role="user", content=(TextPart(text="2+2?"),)),
            Message(role="assistant", content=(TextPart(text="4"),)),
            Message(role="user", content=(TextPart(text="and 3+3?"),)),
            Message(role="assistant", content=(TextPart(text="6"),)),
        )
    )


@pytest.fixture(scope="module")
def renderer() -> HFChatRenderer:
    return HFChatRenderer(TINY, chat_template=CHATML_NOGEN)


def test_encode_decode_roundtrip(renderer: HFChatRenderer) -> None:
    text = "hello anvil <|im_start|>"
    assert renderer.decode(renderer.encode(text)) == text


def test_missing_chat_template_raises() -> None:
    with pytest.raises(ValueError, match="chat template"):
        HFChatRenderer(TINY)


def test_image_part_raises(renderer: HFChatRenderer) -> None:
    ex = Example(
        messages=(
            Message(
                role="user",
                content=(TextPart(text="what is this?"), ImagePart(ref="cas://abc")),
            ),
            Message(role="assistant", content=(TextPart(text="a grasp"),)),
        )
    )
    with pytest.raises(NotImplementedError, match="Phase 3"):
        renderer.render_example_for_sft(ex)


def test_sft_datum_shapes_and_masking(renderer: HFChatRenderer) -> None:
    datum = renderer.render_example_for_sft(_example())
    ids = datum.model_input.token_ids()
    targets = datum.loss_fn_inputs["target_tokens"]
    weights = datum.loss_fn_inputs["weights"]

    # Causal shift: input[:-1] → target[1:], weights aligned to targets.
    assert len(ids) == len(targets) == len(weights)
    assert set(weights) <= {0.0, 1.0}
    assert 0.0 in weights and 1.0 in weights

    # Assistant turns carry weight; user turns do not.
    assistant_tokens = [t for t, w in zip(targets, weights, strict=True) if w == 1.0]
    user_tokens = [t for t, w in zip(targets, weights, strict=True) if w == 0.0]
    assert renderer.decode(assistant_tokens).find("4") != -1
    assert renderer.decode(assistant_tokens).find("6") != -1
    assert renderer.decode(user_tokens).find("2+2?") != -1


def test_train_sample_prefix_consistency(renderer: HFChatRenderer) -> None:
    """The product thesis as a test: sample-side prompt tokens must be an
    exact prefix of the tokens the trainer saw."""
    convo = _example()
    for cut in (1, 3):  # prompts ending before each assistant turn
        prompt_ids = renderer.render_prompt(convo.messages[:cut]).token_ids()
        train_ids = renderer.render_example_for_sft(convo).model_input.token_ids()
        assert train_ids[: len(prompt_ids)] == prompt_ids


def test_native_and_incremental_masks_agree() -> None:
    ex = _example()
    gen = HFChatRenderer(TINY, chat_template=CHATML_GEN)
    nogen = HFChatRenderer(TINY, chat_template=CHATML_NOGEN)
    d_gen = gen.render_example_for_sft(ex)
    d_nogen = nogen.render_example_for_sft(ex)
    assert d_gen.model_input.token_ids() == d_nogen.model_input.token_ids()
    assert d_gen.loss_fn_inputs["weights"] == d_nogen.loss_fn_inputs["weights"]


def test_broken_template_raises_consistency_error() -> None:
    # A template that appends a trailing token after the final message breaks
    # the prefix invariant for incremental masking → must raise, not silently
    # train on one surface and sample on another.
    sneaky = CHATML_NOGEN.replace(
        "{% endfor %}", "{% endfor %}{{ '<|trailing|>' }}"
    )
    r = HFChatRenderer(TINY, chat_template=sneaky)
    with pytest.raises(RendererConsistencyError):
        r.render_example_for_sft(_example())
