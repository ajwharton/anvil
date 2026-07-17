"""LocalBackend golden tests — real torch + PEFT behind the four verbs.

Runs on CPU against hf-internal-testing/tiny-random-gpt2 (5 layers, hidden 32,
vocab 1000 — big enough for LoRA to actually learn, tiny enough for CI).
The SFT test is the Phase 1 gate in miniature: the verbs must actually train,
not just book-keep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # noqa: F401
peft = pytest.importorskip("peft")  # noqa: F401
transformers = pytest.importorskip("transformers")  # noqa: F401

from anvil.backends.fake import FakeBackend
from anvil.backends.local import LocalBackend, ModelTooSmallError
from anvil.client.service import ServiceClient
from anvil.protocol.messages import Example, Message, TextPart
from anvil.protocol.types import (
    AdamParams,
    ExportFormat,
    LossFn,
    ModelInput,
    SamplingParams,
)
from anvil.render.text import ToyTextRenderer

TINY = "hf-internal-testing/tiny-random-gpt2"
SFT_STEPS = 30


@pytest.fixture(scope="module")
def backend(tmp_path_factory) -> LocalBackend:
    torch.manual_seed(0)  # tiny-random-gpt2 has dropout; keep runs reproducible
    return LocalBackend(
        device="cpu",
        root=tmp_path_factory.mktemp("anvil-local"),
        target_modules=["c_attn", "c_proj", "c_fc"],
    )


@pytest.fixture(scope="module")
def adapter_id(backend: LocalBackend):
    from anvil.protocol.types import LoraConfig, TrainConfig

    cfg = TrainConfig(base_model=TINY, lora=LoraConfig(rank=8))
    return backend.create_lora_session(cfg)


def _sft_data() -> list:
    r = ToyTextRenderer()
    examples = [
        Example(
            messages=(
                Message(role="user", content=(TextPart(text="2+2?"),)),
                Message(role="assistant", content=(TextPart(text="4"),)),
            )
        ),
        Example(
            messages=(
                Message(role="user", content=(TextPart(text="3+3?"),)),
                Message(role="assistant", content=(TextPart(text="6"),)),
            )
        ),
    ]
    return [r.render_example_for_sft(ex) for ex in examples]


def test_endpoint_wiring() -> None:
    assert isinstance(ServiceClient(endpoint="local://").backend, LocalBackend)
    assert isinstance(ServiceClient(endpoint="local://fake").backend, FakeBackend)
    assert isinstance(ServiceClient(endpoint="fake://").backend, FakeBackend)


def test_sft_loss_decreases(backend: LocalBackend, adapter_id) -> None:
    """The Phase 1 gate: forward_backward + optim_step actually learn."""
    torch.manual_seed(0)
    data = _sft_data()
    losses = []
    for _ in range(SFT_STEPS):
        fb = backend.forward_backward(adapter_id, data, LossFn.CROSS_ENTROPY)
        backend.optim_step(adapter_id, AdamParams(learning_rate=1e-2))
        losses.append(fb.loss)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert losses[-1] < 0.95 * losses[0], f"weak learning signal: {losses}"


def test_save_and_load_state_roundtrip(backend: LocalBackend, adapter_id) -> None:
    ref = backend.save_state(adapter_id, "after-sft")
    assert ref.path.endswith("after-sft")
    meta = json.loads((Path(ref.path) / "anvil_state.json").read_text())
    assert meta["step"] == SFT_STEPS  # from the SFT test above (module-scoped)

    backend.optim_step(adapter_id, AdamParams())  # step SFT_STEPS + 1
    backend.load_state(adapter_id, ref)
    sess = backend._get(adapter_id)
    assert sess.step == SFT_STEPS


def test_snapshot_and_sample(backend: LocalBackend, adapter_id) -> None:
    ref = backend.snapshot_for_sample(adapter_id, "s1")
    assert ref.kind == "sampler"

    prompt = ModelInput.from_ints(ToyTextRenderer().encode("<|user|>\n2+2?\n"))
    result = backend.sample(
        base_model=TINY,
        adapter_id=adapter_id,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=8, temperature=1.0, seed=1234),
        num_samples=2,
        include_prompt_logprobs=True,
    )
    assert len(result.sequences) == 2
    for seq in result.sequences:
        assert len(seq.tokens) == 8
        assert seq.logprobs is not None
        assert len(seq.logprobs) == len(seq.tokens)
    assert result.prompt_logprobs is not None
    assert result.prompt_logprobs[0] is None
    assert len(result.prompt_logprobs) == len(prompt.token_ids())

    greedy = backend.sample(
        base_model=TINY,
        adapter_id=adapter_id,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
    )
    assert len(greedy.sequences) == 1
    with pytest.raises(ValueError, match="greedy"):
        backend.sample(
            base_model=TINY,
            adapter_id=adapter_id,
            prompt=prompt,
            sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
            num_samples=2,
        )


def test_compute_logprobs(backend: LocalBackend, adapter_id) -> None:
    prompt = ModelInput.from_ints(ToyTextRenderer().encode("hello"))
    lps = backend.compute_logprobs(base_model=TINY, adapter_id=adapter_id, prompt=prompt)
    assert lps[0] is None
    assert all(isinstance(x, float) for x in lps[1:])


def test_export_peft_real_dir(backend: LocalBackend, adapter_id, tmp_path) -> None:
    res = backend.export_adapter(adapter_id, ExportFormat.PEFT, str(tmp_path / "peft-out"))
    assert res.format == ExportFormat.PEFT
    cfg = json.loads((tmp_path / "peft-out" / "adapter_config.json").read_text())
    assert cfg["r"] == 8
    assert cfg["base_model_name_or_path"] == TINY
    assert (tmp_path / "peft-out" / "adapter_model.safetensors").is_file()


def test_rl_losses_raise_for_now(backend: LocalBackend, adapter_id) -> None:
    with pytest.raises(NotImplementedError, match="Phase 2"):
        backend.forward_backward(adapter_id, _sft_data(), LossFn.PPO)


def test_stop_strings_truncate_greedy(backend: LocalBackend, adapter_id) -> None:
    """Greedy is deterministic: learn what it says, then stop inside it."""
    tok = backend._get(adapter_id).tokenizer
    prompt = ModelInput.from_ints(ToyTextRenderer().encode("<|user|>\n2+2?\n"))
    full = backend.sample(
        base_model=TINY,
        adapter_id=adapter_id,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
    )
    toks = list(full.sequences[0].tokens)
    if len(toks) < 6:
        pytest.skip(f"greedy output too short to sub-slice: {toks}")
    stop_str = tok.decode(toks[2:4], skip_special_tokens=False)
    assert stop_str

    out = backend.sample(
        base_model=TINY,
        adapter_id=adapter_id,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0, stop=(stop_str,)),
    )
    seq = out.sequences[0]
    assert seq.stop_reason == "stop"
    assert 0 < len(seq.tokens) < len(toks)
    assert stop_str not in tok.decode(list(seq.tokens), skip_special_tokens=False)
    assert seq.logprobs is not None and len(seq.logprobs) == len(seq.tokens)


def test_stop_strings_per_row(backend: LocalBackend, adapter_id) -> None:
    """num_samples>1: rows truncate independently at the stop string."""
    tok = backend._get(adapter_id).tokenizer
    prompt = ModelInput.from_ints(ToyTextRenderer().encode("<|user|>\nhello\n"))
    base = backend.sample(
        base_model=TINY,
        adapter_id=adapter_id,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=12, temperature=1.0, seed=7),
        num_samples=2,
    )
    text0 = tok.decode(list(base.sequences[0].tokens), skip_special_tokens=False)
    if len(text0) < 4:
        pytest.skip(f"row-0 text too short to sub-slice: {text0!r}")
    stop_str = text0[-3:]
    assert stop_str in text0

    out = backend.sample(
        base_model=TINY,
        adapter_id=adapter_id,
        prompt=prompt,
        sampling_params=SamplingParams(max_tokens=12, temperature=1.0, seed=7, stop=(stop_str,)),
        num_samples=2,
    )
    seq0 = out.sequences[0]
    assert seq0.stop_reason == "stop"
    assert len(seq0.tokens) < len(base.sequences[0].tokens)
    assert stop_str not in tok.decode(list(seq0.tokens), skip_special_tokens=False)


def test_non_text_modality_rejected(backend: LocalBackend) -> None:
    from anvil.protocol.types import TrainConfig

    with pytest.raises(NotImplementedError, match="Phase 3"):
        backend.create_lora_session(
            TrainConfig(base_model=TINY, modalities=("text", "image"))
        )


# --- opinionated gates --------------------------------------------------------


def test_tiny_model_blocked_by_default(tmp_path) -> None:
    """sshleifer/tiny-gpt2 (hidden_size=2) cannot learn via LoRA — hard block."""
    from anvil.protocol.types import TrainConfig

    b = LocalBackend(device="cpu", root=tmp_path / "blocked")
    with pytest.raises(ModelTooSmallError, match="hidden_size=2"):
        b.create_lora_session(TrainConfig(base_model="sshleifer/tiny-gpt2"))


def test_tiny_model_allowed_with_explicit_override(tmp_path) -> None:
    from anvil.protocol.types import TrainConfig

    b = LocalBackend(device="cpu", root=tmp_path / "allowed", allow_tiny_models=True)
    aid = b.create_lora_session(TrainConfig(base_model="sshleifer/tiny-gpt2"))
    assert str(aid).startswith("adapter-")


def test_bogus_target_modules_raise(tmp_path) -> None:
    from anvil.protocol.types import TrainConfig

    b = LocalBackend(
        device="cpu", root=tmp_path / "bogus", target_modules=["no_such_module"]
    )
    with pytest.raises(ValueError):
        b.create_lora_session(TrainConfig(base_model=TINY))
