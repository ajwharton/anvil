"""Phase 0 golden: one-step SFT loop against the fake backend."""

from __future__ import annotations

import anvil
from anvil import (
    AdamParams,
    Datum,
    Example,
    Message,
    ModelInput,
    ServiceClient,
    TextPart,
)
from anvil.render import ToyTextRenderer


def test_version_bumped():
    assert anvil.__version__ == "0.0.2"


def test_sft_one_step_forward_backward_optim_sample_export(tmp_path):
    svc = ServiceClient(endpoint="fake://")
    tc = svc.create_lora_training_client(
        base_model="toy/TinyLM-0.1B",
        rank=8,
        modalities=["text"],
    )

    # Manual Datum (Tinker-shaped)
    tokens = list(range(10, 30))
    datum = Datum(
        model_input=ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [0.0] * 5 + [1.0] * (len(tokens) - 6),
        },
    )

    fb = tc.forward_backward([datum], loss_fn="cross_entropy").result()
    assert fb.loss >= 0.0
    assert fb.metrics["n_examples"] == 1.0

    step = tc.optim_step(AdamParams(learning_rate=1e-3)).result()
    assert step.step == 1

    sc = tc.save_weights_and_get_sampling_client(name="step-1")
    sample = sc.sample(
        ModelInput.from_ints(tokens[:5]),
        sampling_params=anvil.SamplingParams(max_tokens=8, temperature=1.0, seed=0),
        num_samples=2,
    ).result()
    assert len(sample.sequences) == 2
    assert len(sample.sequences[0].tokens) == 8

    export_dir = tmp_path / "adapter"
    result = tc.export_adapter(str(export_dir), format="peft")
    assert result.path == str(export_dir)
    assert (export_dir / "adapter_config.json").is_file()


def test_sft_via_renderer_messages():
    """High-level path: Example → ToyTextRenderer → Datum → train step."""
    renderer = ToyTextRenderer()
    example = Example(
        messages=(
            Message(role="user", content=(TextPart(text="2+2?"),)),
            Message(role="assistant", content=(TextPart(text="4"),)),
        )
    )
    datum = renderer.render_example_for_sft(example)
    assert "target_tokens" in datum.loss_fn_inputs
    assert sum(datum.loss_fn_inputs["weights"]) > 0

    svc = ServiceClient()
    tc = svc.create_lora_training_client(base_model="toy/TinyLM", rank=4)
    loss_before = tc.forward_backward([datum], "cross_entropy").result().loss
    tc.optim_step(AdamParams(learning_rate=0.1)).result()
    # Second pass — still finite (toy weights moved)
    loss_after = tc.forward_backward([datum], "cross_entropy").result().loss
    assert loss_before == loss_before  # finite
    assert loss_after == loss_after


def test_save_load_state(tmp_path, monkeypatch):
    monkeypatch.setenv("ANVIL_FAKE_ROOT", str(tmp_path / "fake-root"))
    from anvil.backends.fake import FakeBackend

    backend = FakeBackend(root=tmp_path / "fake-root")
    svc = ServiceClient(backend=backend)
    tc = svc.create_lora_training_client(base_model="toy/m", rank=4)
    tokens = list(range(5, 20))
    datum = Datum(
        model_input=ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": [1.0] * (len(tokens) - 1)},
    )
    tc.forward_backward([datum], "cross_entropy").result()
    tc.optim_step(AdamParams(learning_rate=0.05)).result()
    ref = tc.save_state("step-1")
    assert ref.kind == "train_state"

    tc.forward_backward([datum], "cross_entropy").result()
    tc.optim_step(AdamParams(learning_rate=0.05)).result()
    assert tc.optim_step(AdamParams()).result().step == 3

    tc.load_state(ref)
    # After load, next optim_step increments from restored step
    out = tc.optim_step(AdamParams()).result()
    assert out.step == 2


def test_rl_importance_sampling_path():
    svc = ServiceClient()
    tc = svc.create_lora_training_client(base_model="toy/m", rank=4)
    tokens = list(range(20))
    datum = Datum(
        model_input=ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [1.0] * (len(tokens) - 1),
            "logprobs": [-1.0] * (len(tokens) - 1),
            "advantages": [0.5] * (len(tokens) - 1),
        },
    )
    out = tc.forward_backward([datum], "importance_sampling").result()
    assert isinstance(out.loss, float)
    tc.optim_step(AdamParams(learning_rate=1e-3)).result()
