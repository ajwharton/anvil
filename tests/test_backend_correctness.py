"""Offline regression tests for LocalBackend, SampleWorkerPool, and vision_rl fixes."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from anvil.backends.fake import FakeBackend  # noqa: E402
from anvil.backends.local import LocalBackend, _LocalSession  # noqa: E402
from anvil.protocol.types import (  # noqa: E402
    AdapterId,
    Datum,
    ExportFormat,
    LoraConfig,
    ModelInput,
    TrainConfig,
)
from anvil.recipes.vision_rl import (  # noqa: E402
    VisionRLQueueRecipe,
    VisionRLStage,
    VisionRollout,
    run_vision_rl_queue,
)
from anvil.workers.pool import SampleWorkerPool  # noqa: E402


class _TinyLM(torch.nn.Module):
    """Minimal causal LM with an explicit weight tensor for offline torch tests."""

    def __init__(self, vocab: int = 16, hidden: int = 8) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None):
        logits = self.head(self.emb(input_ids))
        return SimpleNamespace(logits=logits)


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def save_pretrained(self, dest: Path) -> None:
        (Path(dest) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def _make_backend_with_session(tmp_path: Path) -> tuple[LocalBackend, AdapterId, _LocalSession]:
    be = LocalBackend(device="cpu", root=tmp_path / "local")
    model = _TinyLM()
    cfg = TrainConfig(base_model="toy/local", lora=LoraConfig(rank=4))
    aid = AdapterId("adapter-test")
    sess = _LocalSession(config=cfg, model=model, tokenizer=_DummyTokenizer())
    be._sessions[aid.value] = sess
    return be, aid, sess


def test_dpo_weighted_mean_logp_handles_ignore_index_minus_100(tmp_path: Path):
    be, _aid, sess = _make_backend_with_session(tmp_path)
    datum = Datum(
        model_input=ModelInput.from_ints([2, 3, 4, 5]),
        loss_fn_inputs={
            "target_tokens": [-100, -100, 5, 6],
            "weights": [0.0, 0.0, 1.0, 1.0],
        },
    )
    lp = be._weighted_mean_logp(sess, datum)
    assert torch.isfinite(lp)


def test_rl_forward_backward_clamps_extreme_log_ratio(tmp_path: Path):
    be, aid, _sess = _make_backend_with_session(tmp_path)
    # Huge negative old_logprob would overflow exp(cur_lp - old_lp) without clamping
    datum = Datum(
        model_input=ModelInput.from_ints([2, 3, 4, 5]),
        loss_fn_inputs={
            "target_tokens": [4, 5],
            "logprobs": [-500.0, -500.0],
            "advantages": [0.0, 1.0],
            "weights": [1.0, 1.0],
        },
    )
    out = be.forward_backward(aid, [datum], "ppo")
    assert math.isfinite(out.loss)
    assert math.isfinite(out.metrics["mean_ratio"])


def test_using_sample_weights_restores_live_weights(tmp_path: Path, monkeypatch):
    import peft

    be, aid, sess = _make_backend_with_session(tmp_path)
    # Simulate PEFT state-dict get/set on sess.model.head.weight
    monkeypatch.setattr(
        peft,
        "get_peft_model_state_dict",
        lambda m: {"head.weight": m.head.weight.data.clone()},
    )

    def _set_sd(m, sd):
        m.head.weight.data.copy_(sd["head.weight"])

    monkeypatch.setattr(peft, "set_peft_model_state_dict", _set_sd)

    # Snapshot step-0 weights (all 1.0), then mutate live weights to 5.0
    with torch.no_grad():
        sess.model.head.weight.fill_(1.0)
    sess.sampler_snapshots["snap-0"] = {"head.weight": torch.ones_like(sess.model.head.weight)}
    with torch.no_grad():
        sess.model.head.weight.fill_(5.0)

    # Inside _using_sample_weights, model must see snapshot (1.0)
    with be._using_sample_weights("toy/local", aid) as (m, _tok):
        assert torch.allclose(m.head.weight, torch.ones_like(m.head.weight))

    # After exiting, live training weights (5.0) must be restored intact
    assert torch.allclose(sess.model.head.weight, torch.full_like(sess.model.head.weight, 5.0))


def test_export_merged_uses_non_destructive_merge_unmerge(tmp_path: Path):
    be, aid, sess = _make_backend_with_session(tmp_path)
    calls: list[str] = []

    class _MockBase:
        def save_pretrained(self, dest: Path) -> None:
            calls.append("save_base")
            (Path(dest) / "config.json").write_text("{}", encoding="utf-8")

    class _MockPeft:
        def save_pretrained(self, dest: Path) -> None:
            calls.append("save_peft")
            (Path(dest) / "adapter_config.json").write_text("{}", encoding="utf-8")

        def merge_adapter(self) -> None:
            calls.append("merge")

        def unmerge_adapter(self) -> None:
            calls.append("unmerge")

        def get_base_model(self) -> _MockBase:
            return _MockBase()

        def merge_and_unload(self):  # pragma: no cover - must not be called
            raise AssertionError("merge_and_unload must not destroy live PeftModel")

    sess.model = _MockPeft()
    out_dir = tmp_path / "merged_out"
    res = be.export_adapter(aid, ExportFormat.MERGED_HF, str(out_dir))
    assert Path(res.path) == out_dir
    assert calls == ["merge", "save_base", "unmerge"]
    assert (out_dir / "config.json").is_file()


def test_sample_worker_pool_snapshot_for_sample_raises(tmp_path: Path):
    pool = SampleWorkerPool(backends=[FakeBackend(root=tmp_path / "fake")])
    with pytest.raises(NotImplementedError):
        pool.snapshot_for_sample(AdapterId("a1"), "snap")


def test_vision_rl_queue_forwards_tokenize(tmp_path: Path):
    seen_instructions: list[str] = []

    def custom_tokenize(text: str) -> list[int]:
        seen_instructions.append(text)
        return [42, 43, 44, 45]

    rollout = VisionRollout(
        id="r1",
        instruction="Locate the mug on the table.",
        required_keywords=("mug",),
    )
    recipe = VisionRLQueueRecipe(
        id="tok_test",
        name="Tokenize forwarding test",
        stages=(VisionRLStage(id="s1", rollouts=(rollout,), max_steps=1),),
    )
    res = run_vision_rl_queue(
        recipe,
        endpoint="fake://",
        run_dir=str(tmp_path / "run"),
        tokenize=custom_tokenize,
    )
    assert res.stages_run == 1
    assert seen_instructions == ["Locate the mug on the table."]
