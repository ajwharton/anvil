"""Expert-v2 checkpoint + resume (SFT / GRPO on fake://)."""

from __future__ import annotations

import json
from pathlib import Path

from anvil.client.service import ServiceClient
from anvil.observe.metrics import read_jsonl
from anvil.protocol.messages import Example, Message, TextPart
from anvil.protocol.types import CheckpointRef
from anvil.recipes.checkpoint import (
    RESUME_FILENAME,
    apply_resume_to_client,
    load_resume_state,
    resume_path,
    save_train_checkpoint,
    write_resume_state,
)
from anvil.recipes.dpo import PreferencePair, run_dpo
from anvil.recipes.grpo import run_grpo
from anvil.recipes.sft import run_sft


def _ex(user: str = "2+2?", assistant: str = "4") -> Example:
    return Example(
        messages=(
            Message(role="user", content=(TextPart(text=user),)),
            Message(role="assistant", content=(TextPart(text=assistant),)),
        )
    )


def test_write_load_resume_roundtrip(tmp_path: Path) -> None:
    ref = CheckpointRef(name="step-3", path=str(tmp_path / "ck"), kind="train_state")
    (tmp_path / "ck").mkdir()
    write_resume_state(
        tmp_path,
        job="sft",
        steps_completed=3,
        base_model="toy/lm",
        checkpoint=ref,
        adapter_id="adapter-abc",
        losses=[1.0, 0.5, 0.4],
    )
    assert resume_path(tmp_path).name == RESUME_FILENAME
    state = load_resume_state(tmp_path)
    assert state is not None
    assert state.steps_completed == 3
    assert state.job == "sft"
    assert state.losses == (1.0, 0.5, 0.4)
    assert state.checkpoint.path == ref.path
    assert load_resume_state(tmp_path / "missing") is None


def test_save_train_checkpoint_fake(tmp_path: Path) -> None:
    root = tmp_path / "fake-root"
    run_dir = tmp_path / "run"
    svc = ServiceClient(endpoint=f"fake://{root}")
    tc = svc.create_lora_training_client(base_model="toy/lm", rank=4)
    # one train step so weights exist
    from anvil.recipes.sft import examples_to_data

    data = examples_to_data([_ex()])
    tc.forward_backward(data).result()
    from anvil.protocol.types import AdamParams

    tc.optim_step(AdamParams(learning_rate=1e-4)).result()
    ref = save_train_checkpoint(
        tc,
        run_dir=run_dir,
        job="sft",
        steps_completed=1,
        base_model="toy/lm",
        losses=[0.9],
    )
    assert Path(ref.path).exists()
    state = load_resume_state(run_dir)
    assert state is not None
    assert state.steps_completed == 1
    # new client + load
    tc2 = svc.create_lora_training_client(base_model="toy/lm", rank=4)
    apply_resume_to_client(tc2, state)
    svc.close()


def test_run_sft_checkpoint_and_resume_without_full_replay(tmp_path: Path) -> None:
    """Short SFT: stop mid-budget, resume continues step index + adapter."""
    root = tmp_path / "fake-root"
    run_dir = tmp_path / "sft-run"
    endpoint = f"fake://{root}"
    examples = [_ex()]

    part1 = run_sft(
        base_model="toy/lm",
        examples=examples,
        steps=3,
        endpoint=endpoint,
        run_dir=str(run_dir),
        early_stop=False,
        checkpoint_every=1,
        stop_on_southward=False,
    )
    assert part1.steps_run == 3
    assert part1.checkpoint_path is not None
    assert (run_dir / RESUME_FILENAME).is_file()
    state = load_resume_state(run_dir)
    assert state is not None
    assert state.steps_completed == 3
    assert len(state.losses) == 3

    metrics_before = read_jsonl(run_dir / "metrics.jsonl")
    step_records_before = [r for r in metrics_before if r.get("type") == "step"]
    assert len(step_records_before) == 3
    ckpt_events = [
        r
        for r in metrics_before
        if r.get("type") == "event" and r.get("event") == "checkpoint"
    ]
    assert ckpt_events

    # Resume toward total budget 5 → only 2 new steps (no full replay of 0..2)
    part2 = run_sft(
        base_model="toy/lm",
        examples=examples,
        steps=5,
        endpoint=endpoint,
        run_dir=str(run_dir),
        early_stop=False,
        checkpoint_every=1,
        resume=True,
        stop_on_southward=False,
    )
    assert part2.resumed_from_step == 3
    assert part2.steps_run == 2
    assert len(part2.losses) == 2

    metrics_after = read_jsonl(run_dir / "metrics.jsonl")
    resume_events = [
        r for r in metrics_after if r.get("type") == "event" and r.get("event") == "resume"
    ]
    assert resume_events
    step_records = [r for r in metrics_after if r.get("type") == "step"]
    # 3 from part1 + 2 from part2
    assert len(step_records) == 5
    assert [r["step"] for r in step_records] == [0, 1, 2, 3, 4]

    final = load_resume_state(run_dir)
    assert final is not None
    assert final.steps_completed == 5
    assert len(final.losses) == 5


def test_run_sft_resume_already_complete(tmp_path: Path) -> None:
    root = tmp_path / "fake-root"
    run_dir = tmp_path / "done"
    endpoint = f"fake://{root}"
    run_sft(
        base_model="toy/lm",
        examples=[_ex()],
        steps=2,
        endpoint=endpoint,
        run_dir=str(run_dir),
        early_stop=False,
        checkpoint_every=1,
        stop_on_southward=False,
    )
    again = run_sft(
        base_model="toy/lm",
        examples=[_ex()],
        steps=2,
        endpoint=endpoint,
        run_dir=str(run_dir),
        early_stop=False,
        checkpoint_every=1,
        resume=True,
        stop_on_southward=False,
    )
    assert again.steps_run == 0
    assert again.resumed_from_step == 2


def test_run_sft_checkpoint_every_requires_run_dir() -> None:
    try:
        run_sft(
            base_model="toy/lm",
            steps=1,
            endpoint="fake://",
            checkpoint_every=1,
            early_stop=False,
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "run_dir" in str(e)


def test_run_grpo_checkpoint_and_resume(tmp_path: Path) -> None:
    root = tmp_path / "fake-root"
    run_dir = tmp_path / "grpo-run"
    endpoint = f"fake://{root}"

    part1 = run_grpo(
        base_model="toy/lm",
        steps=2,
        endpoint=endpoint,
        run_dir=str(run_dir),
        early_stop=False,
        checkpoint_every=1,
        stop_on_southward=False,
        group_size=2,
    )
    assert part1.steps_run == 2
    assert (run_dir / RESUME_FILENAME).is_file()
    state = load_resume_state(run_dir)
    assert state is not None
    assert state.job == "grpo"
    assert state.steps_completed == 2

    part2 = run_grpo(
        base_model="toy/lm",
        steps=4,
        endpoint=endpoint,
        run_dir=str(run_dir),
        early_stop=False,
        checkpoint_every=1,
        resume=True,
        stop_on_southward=False,
        group_size=2,
    )
    assert part2.resumed_from_step == 2
    assert part2.steps_run == 2

    steps = [r["step"] for r in read_jsonl(run_dir / "metrics.jsonl") if r.get("type") == "step"]
    assert steps == [0, 1, 2, 3]
    final = load_resume_state(run_dir)
    assert final is not None
    assert final.steps_completed == 4


def test_resume_json_is_valid_json_object(tmp_path: Path) -> None:
    run_sft(
        base_model="toy/lm",
        examples=[_ex()],
        steps=1,
        endpoint=f"fake://{tmp_path / 'r'}",
        run_dir=str(tmp_path / "run"),
        early_stop=False,
        checkpoint_every=1,
        stop_on_southward=False,
    )
    raw = json.loads((tmp_path / "run" / RESUME_FILENAME).read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert "checkpoint" in raw and "path" in raw["checkpoint"]


def test_run_dpo_checkpoint_and_resume(tmp_path: Path) -> None:
    """DPO resume parity: continue from resume.json without replaying early steps."""
    root = tmp_path / "fake-root"
    run_dir = tmp_path / "dpo-run"
    endpoint = f"fake://{root}"
    pairs = [
        PreferencePair(prompt="hi", preferred="ok", rejected="a much longer bad answer"),
    ]
    part1 = run_dpo(
        endpoint=endpoint,
        pairs=pairs,
        steps=3,
        run_dir=str(run_dir),
        early_stop=False,
        stop_on_southward=False,
        checkpoint_every=1,
    )
    assert part1.steps_run == 3
    assert part1.checkpoint_path is not None
    assert (run_dir / RESUME_FILENAME).is_file()
    state = load_resume_state(run_dir)
    assert state is not None
    assert state.job == "dpo"
    assert state.steps_completed == 3
    assert len(state.losses) == 3

    part2 = run_dpo(
        endpoint=endpoint,
        pairs=pairs,
        steps=5,
        run_dir=str(run_dir),
        early_stop=False,
        stop_on_southward=False,
        checkpoint_every=1,
        resume=True,
    )
    assert part2.resumed_from_step == 3
    assert part2.steps_run == 2
    assert len(part2.losses) == 2

    metrics = read_jsonl(run_dir / "metrics.jsonl")
    assert any(r.get("event") == "resume" for r in metrics)
    assert any(r.get("event") == "checkpoint" for r in metrics)
    steps = [r["step"] for r in metrics if r.get("type") == "step"]
    assert steps == [0, 1, 2, 3, 4]
    final = load_resume_state(run_dir)
    assert final is not None
    assert final.steps_completed == 5
    assert len(final.losses) == 5


def test_run_dpo_checkpoint_every_requires_run_dir() -> None:
    try:
        run_dpo(
            endpoint="fake://",
            steps=1,
            early_stop=False,
            checkpoint_every=1,
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "run_dir" in str(e)
