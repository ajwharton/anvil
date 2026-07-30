"""Phase 4 tracks: house pack, edge export, agentic decide (fake://)."""

from __future__ import annotations

import json
from pathlib import Path

from anvil.agent.decide import (
    ActKind,
    RunClass,
    append_decision,
    classify_metrics,
    decide_from_run_dir,
)
from anvil.backends.jetson import JetsonSampleBackend, JetsonSampleConfig
from anvil.client.service import ServiceClient
from anvil.data.robot_pack import (
    HousePackConfig,
    house_pack_to_jsonl,
    house_pack_to_trajectories,
    write_demo_house_pack,
)
from anvil.export.edge import package_edge_export
from anvil.protocol.types import (
    AdamParams,
    Datum,
    ExportFormat,
    LoraConfig,
    LoraTargets,
    ModelInput,
    SamplingParams,
    TrainConfig,
)
from anvil.recipes.robot_offline import run_robot_offline


def test_house_pack_to_trajectories_and_jsonl(tmp_path: Path):
    pack = write_demo_house_pack(tmp_path / "pack", n_episodes=3, frames_per=2)
    cfg = HousePackConfig(
        source=pack,
        media_root=tmp_path / "media",
        action_scheme="bins",
    )
    res = house_pack_to_trajectories(cfg)
    assert res.n_episodes == 3
    assert res.n_steps >= 3
    assert res.trajectories[0].steps[0].observation_refs[0].startswith("cas://")

    jsonl = tmp_path / "rows.jsonl"
    house_pack_to_jsonl(cfg, jsonl)
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 3
    row = json.loads(lines[0])
    assert "instruction" in row and "response" in row and row["images"]
    # bin tokens for vector actions
    assert all(t.lstrip("-").isdigit() for t in row["response"].split())


def test_house_pack_robot_offline(tmp_path: Path):
    pack = write_demo_house_pack(tmp_path / "pack", n_episodes=4, frames_per=2)
    cfg = HousePackConfig(source=pack, media_root=tmp_path / "media")
    res = house_pack_to_trajectories(cfg)
    out = run_robot_offline(
        trajectories=res.trajectories,
        steps=2,
        endpoint="fake://",
        fetch_remote=False,
        run_dir=str(tmp_path / "run"),
        early_stop=False,
    )
    assert out.steps_run == 2
    assert out.n_train_examples >= 1
    assert (tmp_path / "run" / "metrics.jsonl").is_file()


def test_edge_export_gguf_package(tmp_path: Path):
    svc = ServiceClient(endpoint="fake://")
    tc = svc.create_lora_training_client(base_model="toy/edge-256m", rank=8)
    tokens = list(range(5, 20))
    datum = Datum(
        model_input=ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [1.0] * (len(tokens) - 1),
        },
    )
    tc.forward_backward([datum], "cross_entropy").result()
    tc.optim_step(AdamParams()).result()

    dest = tmp_path / "gguf_export"
    result = tc.export_adapter(str(dest), format="gguf")
    assert Path(result.path).is_dir()
    man = Path(result.path) / "edge_manifest.json"
    assert man.is_file()
    data = json.loads(man.read_text(encoding="utf-8"))
    assert data["format"] == "gguf"
    assert data["peft_path"]
    assert (Path(result.path) / "README.md").is_file()
    assert (Path(result.path) / "peft" / "adapter_config.json").is_file()


def test_package_edge_export_with_converter(tmp_path: Path, monkeypatch):
    # touch-style converter
    monkeypatch.setenv("ANVIL_GGUF_CONVERTER", "touch {dst}")
    dest = tmp_path / "pkg"

    def save_peft(d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")

    bundle = package_edge_export(
        fmt=ExportFormat.GGUF,
        root=dest,
        adapter_id="a1",
        base_model="smol",
        save_peft=save_peft,
    )
    assert (dest / "model.gguf").is_file()
    assert bundle.manifest.artifact_path


def test_jetson_sample_dry_run():
    be = JetsonSampleBackend(JetsonSampleConfig(dry_run=True, model="smolvlm-256m"))
    aid = be.create_lora_session(
        TrainConfig(base_model="smolvlm-256m", lora=LoraConfig(rank=4, targets=LoraTargets()))
    )
    result = be.sample(
        base_model="smolvlm-256m",
        adapter_id=aid,
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=8),
    )
    assert result.sequences
    assert "jetson-dry-run" in be.last_text
    # train verbs blocked
    try:
        be.forward_backward(aid)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass


def test_classify_metrics_healthy_and_cliff():
    healthy = [
        {"type": "step", "job": "sft", "loss": 1.0, "step": 0},
        {"type": "step", "job": "sft", "loss": 0.8, "step": 1},
        {"type": "step", "job": "sft", "loss": 0.5, "step": 2},
        {"type": "step", "job": "sft", "loss": 0.3, "step": 3},
    ]
    d = classify_metrics(healthy, job="sft", min_steps=3)
    assert d.classification == RunClass.HEALTHY
    assert d.action in {ActKind.WAIT, ActKind.EXPORT}

    rising = [
        {"type": "step", "job": "sft", "loss": 0.2, "step": i} for i in range(2)
    ] + [
        {"type": "step", "job": "sft", "loss": 0.5, "step": i} for i in range(2, 6)
    ]
    d2 = classify_metrics(rising, job="sft", min_steps=3)
    assert d2.classification == RunClass.CLIFF
    assert d2.action == ActKind.LOWER_LR

    empty = classify_metrics([])
    assert empty.classification == RunClass.BROKEN


def test_decide_from_run_dir_after_sft(tmp_path: Path):
    from anvil.protocol.messages import Example, Message
    from anvil.recipes.sft import run_sft

    run_dir = tmp_path / "dogfood"
    run_sft(
        base_model="toy/tiny",
        examples=[
            Example(
                messages=(
                    Message(role="user", content="hi"),
                    Message(role="assistant", content="hello"),
                )
            )
        ],
        steps=4,
        endpoint="fake://",
        run_dir=str(run_dir),
        early_stop=False,
    )
    decision = decide_from_run_dir(run_dir)
    path = append_decision(run_dir, decision)
    assert path.is_file()
    assert decision.classification in {
        RunClass.HEALTHY,
        RunClass.NOISY,
        RunClass.CLIFF,
        RunClass.BROKEN,
    }


def test_fake_backend_still_exports_peft(tmp_path: Path):
    svc = ServiceClient(endpoint="fake://")
    tc = svc.create_lora_training_client(base_model="toy/x", rank=4)
    r = tc.export_adapter(str(tmp_path / "peft"), format="peft")
    assert (Path(r.path) / "adapter_config.json").is_file()
