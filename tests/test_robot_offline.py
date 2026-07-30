"""Phase 4.A: action tokenization + robot_offline recipe (fake://)."""

from __future__ import annotations

from anvil.protocol.action_tokens import ActionTokenizer, default_edge_tokenizer
from anvil.protocol.trajectory import Trajectory, TrajectoryStep
from anvil.recipes.catalog import get_recipe
from anvil.recipes.profiles import JobPattern, ModelShape, infer_shape, plan_recipe
from anvil.recipes.robot_offline import (
    DEFAULT_ROBOT_BASE,
    run_robot_offline,
    split_heldout_episodes,
    toy_robot_trajectories,
    trajectories_to_robot_examples,
)
from anvil.recipes.throughput import throughput_defaults


def test_action_tokenizer_bins_roundtrip():
    tok = ActionTokenizer(scheme="bins", n_bins=256, min_action=-1.0, max_action=1.0)
    action = [0.0, 0.5, -0.25, 1.0, -1.0, 0.1, 0.0]
    text = tok.encode(action)
    parts = text.split()
    assert len(parts) == 7
    assert all(p.isdigit() or (p.startswith("-") and p[1:].isdigit()) for p in parts)
    # Mid bin for 0.0 ≈ 128
    assert int(parts[0]) == 128
    recovered = tok.decode(text)
    assert len(recovered) == 7
    for a, b in zip(action, recovered):
        assert abs(a - b) < 1.0 / 128  # half-bin tolerance-ish


def test_action_tokenizer_clip_and_continuous():
    tok = ActionTokenizer(scheme="bins", n_bins=10, min_action=0.0, max_action=1.0)
    assert tok.encode([1.5]) == "9"  # clipped to max
    assert tok.encode([-0.5]) == "0"
    cont = ActionTokenizer(scheme="continuous", decimals=2)
    assert cont.encode([0.1, -0.2]) == "0.10 -0.20"
    assert cont.decode("0.10 -0.20") == [0.1, -0.2]


def test_action_tokenizer_string_passthrough_and_dict():
    tok = default_edge_tokenizer()
    assert tok.encode("close gripper") == "close gripper"
    dtext = tok.encode({"z": 0.5, "x": 0.0})
    # dict keys sorted → x then z
    assert len(dtext.split()) == 2
    public = tok.to_public()
    assert ActionTokenizer.from_public(public).n_bins == 256


def test_trajectory_uses_action_tokenizer():
    dig = "d" * 64
    tr = Trajectory(
        episode_id="e1",
        steps=(
            TrajectoryStep(
                observation_refs=(f"cas://sha256/{dig}.png",),
                instruction="pick",
                action=[0.0, 1.0],
            ),
        ),
    )
    tok = ActionTokenizer(scheme="bins", n_bins=8, min_action=-1.0, max_action=1.0)
    exs = tr.to_vlm_sft_examples(action_tokenizer=tok)
    assert len(exs) == 1
    resp = exs[0].messages[1].parts()[0].text  # type: ignore[union-attr]
    assert resp == tok.encode([0.0, 1.0])


def test_smol_infer_shape_edge():
    assert infer_shape("HuggingFaceTB/SmolVLM-256M-Instruct") == ModelShape.EDGE_STUDENT
    assert infer_shape("HuggingFaceTB/SmolLM2-135M-Instruct") == ModelShape.DENSE_LM
    # smolvlm family without size still edge
    assert infer_shape("local/SmolVLM-Instruct") == ModelShape.EDGE_STUDENT


def test_robot_offline_plan_edge_knobs():
    plan = plan_recipe(
        base_model=DEFAULT_ROBOT_BASE,
        pattern=JobPattern.ROBOT_OFFLINE,
        shape=ModelShape.EDGE_STUDENT,
        use_card=False,
    )
    assert plan.pattern == JobPattern.ROBOT_OFFLINE
    assert plan.lora.rank <= 8
    assert plan.lora.vision_encoder is False
    assert "text" in plan.modalities and "image" in plan.modalities


def test_robot_offline_catalog_and_throughput():
    spec = get_recipe("robot_offline_edge")
    assert spec is not None
    assert spec.pattern == JobPattern.ROBOT_OFFLINE
    assert spec.default_rank == 8
    td = throughput_defaults(shape=ModelShape.EDGE_STUDENT, pattern=JobPattern.ROBOT_OFFLINE)
    assert td.rank == 8
    assert td.batch_size == 1


def test_split_heldout_and_examples():
    trs = toy_robot_trajectories()
    train, hold = split_heldout_episodes(trs, heldout_fraction=0.5)
    assert len(train) == 1 and len(hold) == 1
    exs = trajectories_to_robot_examples(train)
    assert len(exs) >= 1
    # assistant target is bin tokens
    text = exs[0].messages[1].parts()[0].text  # type: ignore[union-attr]
    assert all(p.lstrip("-").isdigit() for p in text.split())


def test_run_robot_offline_fake(tmp_path):
    run_dir = tmp_path / "robot-run"
    res = run_robot_offline(
        base_model=DEFAULT_ROBOT_BASE,
        trajectories=toy_robot_trajectories(),
        steps=2,
        endpoint="fake://",
        fetch_remote=False,
        run_dir=str(run_dir),
        early_stop=False,
    )
    assert res.steps_run == 2
    assert res.n_train_examples >= 1
    assert res.n_heldout_episodes >= 1
    assert res.adapter_id
    assert res.action_tokenizer.get("scheme") == "bins"
    assert (run_dir / "metrics.jsonl").is_file()
    # losses recorded
    assert len(res.losses) == 2


def test_run_robot_offline_text_only_fake(tmp_path):
    """SmolLM path: no vision, action bins as pure text SFT."""
    trs = [
        Trajectory(
            episode_id="t0",
            steps=(
                TrajectoryStep(
                    instruction="reach",
                    action=[0.2, -0.1, 0.0],
                    observation_refs=(),
                ),
            ),
        ),
        Trajectory(
            episode_id="t1",
            steps=(
                TrajectoryStep(
                    instruction="retract",
                    action=[-0.2, 0.1, 0.0],
                    observation_refs=(),
                ),
            ),
        ),
    ]
    res = run_robot_offline(
        base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        trajectories=trs,
        steps=1,
        endpoint="fake://",
        fetch_remote=False,
        run_dir=str(tmp_path / "text-robot"),
        text_only=True,
        early_stop=False,
        heldout_fraction=0.5,
    )
    assert res.steps_run == 1
    assert res.n_train_examples >= 1
