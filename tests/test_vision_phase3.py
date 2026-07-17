"""Phase 3.0: media store, message/trajectory serde, JSONL ingest."""

from __future__ import annotations

import json

import pytest

from anvil.data.ingest import (
    example_from_vlm_row,
    examples_from_vlm_jsonl,
    put_images_from_paths,
    write_examples_jsonl,
)
from anvil.media import LocalMediaStore
from anvil.protocol import (
    Example,
    ImagePart,
    Message,
    TextPart,
    Trajectory,
    TrajectoryStep,
    trajectories_to_examples,
)


def test_media_put_path_and_mime(tmp_path):
    store = LocalMediaStore(tmp_path / "cas")
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    ref = store.put_path(img)
    assert ref.endswith(".png")
    assert store.exists(ref)
    assert store.get(ref).startswith(b"\x89PNG")
    assert store.path_for(ref).is_file()
    assert store.mime_type(ref) in ("image/png", None)  # mimetypes may miss .png on some OS


def test_media_rejects_bad_ref(tmp_path):
    store = LocalMediaStore(tmp_path / "cas")
    with pytest.raises(ValueError, match="unsupported"):
        store.get("s3://bucket/x")
    with pytest.raises(ValueError, match="bad ref"):
        store.get("cas://sha256/ab")


def test_message_example_serde_roundtrip():
    ex = Example(
        messages=(
            Message(
                role="user",
                content=(
                    TextPart(text="grasp?"),
                    ImagePart(ref="cas://sha256/" + "a" * 64 + ".jpg", detail="high"),
                ),
            ),
            Message(role="assistant", content="yes"),
        ),
        meta={"env": "robot-sim"},
    )
    pub = ex.to_public()
    back = Example.from_public(pub)
    assert back.meta["env"] == "robot-sim"
    assert back.image_refs()[0].startswith("cas://sha256/")
    assert back.messages[1].parts()[0].text == "yes"  # type: ignore[union-attr]


def test_trajectory_to_vlm_examples():
    dig = "b" * 64
    tr = Trajectory(
        episode_id="ep1",
        meta={"instruction": "pick cube"},
        steps=(
            TrajectoryStep(
                observation_refs=(f"cas://sha256/{dig}.png",),
                instruction="pick cube",
                action="close gripper",
                reward=0.0,
            ),
            TrajectoryStep(
                observation_refs=(f"cas://sha256/{dig}.png",),
                instruction="pick cube",
                action="lift",
                reward=1.0,
                done=True,
            ),
        ),
    )
    assert tr.total_reward() == 1.0
    exs = tr.to_vlm_sft_examples()
    assert len(exs) == 2
    assert exs[0].image_refs()
    assert "gripper" in exs[0].messages[1].parts()[0].text  # type: ignore[union-attr]
    assert len(trajectories_to_examples([tr])) == 2


def test_jsonl_ingest_with_paths(tmp_path):
    store = LocalMediaStore(tmp_path / "cas")
    frame = tmp_path / "obs.jpg"
    frame.write_bytes(b"JFIF-fake")
    row = {
        "instruction": "is grasp reachable?",
        "images": [str(frame)],
        "response": "yes",
        "dataset": "toy",
    }
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    exs = examples_from_vlm_jsonl(path, store)
    assert len(exs) == 1
    assert exs[0].meta["dataset"] == "toy"
    refs = exs[0].image_refs()
    assert len(refs) == 1 and store.exists(refs[0])
    # round-trip public jsonl
    out = tmp_path / "out.jsonl"
    n = write_examples_jsonl(out, exs)
    assert n == 1
    reloaded = Example.from_public(json.loads(out.read_text().splitlines()[0]))
    assert reloaded.image_refs() == refs


def test_example_from_row_prefers_cas_refs():
    dig = "c" * 64
    ref = f"cas://sha256/{dig}"
    ex = example_from_vlm_row(
        {
            "language_instruction": "place cup",
            "observation_refs": [ref],
            "action_text": "move to pose",
        }
    )
    assert ex.image_refs() == [ref]


def test_put_images_from_paths(tmp_path):
    store = LocalMediaStore(tmp_path / "cas")
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"A")
    b.write_bytes(b"B")
    refs = put_images_from_paths(store, [a, b])
    assert len(refs) == 2
    assert store.get(refs[0]) == b"A"
