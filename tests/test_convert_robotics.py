"""Phase 3.B: episode_pack / path_jsonl → CAS + Anvil JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil.data.convert import (
    ConvertConfig,
    convert_corpus,
    format_action_text,
    write_demo_episode_pack,
)
from anvil.data.ingest import examples_from_vlm_jsonl
from anvil.media import LocalMediaStore


def test_format_action_text_vector():
    s = format_action_text([0.1, -0.2, 1.0], decimals=2)
    assert s == "0.10 -0.20 1.00"


def test_format_action_text_dict():
    s = format_action_text({"gripper": 1, "x": 0.5}, decimals=1)
    assert "gripper=1.0" in s
    assert "x=0.5" in s


def test_episode_pack_convert_and_resume(tmp_path):
    pack = write_demo_episode_pack(tmp_path / "pack", n_episodes=4, frames_per=3)
    media = tmp_path / "media"
    out = tmp_path / "out" / "rows.jsonl"

    r1 = convert_corpus(
        ConvertConfig(
            source=pack,
            media_root=media,
            output_jsonl=out,
            max_rows=5,
            dataset="demo_bridge_like",
            license_note="synthetic",
            resume=True,
        )
    )
    assert r1.n_rows == 5
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 5
    row = json.loads(lines[0])
    assert row["images"][0].startswith("cas://sha256/")
    assert row["instruction"].startswith("pick up object")
    assert row["dataset"] == "demo_bridge_like"
    # meta.json license wins over ConvertConfig.license_note
    assert row["license"] == "synthetic-demo-not-bridge"
    assert " " in row["response"]  # vector text

    # resume: no new rows if already done for those keys; can add more budget
    r2 = convert_corpus(
        ConvertConfig(
            source=pack,
            media_root=media,
            output_jsonl=out,
            max_rows=100,
            dataset="demo_bridge_like",
            resume=True,
        )
    )
    # remaining frames from 4 ep * 3 frames = 12 total; 5 done → 7 more
    assert r2.n_rows == 7
    lines2 = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines2) == 12

    # third run: all done → 0 new
    r3 = convert_corpus(
        ConvertConfig(
            source=pack,
            media_root=media,
            output_jsonl=out,
            max_rows=100,
            resume=True,
        )
    )
    assert r3.n_rows == 0
    assert r3.n_skipped >= 1


def test_keyframe_mode(tmp_path):
    pack = write_demo_episode_pack(tmp_path / "pack", n_episodes=3, frames_per=4)
    out = tmp_path / "kf.jsonl"
    r = convert_corpus(
        ConvertConfig(
            source=pack,
            media_root=tmp_path / "media",
            output_jsonl=out,
            row_mode="keyframe",
            frames_per_episode=4,
        )
    )
    assert r.n_rows == 3  # one per episode
    assert r.n_episodes == 3


def test_path_jsonl_convert(tmp_path):
    from anvil.data.convert import write_solid_png

    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    p0 = img_dir / "a.png"
    write_solid_png(p0, rgb=(1, 2, 3), size=8)
    src = tmp_path / "in.jsonl"
    src.write_text(
        json.dumps(
            {
                "instruction": "what color?",
                "images": ["imgs/a.png"],
                "response": "dark",
                "episode_id": "e1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "cas.jsonl"
    r = convert_corpus(
        ConvertConfig(
            source=src,
            media_root=tmp_path / "media",
            output_jsonl=out,
            source_kind="path_jsonl",
            dataset="path_test",
        )
    )
    assert r.n_rows == 1
    row = json.loads(out.read_text().strip())
    assert row["images"][0].startswith("cas://")
    assert row["response"] == "dark"

    store = LocalMediaStore(tmp_path / "media")
    exs = examples_from_vlm_jsonl(out, store, limit=1)
    assert len(exs) == 1
    assert exs[0].image_refs()


def test_frames_per_episode_subsample(tmp_path):
    pack = write_demo_episode_pack(tmp_path / "pack", n_episodes=1, frames_per=10)
    out = tmp_path / "sub.jsonl"
    r = convert_corpus(
        ConvertConfig(
            source=pack,
            media_root=tmp_path / "media",
            output_jsonl=out,
            frames_per_episode=3,
            max_episodes=1,
        )
    )
    assert r.n_rows == 3


def test_convert_rejects_bad_kind(tmp_path):
    with pytest.raises(ValueError, match="source_kind"):
        convert_corpus(
            ConvertConfig(
                source=tmp_path,
                media_root=tmp_path / "m",
                output_jsonl=tmp_path / "o.jsonl",
                source_kind="rlds_magic",
            )
        )


def test_cli_demo(tmp_path):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "convert_robotics_corpus.py"
    media = tmp_path / "media"
    out = tmp_path / "out.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--demo",
            "--demo-episodes",
            "2",
            "--media-root",
            str(media),
            "--output",
            str(out),
            "--max-rows",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert out.is_file()
    n = sum(1 for ln in out.read_text().splitlines() if ln.strip())
    assert n == 3
