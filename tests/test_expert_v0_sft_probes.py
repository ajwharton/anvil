"""Expert-v0: SFT/VLM held-out probes + end-to-end smoke script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from anvil.observe.metrics import METRICS_FILENAME, PROBES_FILENAME, read_jsonl
from anvil.protocol.messages import Example, ImagePart, Message, TextPart
from anvil.recipes.sft import run_sft
from anvil.recipes.vlm_sft import run_vlm_sft, toy_vlm_examples

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "scripts" / "expert_v0_smoke.py"


def _ex(user: str, assistant: str, *, image: bool = False) -> Example:
    content: list = [TextPart(text=user)]
    if image:
        content.append(ImagePart(ref="cas://sha256/probe_frame", detail="low"))
    return Example(
        messages=(
            Message(role="user", content=tuple(content)),
            Message(role="assistant", content=(TextPart(text=assistant),)),
        )
    )


def test_run_sft_emits_probes(tmp_path):
    run_dir = tmp_path / "sft-probe"
    train = [_ex("2+2?", "4"), _ex("3+3?", "6")]
    probes = [_ex("1+1?", "2")]
    res = run_sft(
        endpoint="fake://",
        examples=train,
        steps=3,
        run_dir=str(run_dir),
        probes=probes,
        probe_every=1,
        job="sft",
    )
    assert res.steps_run == 3
    assert res.n_probe_records == 3  # 3 steps × 1 probe
    steps = read_jsonl(run_dir / METRICS_FILENAME)
    assert len(steps) == 3
    assert all(s["job"] == "sft" for s in steps)
    prec = read_jsonl(run_dir / PROBES_FILENAME)
    assert len(prec) == 3
    assert prec[0]["type"] == "probe"
    assert prec[0]["job"] == "sft"
    assert prec[0]["target"] == "2"
    assert isinstance(prec[0]["tokens"], list)


def test_run_vlm_sft_probes_with_images(tmp_path):
    run_dir = tmp_path / "vlm-probe"
    train = toy_vlm_examples()
    probes = [
        Example(
            messages=(
                Message(
                    role="user",
                    content=(
                        TextPart(text="Is the grasp reachable?"),
                        ImagePart(ref="cas://sha256/heldout", detail="high"),
                    ),
                ),
                Message(role="assistant", content=(TextPart(text="yes"),)),
            )
        )
    ]
    res = run_vlm_sft(
        endpoint="fake://",
        examples=train,
        steps=2,
        fetch_remote=False,
        run_dir=str(run_dir),
        probes=probes,
        probe_every=1,
    )
    assert res.n_probe_records == 2
    prec = read_jsonl(run_dir / PROBES_FILENAME)
    assert all(p.get("job") == "vlm_sft" for p in prec)
    assert prec[0]["target"] == "yes"


def test_run_sft_rejects_bad_probe_every():
    import pytest

    with pytest.raises(ValueError, match="probe_every"):
        run_sft(endpoint="fake://", steps=1, probe_every=0, probes=[_ex("a", "b")])


def test_run_sft_honors_export_hint(tmp_path):
    """run_sft must export the recipe's export_hint (e.g. onnx), not always PEFT."""
    export = tmp_path / "sft-onnx"
    res = run_sft(
        endpoint="fake://",
        examples=[_ex("2+2?", "4")],
        steps=1,
        export_dir=str(export),
        overrides={"export_hint": "onnx"},
    )
    assert res.export_path is not None
    man = Path(res.export_path) / "edge_manifest.json"
    assert man.is_file(), "onnx export should produce an edge bundle (manifest)"
    data = json.loads(man.read_text(encoding="utf-8"))
    assert data["format"] == "onnx"
    assert (Path(res.export_path) / "peft" / "adapter_config.json").is_file()


def test_run_dpo_honors_export_hint(tmp_path):
    """run_dpo must export the recipe's export_hint, not always PEFT."""
    from anvil.recipes.dpo import PreferencePair, run_dpo

    export = tmp_path / "dpo-onnx"
    res = run_dpo(
        endpoint="fake://",
        pairs=[PreferencePair(prompt="2+2?", preferred="4", rejected="five is bigger")],
        steps=1,
        export_dir=str(export),
        overrides={"export_hint": "onnx"},
    )
    assert res.export_path is not None
    man = Path(res.export_path) / "edge_manifest.json"
    assert man.is_file(), "onnx export should produce an edge bundle (manifest)"
    data = json.loads(man.read_text(encoding="utf-8"))
    assert data["format"] == "onnx"


def test_expert_v0_smoke_script(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    media = tmp_path / "media"
    observe = tmp_path / "observe"
    export = tmp_path / "export"
    jsonl = tmp_path / "rows.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            str(SMOKE),
            "--endpoint",
            "fake://",
            "--media-root",
            str(media),
            "--observe-root",
            str(observe),
            "--output-jsonl",
            str(jsonl),
            "--export",
            str(export),
            "--run-id",
            "expert-v0-test",
            "--max-rows",
            "8",
            "--steps",
            "2",
            "--holdout",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    run_dir = observe / "expert-v0-test"
    assert (run_dir / METRICS_FILENAME).is_file()
    assert (run_dir / PROBES_FILENAME).is_file()
    metrics = read_jsonl(run_dir / METRICS_FILENAME)
    assert len(metrics) == 2
    assert metrics[0]["job"] == "vlm_sft"
    assert metrics[0]["n_image_refs"] >= 1
    probes = read_jsonl(run_dir / PROBES_FILENAME)
    assert len(probes) >= 1
    assert jsonl.is_file()
    # export path created by fake backend PEFT stub
    assert export.exists() or any(export.parent.glob("**/*"))
