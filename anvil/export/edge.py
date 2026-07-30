"""Phase 4.C — edge export packages (PEFT → merge → GGUF/ONNX/TRT recipe).

Lab trains LoRA; the robot loads a **small** student (SmolVLM-256M class).
Export always materializes:

1. PEFT adapter dir (always)
2. Optional merged HF weights (when model supports merge_and_unload)
3. ``edge_manifest.json`` with conversion steps for GGUF / ONNX / TRT
4. Optional external converter hooks via env vars (never required for CI)

We do **not** vendor llama.cpp or TensorRT builders. Operators point
``ANVIL_GGUF_CONVERTER`` / ``ANVIL_ONNX_CONVERTER`` at host tools when ready.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from anvil.protocol.types import ExportFormat, ExportResult

SequenceStr = list[str] | tuple[str, ...]


@dataclass
class EdgeManifest:
    """Machine-readable edge export bundle."""

    format: str
    adapter_id: str
    base_model: str | None
    peft_path: str | None
    merged_path: str | None
    artifact_path: str | None
    steps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    env_hooks: dict[str, str] = field(default_factory=dict)
    ollama_hint: str | None = None

    def to_public(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_public(), indent=2) + "\n", encoding="utf-8")


@dataclass
class EdgeExportBundle:
    result: ExportResult
    manifest: EdgeManifest
    root: Path

    @property
    def path(self) -> str:
        return str(self.root)


def _write_readme(root: Path, manifest: EdgeManifest) -> None:
    lines = [
        "# Anvil edge export",
        "",
        f"- format: `{manifest.format}`",
        f"- adapter_id: `{manifest.adapter_id}`",
        f"- base_model: `{manifest.base_model or 'unknown'}`",
        "",
        "## Layout",
        "",
        "- `peft/` — LoRA adapter (PEFT)",
        "- `merged/` — full HF weights when merge succeeded",
        "- `edge_manifest.json` — machine-readable steps",
        "- optional converter output (`.gguf`, `model.onnx`, …)",
        "",
        "## Steps",
        "",
    ]
    for i, s in enumerate(manifest.steps, 1):
        lines.append(f"{i}. {s}")
    if manifest.notes:
        lines.extend(["", "## Notes", ""])
        for n in manifest.notes:
            lines.append(f"- {n}")
    if manifest.ollama_hint:
        lines.extend(["", "## Ollama / on-robot", "", manifest.ollama_hint, ""])
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def gguf_recipe_steps(*, merged_or_peft: str, out_gguf: str) -> list[str]:
    return [
        f"Have PEFT/merged weights at {merged_or_peft}",
        "Install llama.cpp (or HF GGUF convert tooling) on the **lab** host — not required in Anvil CI",
        f"Convert HF → GGUF (example): python convert_hf_to_gguf.py {merged_or_peft} --outfile {out_gguf}",
        "Quantize if needed (Q8_0 / Q4_K_M) for Orin 4GB",
        "Copy GGUF to robot; register with Ollama (`ollama create …`) or llama.cpp server",
        "Never actuate motors from raw samples without a supervisor",
    ]


def onnx_recipe_steps(*, merged_or_peft: str, out_onnx: str) -> list[str]:
    return [
        f"Have merged HF weights at {merged_or_peft}",
        "Export ONNX with optimum/onnxruntime or torch.onnx (VLM export is model-specific)",
        f"Target path: {out_onnx}",
        "Optional: TensorRT build on Jetson from ONNX (`trtexec`)",
        "Validate FPS/power on device before closing the edge loop",
    ]


def trt_recipe_steps(*, onnx_path: str) -> list[str]:
    return [
        f"Start from ONNX at {onnx_path} (or build ONNX first)",
        "On Jetson: trtexec --onnx=… --saveEngine=model.trt --fp16",
        "Prefer FP16 on Orin Nano; watch memory for VLM vision towers",
        "Ship engine + tokenizer/processor assets together",
    ]


def build_manifest(
    *,
    fmt: ExportFormat,
    adapter_id: str,
    base_model: str | None,
    root: Path,
    peft_path: Path | None,
    merged_path: Path | None,
    artifact_path: Path | None = None,
    extra_notes: SequenceStr | None = None,
) -> EdgeManifest:
    peft_s = str(peft_path) if peft_path else None
    merged_s = str(merged_path) if merged_path else None
    art_s = str(artifact_path) if artifact_path else None
    weight_src = merged_s or peft_s or str(root)

    notes = [
        "Anvil owns PEFT/merge; GGUF/ONNX/TRT converters are host tools.",
        "SmolVLM-256M is the default on-robot student when memory is tight.",
    ]
    if extra_notes:
        notes.extend(list(extra_notes))

    env_hooks = {
        "ANVIL_GGUF_CONVERTER": "command that reads HF dir and writes .gguf (optional)",
        "ANVIL_ONNX_CONVERTER": "command that reads HF dir and writes model.onnx (optional)",
    }

    if fmt == ExportFormat.GGUF:
        out = art_s or str(root / "model.gguf")
        steps = gguf_recipe_steps(merged_or_peft=weight_src, out_gguf=out)
        ollama = (
            "Modelfile example: FROM ./model.gguf  then  ollama create smolvlm-anvil -f Modelfile"
        )
    elif fmt == ExportFormat.ONNX:
        out = art_s or str(root / "model.onnx")
        steps = onnx_recipe_steps(merged_or_peft=weight_src, out_onnx=out)
        ollama = None
    elif fmt == ExportFormat.TRT:
        onnx = str(root / "model.onnx")
        steps = trt_recipe_steps(onnx_path=onnx)
        ollama = None
        notes.append("TRT engines are device-specific — rebuild on the target Jetson.")
    elif fmt == ExportFormat.MERGED_HF:
        steps = [
            f"Merged HF weights at {merged_s or weight_src}",
            "Load with transformers+peft disabled, or convert further to GGUF/ONNX",
        ]
        ollama = None
    else:  # PEFT
        steps = [
            f"PEFT adapter at {peft_s or weight_src}",
            "Load base + adapter on lab; merge before quantizing for edge",
        ]
        ollama = None

    return EdgeManifest(
        format=fmt.value,
        adapter_id=adapter_id,
        base_model=base_model,
        peft_path=peft_s,
        merged_path=merged_s,
        artifact_path=art_s,
        steps=steps,
        notes=notes,
        env_hooks=env_hooks,
        ollama_hint=ollama,
    )


def try_run_converter(
    *,
    env_key: str,
    source_dir: Path,
    dest: Path,
    timeout: float = 600.0,
) -> tuple[bool, str]:
    """Run optional shell converter. Returns (ok, message)."""
    cmd_tmpl = os.environ.get(env_key, "").strip()
    if not cmd_tmpl:
        return False, f"{env_key} not set — recipe only"
    # Support {src} and {dst} placeholders
    cmd = cmd_tmpl.replace("{src}", str(source_dir)).replace("{dst}", str(dest))
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"{env_key} timed out"
    except OSError as e:
        return False, f"{env_key} failed to start: {e}"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[:500]
        return False, f"{env_key} exit {proc.returncode}: {err}"
    if not dest.exists():
        return False, f"{env_key} exited 0 but missing {dest}"
    return True, f"{env_key} wrote {dest}"


def package_edge_export(
    *,
    fmt: ExportFormat,
    root: Path,
    adapter_id: str,
    base_model: str | None,
    save_peft: Any | None = None,
    save_merged: Any | None = None,
    extra_notes: SequenceStr | None = None,
) -> EdgeExportBundle:
    """Build an edge export directory.

    ``save_peft`` / ``save_merged`` are callables ``(Path) -> None`` when the
    backend can materialize weights. For fake backends they write stubs.
    """
    root = Path(root)
    if root.exists():
        # do not wipe arbitrary trees — require empty or create subdirs
        root.mkdir(parents=True, exist_ok=True)
    else:
        root.mkdir(parents=True, exist_ok=True)

    peft_dir = root / "peft"
    merged_dir = root / "merged"
    peft_path: Path | None = None
    merged_path: Path | None = None
    artifact: Path | None = None

    if save_peft is not None:
        peft_dir.mkdir(parents=True, exist_ok=True)
        save_peft(peft_dir)
        peft_path = peft_dir
    elif fmt == ExportFormat.PEFT:
        peft_dir.mkdir(parents=True, exist_ok=True)
        peft_path = peft_dir

    if save_merged is not None and fmt in {
        ExportFormat.MERGED_HF,
        ExportFormat.GGUF,
        ExportFormat.ONNX,
        ExportFormat.TRT,
    }:
        merged_dir.mkdir(parents=True, exist_ok=True)
        try:
            save_merged(merged_dir)
            merged_path = merged_dir
        except Exception as e:  # noqa: BLE001 — merge optional
            (root / "merge_error.txt").write_text(str(e), encoding="utf-8")

    weight_src = merged_path or peft_path or root

    if fmt == ExportFormat.GGUF:
        artifact = root / "model.gguf"
        ok, msg = try_run_converter(
            env_key="ANVIL_GGUF_CONVERTER",
            source_dir=weight_src,
            dest=artifact,
        )
        notes = list(extra_notes or [])
        notes.append(msg)
        if not ok:
            artifact = None
        manifest = build_manifest(
            fmt=fmt,
            adapter_id=adapter_id,
            base_model=base_model,
            root=root,
            peft_path=peft_path,
            merged_path=merged_path,
            artifact_path=artifact,
            extra_notes=notes,
        )
    elif fmt == ExportFormat.ONNX:
        artifact = root / "model.onnx"
        ok, msg = try_run_converter(
            env_key="ANVIL_ONNX_CONVERTER",
            source_dir=weight_src,
            dest=artifact,
        )
        notes = list(extra_notes or [])
        notes.append(msg)
        if not ok:
            artifact = None
        manifest = build_manifest(
            fmt=fmt,
            adapter_id=adapter_id,
            base_model=base_model,
            root=root,
            peft_path=peft_path,
            merged_path=merged_path,
            artifact_path=artifact,
            extra_notes=notes,
        )
    elif fmt == ExportFormat.TRT:
        manifest = build_manifest(
            fmt=fmt,
            adapter_id=adapter_id,
            base_model=base_model,
            root=root,
            peft_path=peft_path,
            merged_path=merged_path,
            extra_notes=extra_notes,
        )
    elif fmt == ExportFormat.MERGED_HF:
        if merged_path is None and peft_path is not None:
            # copy peft as best-effort marker when merge unavailable
            shutil.copytree(peft_path, merged_dir, dirs_exist_ok=True)
            merged_path = merged_dir
            extra = list(extra_notes or [])
            extra.append("merge unavailable — peft tree copied under merged/ as placeholder")
            extra_notes = extra
        manifest = build_manifest(
            fmt=fmt,
            adapter_id=adapter_id,
            base_model=base_model,
            root=root,
            peft_path=peft_path,
            merged_path=merged_path,
            extra_notes=extra_notes,
        )
    else:
        manifest = build_manifest(
            fmt=ExportFormat.PEFT,
            adapter_id=adapter_id,
            base_model=base_model,
            root=root,
            peft_path=peft_path,
            merged_path=merged_path,
            extra_notes=extra_notes,
        )

    manifest.write(root / "edge_manifest.json")
    _write_readme(root, manifest)
    result = ExportResult(format=fmt, path=str(root), adapter_id=adapter_id)
    return EdgeExportBundle(result=result, manifest=manifest, root=root)
