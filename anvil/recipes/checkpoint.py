"""Train checkpoint + resume helpers (Expert-v2).

Minimal long-job path: adapter weights via backend ``save_state`` /
``load_state``, plus a small ``resume.json`` under the run dir that records
steps completed and the checkpoint ref. Recipes call these; they do not
own backend PEFT details.

Layout under ``run_dir``::

    run_dir/
      resume.json          # SSOT for resume (step + checkpoint path)
      metrics.jsonl        # unchanged observe SSOT
      ...

``resume.json`` schema (v1)::

    {
      "schema_version": 1,
      "job": "sft" | "grpo" | ...,
      "steps_completed": int,   # next step index to run
      "base_model": str,
      "adapter_id": str,        # informational (new session gets a new id)
      "checkpoint": {
        "name": str,
        "path": str,
        "kind": "train_state"
      },
      "losses": [...],          # optional early-stop continuity
      "mean_reward": [...],     # optional GRPO
      "dead_signals": [...]     # optional GRPO
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from anvil.protocol.types import CheckpointRef

RESUME_FILENAME = "resume.json"
RESUME_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Parsed ``resume.json`` for continuing a recipe without full replay."""

    job: str
    steps_completed: int
    base_model: str
    checkpoint: CheckpointRef
    adapter_id: str | None = None
    losses: tuple[float, ...] = ()
    mean_reward: tuple[float, ...] = ()
    dead_signals: tuple[str | None, ...] = ()
    extra: Mapping[str, Any] | None = None

    @property
    def checkpoint_path(self) -> str:
        return self.checkpoint.path


def resume_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / RESUME_FILENAME


def write_resume_state(
    run_dir: str | Path,
    *,
    job: str,
    steps_completed: int,
    base_model: str,
    checkpoint: CheckpointRef,
    adapter_id: str | None = None,
    losses: Sequence[float] | None = None,
    mean_reward: Sequence[float] | None = None,
    dead_signals: Sequence[str | None] | None = None,
    **extra: Any,
) -> Path:
    """Atomically write ``run_dir/resume.json`` (tmp + replace)."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "job": str(job),
        "steps_completed": int(steps_completed),
        "base_model": str(base_model),
        "adapter_id": adapter_id,
        "checkpoint": {
            "name": checkpoint.name,
            "path": checkpoint.path,
            "kind": checkpoint.kind,
        },
    }
    if losses is not None:
        payload["losses"] = [float(x) for x in losses]
    if mean_reward is not None:
        payload["mean_reward"] = [float(x) for x in mean_reward]
    if dead_signals is not None:
        payload["dead_signals"] = list(dead_signals)
    for k, v in extra.items():
        if k not in payload:
            payload[k] = v
    path = resume_path(root)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_resume_state(run_dir: str | Path) -> ResumeState | None:
    """Load ``resume.json`` if present; return None if missing."""
    path = resume_path(run_dir)
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return resume_state_from_dict(raw)


def resume_state_from_dict(raw: Mapping[str, Any]) -> ResumeState:
    if not isinstance(raw, Mapping):
        raise TypeError("resume state must be a mapping")
    ver = int(raw.get("schema_version", 1))
    if ver > RESUME_SCHEMA_VERSION:
        raise ValueError(
            f"resume.json schema_version {ver} is newer than supported "
            f"{RESUME_SCHEMA_VERSION}"
        )
    ck = raw.get("checkpoint")
    if not isinstance(ck, Mapping):
        raise ValueError("resume.json missing checkpoint object")
    name = str(ck.get("name") or "")
    path = str(ck.get("path") or "")
    if not path:
        raise ValueError("resume.json checkpoint.path is required")
    kind = str(ck.get("kind") or "train_state")
    steps = int(raw.get("steps_completed", 0))
    if steps < 0:
        raise ValueError(f"steps_completed must be >= 0, got {steps}")
    losses_raw = raw.get("losses") or []
    mean_raw = raw.get("mean_reward") or []
    dead_raw = raw.get("dead_signals") or []
    known = {
        "schema_version",
        "job",
        "steps_completed",
        "base_model",
        "adapter_id",
        "checkpoint",
        "losses",
        "mean_reward",
        "dead_signals",
    }
    extra = {k: v for k, v in raw.items() if k not in known}
    return ResumeState(
        job=str(raw.get("job") or "unknown"),
        steps_completed=steps,
        base_model=str(raw.get("base_model") or ""),
        checkpoint=CheckpointRef(name=name or "resume", path=path, kind=kind),
        adapter_id=str(raw["adapter_id"]) if raw.get("adapter_id") else None,
        losses=tuple(float(x) for x in losses_raw),
        mean_reward=tuple(float(x) for x in mean_raw),
        dead_signals=tuple(
            (None if x is None else str(x)) for x in dead_raw
        ),
        extra=extra or None,
    )


def save_train_checkpoint(
    training_client: Any,
    *,
    run_dir: str | Path,
    job: str,
    steps_completed: int,
    base_model: str,
    name: str | None = None,
    losses: Sequence[float] | None = None,
    mean_reward: Sequence[float] | None = None,
    dead_signals: Sequence[str | None] | None = None,
    **extra: Any,
) -> CheckpointRef:
    """``save_state`` on the training client + write ``resume.json``.

    Returns the backend :class:`CheckpointRef`. Does not export PEFT for
    deploy — use ``export_adapter`` for that.
    """
    ckpt_name = name or f"step-{int(steps_completed)}"
    ref = training_client.save_state(ckpt_name)
    write_resume_state(
        run_dir,
        job=job,
        steps_completed=int(steps_completed),
        base_model=base_model,
        checkpoint=ref,
        adapter_id=str(getattr(training_client, "adapter_id", "") or None),
        losses=losses,
        mean_reward=mean_reward,
        dead_signals=dead_signals,
        **extra,
    )
    return ref


def apply_resume_to_client(
    training_client: Any,
    state: ResumeState,
) -> None:
    """Load adapter (+ optim when present) from ``state.checkpoint`` into client."""
    path = Path(state.checkpoint.path)
    if not path.exists():
        raise FileNotFoundError(
            f"resume checkpoint not found: {path} "
            f"(steps_completed={state.steps_completed})"
        )
    training_client.load_state(state.checkpoint)
