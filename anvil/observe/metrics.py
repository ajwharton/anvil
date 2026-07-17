"""Per-run metrics scaffolding (Phase 2.5 — the RL debugger).

Every RL step appends one JSON object to ``runs/<run_id>/metrics.jsonl``;
probe completions (the "eyes" signal — no scalar catches reward hacking,
eyes do) go to ``probes.jsonl`` alongside. The web UI tails both over SSE.
All records carry ``schema_version`` so panels can evolve without breaking
old runs.

Signal order (roadmap §Phase 2.5): reward mean/std per step;
**group_reward_std_mean** — mean within-group reward std, the
advantage-collapse tripwire that usually fires FIRST; IS mean_ratio drift
(passed through from the backend's forward_backward metrics); wall time.
Entropy collapse lands when the sampler exposes it — do not fake it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

METRICS_FILENAME = "metrics.jsonl"
PROBES_FILENAME = "probes.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # open/append/close per record: a tailing reader always sees whole lines
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: str | Path, tail: int | None = None) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    records = [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records[-tail:] if tail is not None else records


def advantage_collapsed(step_record: dict[str, Any], eps: float = 1e-8) -> bool:
    """Advantage-collapse tripwire — usually the FIRST visible RL failure.

    Fires when the mean within-group reward std hits ~0: every completion in
    a group scored the same, so every advantage is 0 and the gradient signal
    has died even though loss/reward curves still look alive.
    """
    std = step_record.get("group_reward_std_mean")
    return std is not None and float(std) < eps


class RunMetricsWriter:
    """Appends step/probe records for one run (one line per record, flushed
    on close so SSE readers never see a torn line)."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.metrics_path = self.run_dir / METRICS_FILENAME
        self.probes_path = self.run_dir / PROBES_FILENAME

    def log_step(
        self,
        *,
        step: int,
        reward_mean: float,
        reward_std: float,
        group_reward_std_mean: float,
        loss: float,
        n_datums: int,
        fb_metrics: dict[str, float] | None = None,
        wall_time_s: float | None = None,
    ) -> dict[str, Any]:
        fb = dict(fb_metrics or {})
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "type": "step",
            "ts": time.time(),
            "step": int(step),
            "reward_mean": float(reward_mean),
            "reward_std": float(reward_std),
            "group_reward_std_mean": float(group_reward_std_mean),
            "loss": float(loss),
            "n_datums": int(n_datums),
            "is_mean_ratio": fb.get("mean_ratio"),
            "fb": fb,
            "wall_time_s": wall_time_s,
        }
        _append_jsonl(self.metrics_path, record)
        return record

    def log_probe(
        self,
        *,
        step: int,
        probe_idx: int,
        tokens: Iterable[int],
        text: str | None = None,
        reward: float | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "type": "probe",
            "ts": time.time(),
            "step": int(step),
            "probe_idx": int(probe_idx),
            "tokens": [int(t) for t in tokens],
            "text": text,
            "reward": None if reward is None else float(reward),
        }
        _append_jsonl(self.probes_path, record)
        return record
