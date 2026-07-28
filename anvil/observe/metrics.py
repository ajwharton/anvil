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
# J1: J-Lens residual readouts (see anvil.observe.jlens)
JLENS_FILENAME = "jlens.jsonl"


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
        self.jlens_path = self.run_dir / JLENS_FILENAME

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
        adapter_synced: bool | None = None,
        snapshot_path: str | None = None,
        sample_endpoint: str | None = None,
    ) -> dict[str, Any]:
        fb = dict(fb_metrics or {})
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "type": "step",
            "job": "grpo",
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
            "adapter_synced": adapter_synced,
            "snapshot_path": snapshot_path,
            "sample_endpoint": sample_endpoint,
        }
        _append_jsonl(self.metrics_path, record)
        return record

    def log_sft_step(
        self,
        *,
        step: int,
        loss: float,
        n_datums: int,
        n_image_refs: int = 0,
        n_tokens: float | None = None,
        fb_metrics: dict[str, float] | None = None,
        wall_time_s: float | None = None,
        job: str = "sft",
    ) -> dict[str, Any]:
        """SFT / VLM SFT step record for the same metrics.jsonl SSOT as GRPO.

        Live sufficiency signals: loss, wall time, n_image_refs (vision),
        optional n_tokens from the backend. No reward / advantage fields —
        the observe UI charts loss when ``job`` is sft/vlm_sft.
        """
        fb = dict(fb_metrics or {})
        if n_tokens is None and "n_tokens" in fb:
            n_tokens = float(fb["n_tokens"])
        if "n_image_refs" in fb and n_image_refs == 0:
            n_image_refs = int(fb["n_image_refs"])
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "type": "step",
            "job": str(job),
            "ts": time.time(),
            "step": int(step),
            "loss": float(loss),
            "n_datums": int(n_datums),
            "n_image_refs": int(n_image_refs),
            "n_tokens": None if n_tokens is None else float(n_tokens),
            "fb": fb,
            "wall_time_s": wall_time_s,
        }
        _append_jsonl(self.metrics_path, record)
        return record

    def log_dpo_step(
        self,
        *,
        step: int,
        loss: float,
        n_pairs: int,
        preferred_tokens: float | None = None,
        rejected_tokens: float | None = None,
        margin: float | None = None,
        length_bias: float | None = None,
        fb_metrics: dict[str, float] | None = None,
        wall_time_s: float | None = None,
        job: str = "dpo",
    ) -> dict[str, Any]:
        """Preference/DPO step on the same metrics.jsonl SSOT.

        Signals: loss, pair count, optional margin proxy (preferred vs rejected
        length/score), length_bias (preferred_len - rejected_len). Observe UI
        charts loss when ``job`` is dpo (same path as SFT).
        """
        fb = dict(fb_metrics or {})
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "type": "step",
            "job": str(job),
            "ts": time.time(),
            "step": int(step),
            "loss": float(loss),
            "n_datums": int(n_pairs),
            "n_pairs": int(n_pairs),
            "preferred_tokens": None if preferred_tokens is None else float(preferred_tokens),
            "rejected_tokens": None if rejected_tokens is None else float(rejected_tokens),
            "margin": None if margin is None else float(margin),
            "length_bias": None if length_bias is None else float(length_bias),
            "fb": fb,
            "wall_time_s": wall_time_s,
        }
        _append_jsonl(self.metrics_path, record)
        return record

    def log_event(
        self,
        *,
        step: int,
        event: str,
        reason: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Non-step event (e.g. early_stop) written to metrics.jsonl for the UI."""
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "type": "event",
            "event": str(event),
            "ts": time.time(),
            "step": int(step),
            "reason": reason,
        }
        for k, v in extra.items():
            if k not in record:
                record[k] = v
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
        target: str | None = None,
        job: str | None = None,
    ) -> dict[str, Any]:
        """Append a live-policy probe (GRPO reward score or SFT sample + optional match).

        For SFT/VLM, ``reward`` may be a simple match score (0/1) against ``target``;
        ``text`` is the greedy completion under the live adapter.
        """
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "type": "probe",
            "ts": time.time(),
            "step": int(step),
            "probe_idx": int(probe_idx),
            "tokens": [int(t) for t in tokens],
            "text": text,
            "reward": None if reward is None else float(reward),
            "target": target,
            "job": job,
        }
        _append_jsonl(self.probes_path, record)
        return record

    def log_jlens(
        self,
        *,
        step: int,
        probe_idx: int | None = None,
        prompt_preview: str | None = None,
        completion_preview: str | None = None,
        layers: Iterable[int] | None = None,
        positions: str | Iterable[int] = "last_prompt",
        top_k: int = 5,
        slice_: dict[str, Any] | None = None,
        signals: dict[str, Any] | None = None,
        stages: Iterable[Iterable[str]] | None = None,
        answer: str | None = None,
        lens_id: str | None = None,
        adapter_id: str | None = None,
        wall_time_s: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one J-Lens debugger record to ``jlens.jsonl`` (schema v1).

        Does not run the lens — callers pass a compact ``slice_`` (or precomputed
        ``signals``). Prefer building via :func:`anvil.observe.jlens.build_jlens_record`.
        """
        from anvil.observe.jlens import build_jlens_record

        pos: str | list[int]
        if isinstance(positions, str):
            pos = positions
        else:
            pos = [int(p) for p in positions]
        st: list[list[str]] | None = None
        if stages is not None:
            st = [[str(x) for x in stage] for stage in stages]
        record = build_jlens_record(
            step=step,
            probe_idx=probe_idx,
            prompt_preview=prompt_preview,
            completion_preview=completion_preview,
            layers=list(layers) if layers is not None else None,
            positions=pos,
            top_k=top_k,
            slice_=slice_,
            signals=signals,
            stages=st,
            answer=answer,
            lens_id=lens_id,
            adapter_id=adapter_id,
            wall_time_s=wall_time_s,
            extra=extra,
        )
        # keep package SCHEMA_VERSION on envelope for observe readers that
        # only know metrics.SCHEMA_VERSION; jlens has its own field too
        record["schema_version"] = max(int(record.get("schema_version", 1)), SCHEMA_VERSION)
        _append_jsonl(self.jlens_path, record)
        return record
