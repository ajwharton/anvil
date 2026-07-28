"""Southward-turn detectors — machine-readable flags for live sufficiency.

Scan ``metrics.jsonl`` + ``probes.jsonl`` for patterns that mean “returns are
going the wrong way” even if a single scalar still looks alive:

- **advantage_collapse** — group reward std ≈ 0 (GRPO)
- **reward_up_probes_down** — reward_mean rising while probe rewards fall
- **probe_regression** — probe match/reward worse than an earlier window
- **loss_flat_probes_down** — SFT/DPO loss plateau while probes worsen
- **length_bias_spike** — DPO preferred completions much longer than rejected

Flags are pure functions of records (no training). Callers can log them as
``type=event, event=southward`` or fail smokes / stop agents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from anvil.observe.metrics import METRICS_FILENAME, PROBES_FILENAME, advantage_collapsed, read_jsonl

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SouthwardFlag:
    """One detector firing."""

    name: str
    severity: str  # warn | cliff
    step: int | None
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        return d


@dataclass(frozen=True, slots=True)
class SouthwardReport:
    flags: tuple[SouthwardFlag, ...]
    n_metric_steps: int = 0
    n_probe_records: int = 0

    @property
    def ok(self) -> bool:
        return not any(f.severity == "cliff" for f in self.flags)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.flags]

    def to_public(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": self.ok,
            "n_metric_steps": self.n_metric_steps,
            "n_probe_records": self.n_probe_records,
            "flags": [f.to_public() for f in self.flags],
        }


def _step_records(metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in metrics
        if r.get("type") == "step"
        or (r.get("type") is None and r.get("step") is not None and "event" not in r)
    ]


def _probe_rewards_by_step(probes: Sequence[dict[str, Any]]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for p in probes:
        if p.get("reward") is None:
            continue
        try:
            step = int(p["step"])
            out.setdefault(step, []).append(float(p["reward"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _mean(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def detect_advantage_collapse(
    steps: Sequence[dict[str, Any]],
    *,
    last_n: int = 3,
) -> list[SouthwardFlag]:
    flags: list[SouthwardFlag] = []
    window = list(steps[-last_n:]) if last_n > 0 else list(steps)
    collapsed = [s for s in window if advantage_collapsed(s)]
    if collapsed and len(collapsed) >= max(1, min(last_n, len(window))):
        last = collapsed[-1]
        flags.append(
            SouthwardFlag(
                name="advantage_collapse",
                severity="cliff",
                step=int(last["step"]) if last.get("step") is not None else None,
                detail="group_reward_std_mean ≈ 0 for recent GRPO steps",
                evidence={
                    "last_group_reward_std_mean": last.get("group_reward_std_mean"),
                    "n_collapsed": len(collapsed),
                },
            )
        )
    return flags


def detect_reward_up_probes_down(
    steps: Sequence[dict[str, Any]],
    probes: Sequence[dict[str, Any]],
    *,
    window: int = 5,
    reward_eps: float = 0.02,
    probe_eps: float = 0.05,
) -> list[SouthwardFlag]:
    """Reward mean up over window while mean probe reward down — hacking signature."""
    flags: list[SouthwardFlag] = []
    if len(steps) < window * 2:
        return flags
    early = steps[:window]
    late = steps[-window:]
    if any(s.get("reward_mean") is None for s in early + late):
        return flags
    r0 = _mean([float(s["reward_mean"]) for s in early])
    r1 = _mean([float(s["reward_mean"]) for s in late])
    by = _probe_rewards_by_step(probes)
    if not by:
        return flags
    steps_sorted = sorted(by)
    if len(steps_sorted) < 2:
        return flags
    early_p = _mean([x for s in steps_sorted[: max(1, len(steps_sorted) // 3)] for x in by[s]])
    late_p = _mean([x for s in steps_sorted[-max(1, len(steps_sorted) // 3) :] for x in by[s]])
    if r0 is None or r1 is None or early_p is None or late_p is None:
        return flags
    if (r1 - r0) >= reward_eps and (early_p - late_p) >= probe_eps:
        flags.append(
            SouthwardFlag(
                name="reward_up_probes_down",
                severity="cliff",
                step=int(late[-1]["step"]) if late[-1].get("step") is not None else None,
                detail="reward_mean rose while probe rewards fell",
                evidence={
                    "reward_early": r0,
                    "reward_late": r1,
                    "probe_early": early_p,
                    "probe_late": late_p,
                },
            )
        )
    return flags


def detect_probe_regression(
    probes: Sequence[dict[str, Any]],
    *,
    min_records: int = 6,
    drop: float = 0.25,
) -> list[SouthwardFlag]:
    """Mean probe reward in the last third is substantially worse than first third."""
    flags: list[SouthwardFlag] = []
    scored = [p for p in probes if p.get("reward") is not None]
    if len(scored) < min_records:
        return flags
    n = len(scored)
    first = scored[: n // 3] or scored[:1]
    last = scored[-(n // 3) :] or scored[-1:]
    m0 = _mean([float(p["reward"]) for p in first])
    m1 = _mean([float(p["reward"]) for p in last])
    if m0 is None or m1 is None:
        return flags
    if (m0 - m1) >= drop:
        flags.append(
            SouthwardFlag(
                name="probe_regression",
                severity="cliff" if (m0 - m1) >= drop * 1.5 else "warn",
                step=int(last[-1]["step"]) if last[-1].get("step") is not None else None,
                detail=f"probe reward {m0:.3f} → {m1:.3f}",
                evidence={"probe_early": m0, "probe_late": m1, "drop": m0 - m1},
            )
        )
    return flags


def detect_loss_flat_probes_down(
    steps: Sequence[dict[str, Any]],
    probes: Sequence[dict[str, Any]],
    *,
    flat_rel: float = 0.02,
    flat_abs: float = 1e-3,
    probe_drop: float = 0.2,
    min_steps: int = 8,
) -> list[SouthwardFlag]:
    """SFT/DPO: late loss not improving but probes got worse."""
    flags: list[SouthwardFlag] = []
    job = (steps[-1].get("job") if steps else None) or ""
    if job not in {"sft", "vlm_sft", "dpo"} and not (
        steps and steps[-1].get("reward_mean") is None and steps[-1].get("loss") is not None
    ):
        # still allow if loss-only steps
        if not steps or steps[-1].get("loss") is None:
            return flags
    if len(steps) < min_steps:
        return flags
    losses = [float(s["loss"]) for s in steps if s.get("loss") is not None]
    if len(losses) < min_steps:
        return flags
    # last half nearly flat
    mid = len(losses) // 2
    late = losses[mid:]
    if not late:
        return flags
    lo, hi = min(late), max(late)
    span = hi - lo
    scale = max(abs(late[0]), 1e-12)
    flat = span <= max(flat_abs, flat_rel * scale)
    if not flat:
        return flags
    scored = [p for p in probes if p.get("reward") is not None]
    if len(scored) < 4:
        return flags
    n = len(scored)
    m0 = _mean([float(p["reward"]) for p in scored[: n // 3]])
    m1 = _mean([float(p["reward"]) for p in scored[-(n // 3) :]])
    if m0 is None or m1 is None or (m0 - m1) < probe_drop:
        return flags
    flags.append(
        SouthwardFlag(
            name="loss_flat_probes_down",
            severity="cliff",
            step=int(steps[-1]["step"]) if steps[-1].get("step") is not None else None,
            detail="loss plateau while probes regressed",
            evidence={
                "loss_late_span": span,
                "probe_early": m0,
                "probe_late": m1,
                "job": job,
            },
        )
    )
    return flags


def detect_length_bias_spike(
    steps: Sequence[dict[str, Any]],
    *,
    threshold: float = 8.0,
    last_n: int = 5,
) -> list[SouthwardFlag]:
    """DPO length_bias large positive on recent steps (preferred much longer)."""
    flags: list[SouthwardFlag] = []
    dpo = [s for s in steps if s.get("job") == "dpo" and s.get("length_bias") is not None]
    if len(dpo) < last_n:
        return flags
    window = dpo[-last_n:]
    biases = [float(s["length_bias"]) for s in window]
    m = _mean(biases)
    if m is not None and m >= threshold:
        flags.append(
            SouthwardFlag(
                name="length_bias_spike",
                severity="cliff" if m >= threshold * 1.5 else "warn",
                step=int(window[-1]["step"]) if window[-1].get("step") is not None else None,
                detail=f"DPO length_bias mean={m:.2f} (preferred longer)",
                evidence={"length_bias_mean": m, "threshold": threshold},
            )
        )
    return flags


def scan_records(
    metrics: Sequence[dict[str, Any]],
    probes: Sequence[dict[str, Any]] | None = None,
) -> SouthwardReport:
    """Run all detectors on in-memory records."""
    steps = _step_records(list(metrics))
    probe_list = list(probes or [])
    flags: list[SouthwardFlag] = []
    flags.extend(detect_advantage_collapse(steps))
    flags.extend(detect_reward_up_probes_down(steps, probe_list))
    flags.extend(detect_probe_regression(probe_list))
    flags.extend(detect_loss_flat_probes_down(steps, probe_list))
    flags.extend(detect_length_bias_spike(steps))
    return SouthwardReport(
        flags=tuple(flags),
        n_metric_steps=len(steps),
        n_probe_records=len(probe_list),
    )


def scan_run_dir(run_dir: str | Path) -> SouthwardReport:
    """Load metrics/probes from an observe run directory."""
    d = Path(run_dir)
    metrics = read_jsonl(d / METRICS_FILENAME) if (d / METRICS_FILENAME).is_file() else []
    probes = read_jsonl(d / PROBES_FILENAME) if (d / PROBES_FILENAME).is_file() else []
    return scan_records(metrics, probes)


def log_southward_flags(
    run_dir: str | Path,
    report: SouthwardReport | None = None,
    *,
    step: int | None = None,
) -> list[dict[str, Any]]:
    """Append each flag as ``event=southward`` to metrics.jsonl."""
    from anvil.observe.metrics import RunMetricsWriter

    d = Path(run_dir)
    rep = report if report is not None else scan_run_dir(d)
    if not rep.flags:
        return []
    writer = RunMetricsWriter(d)
    out: list[dict[str, Any]] = []
    for f in rep.flags:
        rec = writer.log_event(
            step=step if step is not None else (f.step if f.step is not None else -1),
            event="southward",
            reason=f.name,
            severity=f.severity,
            detail=f.detail,
            **f.evidence,
        )
        out.append(rec)
    return out


def scan_and_log(run_dir: str | Path) -> SouthwardReport:
    """Scan a run dir and persist cliff/warn events."""
    rep = scan_run_dir(run_dir)
    log_southward_flags(run_dir, rep)
    return rep


def cliff_stop_reason(report: SouthwardReport) -> str | None:
    """If any cliff-severity flag is present, return an early_stop reason string."""
    cliffs = [f for f in report.flags if f.severity == "cliff"]
    if not cliffs:
        return None
    names = ",".join(sorted({f.name for f in cliffs}))
    return f"southward:{names}"


def maybe_stop_on_southward(
    run_dir: str | Path | None,
    *,
    step: int,
    enabled: bool = True,
    min_steps: int = 5,
) -> str | None:
    """Mid-train hook: scan disk artifacts; log flags; return stop reason or None.

    Requires ``run_dir`` with enough steps so detectors have signal. No-op when
    disabled, missing dir, or ``step + 1 < min_steps``.
    """
    if not enabled or run_dir is None:
        return None
    if step + 1 < min_steps:
        return None
    d = Path(run_dir)
    if not d.is_dir():
        return None
    rep = scan_run_dir(d)
    if not rep.flags:
        return None
    log_southward_flags(d, rep, step=step)
    return cliff_stop_reason(rep)


__all__ = [
    "SouthwardFlag",
    "SouthwardReport",
    "cliff_stop_reason",
    "detect_advantage_collapse",
    "detect_length_bias_spike",
    "detect_loss_flat_probes_down",
    "detect_probe_regression",
    "detect_reward_up_probes_down",
    "log_southward_flags",
    "maybe_stop_on_southward",
    "scan_and_log",
    "scan_records",
    "scan_run_dir",
]
