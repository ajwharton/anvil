"""Watch → classify → act (rule-based brain for dogfood / CI).

Mirrors ``prompts/agent/watch_loop.md`` without requiring an external LLM.
Operators and agents can swap this classifier for a frontier model; Anvil
owns the metric SSOT and the action surface.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class RunClass(str, Enum):
    HEALTHY = "healthy"
    NOISY = "noisy"
    CLIFF = "cliff"
    BROKEN = "broken"


class ActKind(str, Enum):
    WAIT = "wait"
    PAUSE = "pause"
    STOP = "stop"
    EXPORT = "export"
    LOWER_LR = "lower_lr"
    TIGHTEN_PROBES = "tighten_probes"
    REPORT = "report"


@dataclass(frozen=True, slots=True)
class Decision:
    classification: RunClass
    action: ActKind
    evidence: tuple[str, ...] = ()
    knobs_patch: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        d["classification"] = self.classification.value
        d["action"] = self.action.value
        return d


def load_metrics_jsonl(path: Path | str, *, tail: int = 50) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if tail and len(rows) > tail:
        rows = rows[-tail:]
    return rows


def step_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Filter metrics to training steps (skip events)."""
    out: list[dict[str, Any]] = []
    for r in rows:
        t = r.get("type")
        if t in (None, "step", "train"):
            if "loss" in r or "reward_mean" in r or "step" in r:
                out.append(dict(r))
        elif t == "step":
            out.append(dict(r))
    return out


def classify_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    job: str | None = None,
    min_steps: int = 3,
) -> Decision:
    """Classify a metrics window and recommend one action.

    Heuristics (deliberately conservative for dogfood):
    - no steps / error events → BROKEN
    - southward / cliff events → CLIFF → pause
    - SFT loss flat or rising after min_steps → CLIFF → lower_lr or stop
    - high variance reward → NOISY → wait / tighten probes
    - otherwise HEALTHY → wait (or export if run looks done)
    """
    raw = [dict(r) for r in rows]
    if not raw:
        return Decision(
            classification=RunClass.BROKEN,
            action=ActKind.REPORT,
            evidence=("no metrics rows",),
            summary="empty metrics — control plane or run_dir missing?",
        )

    # Explicit southward / error events
    for r in raw:
        if r.get("event") == "southward" or r.get("type") == "southward":
            sev = str(r.get("severity") or "cliff")
            name = str(r.get("name") or r.get("flag") or "southward")
            return Decision(
                classification=RunClass.CLIFF,
                action=ActKind.PAUSE if sev == "cliff" else ActKind.WAIT,
                evidence=(f"southward:{name}:{sev}",),
                summary=f"southward flag {name} ({sev})",
            )
        if r.get("type") == "error" or r.get("error"):
            return Decision(
                classification=RunClass.BROKEN,
                action=ActKind.PAUSE,
                evidence=(str(r.get("error") or "error event"),),
                summary="error in metrics stream",
            )

    steps = step_rows(raw)
    if not steps:
        return Decision(
            classification=RunClass.BROKEN,
            action=ActKind.REPORT,
            evidence=("no step rows after filter",),
            summary="metrics present but no train steps",
        )

    job = job or str(steps[-1].get("job") or "sft")
    evidence: list[str] = [f"job={job}", f"n_steps={len(steps)}"]

    if job in {"grpo", "rl_verifiable", "rl"}:
        return _classify_rl(steps, evidence)
    if job == "dpo":
        return _classify_sft_like(steps, evidence, min_steps=min_steps, kind="dpo")
    return _classify_sft_like(steps, evidence, min_steps=min_steps, kind=job)


def _classify_sft_like(
    steps: list[dict[str, Any]],
    evidence: list[str],
    *,
    min_steps: int,
    kind: str,
) -> Decision:
    losses = [float(s["loss"]) for s in steps if s.get("loss") is not None]
    if len(losses) < min_steps:
        evidence.append(f"warmup losses={len(losses)}<{min_steps}")
        return Decision(
            classification=RunClass.NOISY,
            action=ActKind.WAIT,
            evidence=tuple(evidence),
            summary="too few steps to judge",
        )

    first = sum(losses[: max(1, len(losses) // 4)]) / max(1, len(losses) // 4)
    last = sum(losses[-max(1, len(losses) // 4) :]) / max(1, len(losses) // 4)
    evidence.append(f"loss_first_q={first:.4f}")
    evidence.append(f"loss_last_q={last:.4f}")

    # Rising loss window → cliff
    if last > first * 1.05 and last > first + 0.02:
        return Decision(
            classification=RunClass.CLIFF,
            action=ActKind.LOWER_LR,
            evidence=tuple(evidence + ["loss rising"]),
            knobs_patch={"learning_rate": "×0.5"},
            summary=f"{kind}: loss trending up — lower LR or stop",
        )

    # Flat near end
    tail = losses[-min(5, len(losses)) :]
    if len(tail) >= 3 and max(tail) - min(tail) < 1e-4 and last < 0.05:
        return Decision(
            classification=RunClass.HEALTHY,
            action=ActKind.EXPORT,
            evidence=tuple(evidence + ["loss plateau near zero"]),
            summary=f"{kind}: looks done — export adapter",
        )

    # Healthy decrease or mild noise
    if last <= first * 0.98 or last < first - 0.01:
        return Decision(
            classification=RunClass.HEALTHY,
            action=ActKind.WAIT,
            evidence=tuple(evidence + ["loss improving"]),
            summary=f"{kind}: healthy — keep training",
        )

    return Decision(
        classification=RunClass.NOISY,
        action=ActKind.TIGHTEN_PROBES,
        evidence=tuple(evidence + ["loss flat/noisy"]),
        knobs_patch={"probe_every": 1},
        summary=f"{kind}: noisy/flat — more probes before method switch",
    )


def _classify_rl(steps: list[dict[str, Any]], evidence: list[str]) -> Decision:
    rewards = [float(s["reward_mean"]) for s in steps if s.get("reward_mean") is not None]
    if len(rewards) < 3:
        return Decision(
            classification=RunClass.NOISY,
            action=ActKind.WAIT,
            evidence=tuple(evidence + ["few reward points"]),
            summary="RL warmup",
        )
    first = sum(rewards[: len(rewards) // 3]) / max(1, len(rewards) // 3)
    last = sum(rewards[-len(rewards) // 3 :]) / max(1, len(rewards) // 3)
    evidence.append(f"reward_first={first:.4f}")
    evidence.append(f"reward_last={last:.4f}")
    stds = [
        float(s["group_reward_std_mean"])
        for s in steps
        if s.get("group_reward_std_mean") is not None
    ]
    if stds and sum(stds[-3:]) / 3 > 0.5:
        return Decision(
            classification=RunClass.NOISY,
            action=ActKind.WAIT,
            evidence=tuple(evidence + ["high group reward std"]),
            summary="RL noisy groups — wait before switch",
        )
    if last < first - 0.05:
        return Decision(
            classification=RunClass.CLIFF,
            action=ActKind.PAUSE,
            evidence=tuple(evidence + ["reward drop"]),
            summary="RL reward cliff — pause and inspect probes",
        )
    return Decision(
        classification=RunClass.HEALTHY,
        action=ActKind.WAIT,
        evidence=tuple(evidence),
        summary="RL healthy",
    )


def decide_from_run_dir(run_dir: Path | str, *, tail: int = 50) -> Decision:
    rows = load_metrics_jsonl(Path(run_dir) / "metrics.jsonl", tail=tail)
    job = None
    for r in reversed(rows):
        if r.get("job"):
            job = str(r["job"])
            break
    return classify_metrics(rows, job=job)


def append_decision(run_dir: Path | str, decision: Decision) -> Path:
    """Append decision to ``decisions.jsonl`` under run_dir."""
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "decisions.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(decision.to_public()) + "\n")
    return path
