#!/usr/bin/env python3
"""Agentic watch → decide → act dogfood (no external LLM required).

Modes
-----
* **local** (default): run a short job, classify ``metrics.jsonl``, append
  ``decisions.jsonl``, optionally export / multi-cycle continue.
* **control**: if ``ANVIL_CONTROL_URL`` is up, list runs / observe via
  :class:`~anvil.agent.client.AnvilControlClient` after the local train.

Examples::

  python scripts/agent_dogfood.py --steps 5
  python scripts/agent_dogfood.py --job robot_offline --steps 3
  python scripts/agent_dogfood.py --cycles 2 --steps 3 --export
  ANVIL_CONTROL_URL=http://127.0.0.1:7600 python scripts/agent_dogfood.py --control
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _run_job(
    job: str,
    *,
    steps: int,
    endpoint: str,
    run_dir: Path,
    export: bool,
) -> tuple[str, list[float]]:
    if job == "robot_offline":
        from anvil.recipes.robot_offline import run_robot_offline, toy_robot_trajectories

        res = run_robot_offline(
            trajectories=toy_robot_trajectories(),
            steps=steps,
            endpoint=endpoint,
            run_dir=str(run_dir),
            fetch_remote=False,
            early_stop=False,
            export_dir=str(run_dir / "export") if export else None,
        )
        return res.adapter_id, list(res.losses)

    from anvil.protocol.messages import Example, Message
    from anvil.recipes.sft import run_sft

    examples = [
        Example(
            messages=(
                Message(role="user", content="What is 2+2?"),
                Message(role="assistant", content="4"),
            )
        ),
        Example(
            messages=(
                Message(role="user", content="Capital of France?"),
                Message(role="assistant", content="Paris"),
            )
        ),
    ]
    sft = run_sft(
        base_model="sshleifer/tiny-gpt2",
        examples=examples,
        steps=steps,
        endpoint=endpoint,
        run_dir=str(run_dir),
        job="sft",
        early_stop=False,
        export_dir=str(run_dir / "export") if export else None,
    )
    return sft.adapter_id, list(sft.losses)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Watch→decide→act dogfood (rule brain)")
    p.add_argument("--job", choices=("sft", "robot_offline"), default="sft")
    p.add_argument("--steps", type=int, default=5, help="train steps per cycle")
    p.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="watch→decide→act cycles (train → classify → optional continue)",
    )
    p.add_argument("--run-dir", default=None)
    p.add_argument("--endpoint", default="fake://")
    p.add_argument("--control", action="store_true", help="also hit control plane")
    p.add_argument("--control-url", default=None)
    p.add_argument("--export", action="store_true", help="export peft when decision says so")
    args = p.parse_args(argv)

    from anvil.agent.decide import (
        ActKind,
        append_decision,
        decide_from_run_dir,
    )

    run_dir = Path(args.run_dir or "/tmp/anvil-agent-dogfood")
    run_dir.mkdir(parents=True, exist_ok=True)

    cycles = max(1, int(args.cycles))
    adapter_id = ""
    losses: list[float] = []
    cycle_log: list[dict] = []
    acted: list[str] = []

    for c in range(cycles):
        adapter_id, step_losses = _run_job(
            args.job,
            steps=args.steps,
            endpoint=args.endpoint,
            run_dir=run_dir,
            export=args.export and c == cycles - 1,
        )
        losses.extend(step_losses)
        decision = decide_from_run_dir(run_dir)
        dec_path = append_decision(run_dir, decision)
        cycle_acted: list[str] = []
        if decision.action == ActKind.EXPORT and args.export:
            cycle_acted.append(f"export path={run_dir / 'export'}")
        elif decision.action == ActKind.WAIT:
            cycle_acted.append("wait — continue next cycle" if c + 1 < cycles else "wait")
        elif decision.action == ActKind.LOWER_LR:
            cycle_acted.append(f"knobs_patch={decision.knobs_patch}")
        elif decision.action in {ActKind.PAUSE, ActKind.STOP}:
            cycle_acted.append(f"{decision.action.value} — halt remaining cycles")
            acted.extend(cycle_acted)
            cycle_log.append(
                {
                    "cycle": c,
                    "decision": decision.to_public(),
                    "acted": cycle_acted,
                    "decisions_path": str(dec_path),
                }
            )
            break
        else:
            cycle_acted.append(decision.action.value)
        acted.extend(cycle_acted)
        cycle_log.append(
            {
                "cycle": c,
                "decision": decision.to_public(),
                "acted": cycle_acted,
                "decisions_path": str(dec_path),
            }
        )

    control_info: dict = {}
    if args.control:
        try:
            from anvil.agent.client import AnvilControlClient

            client = AnvilControlClient(base_url=args.control_url)
            control_info["health"] = client.health()
            control_info["overview"] = client.overview()
            runs = client.list_runs()
            control_info["n_runs"] = len(runs) if isinstance(runs, list) else runs
            acted.append("control plane reachable")
        except Exception as e:  # noqa: BLE001
            control_info["error"] = str(e)
            acted.append(f"control unreachable: {e}")

    last = cycle_log[-1]["decision"] if cycle_log else {}
    summary = {
        "run_dir": str(run_dir),
        "adapter_id": adapter_id,
        "steps_per_cycle": args.steps,
        "cycles": len(cycle_log),
        "losses": losses,
        "decision": last,
        "cycle_log": cycle_log,
        "acted": acted,
        "control": control_info,
    }
    print(json.dumps(summary, indent=2))
    if last.get("classification") == "broken" and not losses:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
