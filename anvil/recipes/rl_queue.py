"""Multi-stage RL recipe queue — advance when a stage early-stops.

When one verifiable GRPO task hits a dead signal (ceiling/floor/collapse) and
abandons early, the next stage is already specified in the recipe and starts
immediately on the **same LoRA adapter** (curriculum), unless configured to
spawn a fresh adapter.

Recipe JSON lives under ``recipes/*.json``; see ``recipes/arith_curriculum_v1.json``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from anvil.client.service import ServiceClient
from anvil.observe.metrics import RunMetricsWriter
from anvil.recipes.grpo import GRPOResult, build_plan, run_grpo
from anvil.recipes.verifiable import (
    DEFAULT_HARD_PROBLEMS,
    detokenize_via_tokenizer,
    exact_integer_reward,
)


@dataclass(frozen=True, slots=True)
class RLStage:
    """One verifiable RL problem in the queue."""

    id: str
    prompt: str
    gold: str
    max_steps: int = 100
    group_size: int | None = None  # inherit queue default
    early_stop_patience: int | None = None


@dataclass(frozen=True, slots=True)
class RLQueueRecipe:
    """Ordered stages + advance/stop policy."""

    id: str
    name: str
    stages: tuple[RLStage, ...]
    group_size: int = 8
    early_stop_patience: int = 8
    # If early_stop_reason matches these prefixes → start next stage
    advance_on: tuple[str, ...] = ("ceiling", "collapsed")
    # If early_stop_reason matches these → stop the whole queue (no next stage)
    stop_queue_on: tuple[str, ...] = ("floor",)
    # If a stage finishes all max_steps without early stop, still advance?
    advance_on_budget: bool = True
    notes: str = ""


@dataclass
class StageOutcome:
    stage: RLStage
    result: GRPOResult
    advanced: bool
    queue_halted: bool
    observe_run_id: str


@dataclass
class RLQueueResult:
    recipe: RLQueueRecipe
    stages: list[StageOutcome] = field(default_factory=list)
    adapter_id: str | None = None

    @property
    def stages_run(self) -> int:
        return len(self.stages)


def _reason_matches(reason: str | None, prefixes: Sequence[str]) -> bool:
    if not reason:
        return False
    r = reason.lower()
    return any(r == p.lower() or r.startswith(p.lower() + "_") for p in prefixes)


def should_advance(
    reason: str | None,
    *,
    hit_budget: bool,
    advance_on: Sequence[str],
    stop_queue_on: Sequence[str],
    advance_on_budget: bool,
) -> tuple[bool, bool]:
    """Return (advance_to_next, halt_queue).

    halt_queue takes precedence (floor → stop entire recipe).
    """
    if _reason_matches(reason, stop_queue_on):
        return False, True
    if _reason_matches(reason, advance_on):
        return True, False
    if hit_budget and advance_on_budget and reason is None:
        return True, False
    # early stop for unknown reason, or incomplete without policy match
    if reason is not None:
        # treat any early_stop as advance unless stop_queue matched
        return True, False
    return False, False


def recipe_from_hard_bank(
    *,
    recipe_id: str = "hard-bank-queue",
    max_steps: int = 100,
    patience: int = 8,
) -> RLQueueRecipe:
    """Build a queue from ``DEFAULT_HARD_PROBLEMS`` (unit tests / default demo)."""
    stages = tuple(
        RLStage(
            id=f"stage-{i}-{gold}",
            prompt=prompt,
            gold=gold,
            max_steps=max_steps,
        )
        for i, (prompt, gold) in enumerate(DEFAULT_HARD_PROBLEMS)
    )
    return RLQueueRecipe(
        id=recipe_id,
        name="Default hard arithmetic curriculum",
        stages=stages,
        early_stop_patience=patience,
        notes="Auto from DEFAULT_HARD_PROBLEMS; advance on ceiling.",
    )


def load_rl_queue_recipe(path: str | Path) -> RLQueueRecipe:
    """Load a recipe JSON file."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    stages = tuple(
        RLStage(
            id=str(s["id"]),
            prompt=str(s["prompt"]),
            gold=str(s["gold"]),
            max_steps=int(s.get("max_steps", 100)),
            group_size=s.get("group_size"),
            early_stop_patience=s.get("early_stop_patience"),
        )
        for s in data["stages"]
    )
    if not stages:
        raise ValueError(f"recipe {p} has no stages")
    return RLQueueRecipe(
        id=str(data.get("id", p.stem)),
        name=str(data.get("name", p.stem)),
        stages=stages,
        group_size=int(data.get("group_size", 8)),
        early_stop_patience=int(data.get("early_stop_patience", 8)),
        advance_on=tuple(data.get("advance_on", ["ceiling", "collapsed"])),
        stop_queue_on=tuple(data.get("stop_queue_on", ["floor"])),
        advance_on_budget=bool(data.get("advance_on_budget", True)),
        notes=str(data.get("notes", "")),
    )


def _safe_run_id(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
    if not s or not re.match(r"[A-Za-z0-9]", s[0]):
        s = "q-" + s
    return s[:80]


def run_rl_queue(
    recipe: RLQueueRecipe,
    *,
    base_model: str,
    endpoint: str = "fake://",
    observe_root: str | Path | None = None,
    run_prefix: str | None = None,
    rank: int = 16,
    max_tokens: int = 16,
    temperature: float = 1.1,
    probe_every: int = 1,
    carry_adapter: bool = True,
    fake_prompts: bool = False,
) -> RLQueueResult:
    """Execute all stages until halt or exhausted.

    ``carry_adapter=True`` (default): one LoRA session for the whole queue.
    Each stage gets its own observe subdir ``{prefix}-{stage.id}``.
    """
    prefix = _safe_run_id(run_prefix or recipe.id)
    plan = build_plan(
        base_model,
        rank=rank,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    svc = ServiceClient(endpoint=endpoint)
    tc = None
    if carry_adapter and not endpoint.startswith("fake://"):
        # Create once; run_grpo reuses via training_client=
        k = plan.as_knobs()
        from anvil.protocol.types import LoraTargets

        tc = svc.create_lora_training_client(
            base_model=plan.base_model,
            rank=k["rank"],
            modalities=k["modalities"],
            lora_targets=LoraTargets(
                language=k["language_lora"],
                vision_encoder=k["vision_encoder_lora"],
                mm_projector=k["mm_projector_lora"],
            ),
        )
    elif carry_adapter and endpoint.startswith("fake://"):
        tc = svc.create_lora_training_client(
            base_model=plan.base_model,
            rank=rank,
            modalities=("text",),
        )

    tok = None
    detok = None
    if not fake_prompts and not endpoint.startswith("fake://"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        detok = detokenize_via_tokenizer(tok)

    out = RLQueueResult(recipe=recipe, adapter_id=str(tc.adapter_id) if tc else None)
    root = Path(observe_root) if observe_root else None

    try:
        for i, stage in enumerate(recipe.stages):
            stage_run_id = _safe_run_id(f"{prefix}-{stage.id}")
            run_dir = None
            writer_root = None
            if root is not None:
                run_dir = root / stage_run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                writer_root = str(run_dir)
                # stage boundary event on a queue-level log
                queue_log = root / _safe_run_id(f"{prefix}-queue")
                queue_log.mkdir(parents=True, exist_ok=True)
                RunMetricsWriter(queue_log).log_event(
                    step=i,
                    event="stage_start",
                    reason=None,
                    stage_id=stage.id,
                    stage_index=i,
                    gold=stage.gold,
                    prompt_preview=stage.prompt[:80],
                    observe_run_id=stage_run_id,
                )

            gs = stage.group_size or recipe.group_size
            patience = stage.early_stop_patience or recipe.early_stop_patience

            if fake_prompts or endpoint.startswith("fake://"):
                # Deterministic toy prompts for CI; gold encodes as constant reward in tests
                prompts = [list(range(10 + i, 26 + i))]
                probes = [list(range(10 + i, 18 + i))]
                reward_fn = None  # toy even-token unless test injects via stage gold "1"
                if stage.gold == "always_one":
                    reward_fn = lambda _t, _toks: 1.0  # noqa: E731
                elif stage.gold == "always_zero":
                    reward_fn = lambda _t, _toks: 0.0  # noqa: E731
                detokenize = lambda toks: f"<{len(list(toks))} toks>"  # noqa: E731
            else:
                assert tok is not None and detok is not None
                messages = [{"role": "user", "content": stage.prompt}]
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                ids = [int(t) for t in tok.encode(text, add_special_tokens=False)]
                prompts = [list(ids) for _ in range(4)]
                probes = [list(ids)]
                reward_fn = exact_integer_reward(detok, stage.gold)
                detokenize = detok

            result = run_grpo(
                base_model=base_model,
                prompts=prompts,
                reward_fn=reward_fn,
                group_size=gs,
                steps=stage.max_steps,
                endpoint=endpoint,
                plan=plan,
                run_dir=writer_root,
                probes=probes,
                probe_every=probe_every,
                detokenize=detokenize,
                early_stop=True,
                early_stop_patience=patience,
                service_client=svc,
                training_client=tc,
                close_clients=False,
            )
            if out.adapter_id is None:
                out.adapter_id = result.adapter_id

            hit_budget = (
                result.early_stop_reason is None
                and result.steps_run >= stage.max_steps
            )
            advance, halt = should_advance(
                result.early_stop_reason,
                hit_budget=hit_budget,
                advance_on=recipe.advance_on,
                stop_queue_on=recipe.stop_queue_on,
                advance_on_budget=recipe.advance_on_budget,
            )
            # last stage cannot advance
            if i >= len(recipe.stages) - 1:
                advance = False

            outcome = StageOutcome(
                stage=stage,
                result=result,
                advanced=advance and not halt,
                queue_halted=halt,
                observe_run_id=stage_run_id,
            )
            out.stages.append(outcome)

            if root is not None:
                queue_log = root / _safe_run_id(f"{prefix}-queue")
                RunMetricsWriter(queue_log).log_event(
                    step=i,
                    event="stage_end",
                    reason=result.early_stop_reason,
                    stage_id=stage.id,
                    steps_run=result.steps_run,
                    advanced=advance and not halt,
                    queue_halted=halt,
                    observe_run_id=stage_run_id,
                    final_reward=(
                        result.mean_reward[-1] if result.mean_reward else None
                    ),
                )

            if halt or not advance:
                break
    finally:
        svc.close()

    return out
