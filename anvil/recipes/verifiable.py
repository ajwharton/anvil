"""Verifiable reward helpers for GRPO / productized RL demos.

Stock ``run_grpo`` only passes *completion* tokens into ``reward_fn`` (no prompt
identity). Rewards that check against a *set* of gold answers are therefore
too loose: a model that always emits ``4`` scores 1.0 on a multi-problem set
that happens to include 4. Prefer one gold per run, or extend ``run_grpo`` to
pass prompt context (later).
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

# Matches run_grpo.RewardFn: (text, tokens) -> float
RewardFn = Callable[[str, Sequence[int]], float]
DetokenizeFn = Callable[[Sequence[int]], str]


def extract_first_int(text: str) -> str | None:
    """First integer substring in ``text``, or None."""
    m = re.search(r"-?\d+", text)
    return m.group(0) if m else None


def exact_integer_reward(
    detokenize: DetokenizeFn,
    gold: str,
) -> RewardFn:
    """1.0 iff the first integer in the completion equals ``gold``, else 0.0."""

    gold = str(gold).strip()

    def reward_fn(_text: str, tokens: Sequence[int]) -> float:
        text = detokenize(tokens)
        got = extract_first_int(text)
        if got is None:
            return 0.0
        return 1.0 if got == gold else 0.0

    return reward_fn


def multi_gold_membership_reward(
    detokenize: DetokenizeFn,
    golds: Sequence[str],
) -> RewardFn:
    """Loose scorer: first int in *any* gold set → 1.0.

    **Not recommended for multi-prompt GRPO** — documented so tests can prove
    the failure mode (constant high reward / advantage collapse).
    """
    gold_set = {str(g).strip() for g in golds}

    def reward_fn(_text: str, tokens: Sequence[int]) -> float:
        got = extract_first_int(detokenize(tokens))
        if got is None:
            return 0.0
        return 1.0 if got in gold_set else 0.0

    return reward_fn


def detokenize_via_tokenizer(tokenizer: object) -> DetokenizeFn:
    """Build a detokenize callable from a HF tokenizer-like object."""

    def detokenize(tokens: Sequence[int]) -> str:
        return tokenizer.decode(list(tokens), skip_special_tokens=True)  # type: ignore[attr-defined]

    return detokenize


# Calibrated on forge 2026-07-26 against Qwen2.5-1.5B-Instruct (n=16, T=1.0).
# Prefer *partial* hit rate so GRPO groups have reward variance (not all-0 / all-1).
#   15*8+7 → ~10/16  |  9*9+9 → ~7/16  |  56+78-19 → ~8/16
# Avoid pure 2+2 / 17+28 (already 16/16 — flat reward, advantage collapse).
DEFAULT_HARD_PROBLEMS: tuple[tuple[str, str], ...] = (
    (
        "What is 15*8+7? Reply with only the number, no words.",
        "127",
    ),
    (
        "What is 9*9+9? Reply with only the number, no words.",
        "90",
    ),
    (
        "What is 56+78-19? Reply with only the number, no words.",
        "115",
    ),
)
