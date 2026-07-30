"""Vision-aware verifiable rewards for on-policy GRPO (Phase 4.B).

Rewards score **detokenized completion text** (and optional gold fields).
Image refs are prompt context for the sampler; they do not enter the reward
tensor directly in v0 (no learned vision RM). Rubrics stay explicit and
auditable — same live-sufficiency contract as text GRPO.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from anvil.recipes.grpo import RewardFn

DetokenizeFn = Callable[[Sequence[int]], str]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def keyword_caption_reward(
    *,
    required: Sequence[str],
    any_of: Sequence[str] = (),
    detokenize: DetokenizeFn | None = None,
    case_insensitive: bool = True,
) -> RewardFn:
    """1.0 if all ``required`` substrings appear (and any of ``any_of`` if set).

    Use for scene captions: gold keywords from SSD labels / human notes.
    Partial credit: fraction of required hits when ``any_of`` is empty.
    """
    req = [str(x) for x in required if str(x).strip()]
    opt = [str(x) for x in any_of if str(x).strip()]

    def reward_fn(text: str, tokens: Sequence[int]) -> float:
        body = text if text else (detokenize(tokens) if detokenize else "")
        hay = _norm(body) if case_insensitive else body
        if not req and not opt:
            return 0.0
        hits = 0
        for k in req:
            needle = _norm(k) if case_insensitive else k
            if needle and needle in hay:
                hits += 1
        if req:
            score = hits / len(req)
        else:
            score = 0.0
        if opt:
            if any((_norm(k) if case_insensitive else k) in hay for k in opt):
                score = min(1.0, score + 0.25) if req else 1.0
            elif not req:
                score = 0.0
        return float(score)

    return reward_fn


def exact_phrase_reward(
    gold: str,
    *,
    detokenize: DetokenizeFn | None = None,
) -> RewardFn:
    """1.0 iff normalized completion equals normalized gold (or contains it)."""
    g = _norm(gold)

    def reward_fn(text: str, tokens: Sequence[int]) -> float:
        body = text if text else (detokenize(tokens) if detokenize else "")
        b = _norm(body)
        if not g:
            return 0.0
        if b == g or g in b:
            return 1.0
        return 0.0

    return reward_fn


def action_bin_overlap_reward(
    gold_bins: Sequence[int] | str,
    *,
    detokenize: DetokenizeFn | None = None,
) -> RewardFn:
    """Fraction of space-separated bin tokens that match gold (OpenVLA-style)."""
    if isinstance(gold_bins, str):
        gold = [int(x) for x in gold_bins.split() if x.lstrip("-").isdigit()]
    else:
        gold = [int(x) for x in gold_bins]

    def reward_fn(text: str, tokens: Sequence[int]) -> float:
        body = text if text else (detokenize(tokens) if detokenize else "")
        pred = [int(x) for x in body.split() if x.lstrip("-").isdigit()]
        if not gold:
            return 0.0
        n = min(len(gold), len(pred))
        if n == 0:
            return 0.0
        hits = sum(1 for i in range(n) if pred[i] == gold[i])
        # length penalty if completion much shorter
        if len(pred) < len(gold):
            return hits / len(gold)
        return hits / len(gold)

    return reward_fn


@dataclass(frozen=True, slots=True)
class RubricCriterion:
    """One graded criterion (keyword or phrase)."""

    name: str
    weight: float = 1.0
    required_keywords: tuple[str, ...] = ()
    forbidden_keywords: tuple[str, ...] = ()


def rubric_reward(
    criteria: Sequence[RubricCriterion],
    *,
    detokenize: DetokenizeFn | None = None,
) -> RewardFn:
    """Weighted sum of criterion scores in [0, 1]."""
    crits = list(criteria)
    total_w = sum(max(0.0, c.weight) for c in crits) or 1.0

    def reward_fn(text: str, tokens: Sequence[int]) -> float:
        body = text if text else (detokenize(tokens) if detokenize else "")
        hay = _norm(body)
        acc = 0.0
        for c in crits:
            w = max(0.0, c.weight)
            if any(_norm(f) in hay for f in c.forbidden_keywords if f):
                score = 0.0
            elif not c.required_keywords:
                score = 1.0 if hay else 0.0
            else:
                hits = sum(1 for k in c.required_keywords if _norm(k) in hay)
                score = hits / len(c.required_keywords)
            acc += w * score
        return float(acc / total_w)

    return reward_fn


def toy_detokenize(tokens: Sequence[int]) -> str:
    """CI detokenizer: map small ids to fixed vocabulary (kitchen scene words)."""
    vocab = {
        0: "unsure",
        1: "kitchen",
        2: "chair",
        3: "table",
        4: "cabinet",
        5: "stove",
        6: "person",
        7: "dog",
        8: "room",
        9: "hallway",
    }
    parts = [vocab.get(int(t) % 10, str(int(t) % 10)) for t in tokens]
    return " ".join(parts)
