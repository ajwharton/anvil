"""Named loss registry — no arbitrary remote code execution in v0.

Workers look up losses by string id. Plugins register at process start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from anvil.protocol.types import LossFn

# Placeholder signature: real implementation will take model outputs + Datum tensors.
LossComputeFn = Callable[..., float]


@dataclass(frozen=True, slots=True)
class LossSpec:
    name: str
    description: str
    required_inputs: tuple[str, ...]
    family: str  # "sft" | "rl" | "preference"


_REGISTRY: dict[str, LossSpec] = {}


def register_loss(spec: LossSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"loss already registered: {spec.name}")
    _REGISTRY[spec.name] = spec


def get_loss(name: str | LossFn) -> LossSpec:
    key = name.value if isinstance(name, LossFn) else name
    try:
        return _REGISTRY[key]
    except KeyError as e:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown loss {key!r}; known: {known}") from e


def list_losses() -> Mapping[str, LossSpec]:
    return dict(_REGISTRY)


def _bootstrap_builtin() -> None:
    if _REGISTRY:
        return
    builtins = [
        LossSpec(
            name=LossFn.CROSS_ENTROPY.value,
            description="Token-level cross-entropy (SFT / NLL).",
            required_inputs=("target_tokens", "weights"),
            family="sft",
        ),
        LossSpec(
            name=LossFn.IMPORTANCE_SAMPLING.value,
            description="On-policy IS policy gradient with advantages.",
            required_inputs=("target_tokens", "weights", "logprobs", "advantages"),
            family="rl",
        ),
        LossSpec(
            name=LossFn.PPO.value,
            description="PPO-clip style surrogate (worker-side clip).",
            required_inputs=("target_tokens", "weights", "logprobs", "advantages"),
            family="rl",
        ),
        LossSpec(
            name=LossFn.CISPO.value,
            description="CISPO-style RL loss (named stub).",
            required_inputs=("target_tokens", "weights", "logprobs", "advantages"),
            family="rl",
        ),
        LossSpec(
            name=LossFn.DRO.value,
            description="DRO-style RL loss (named stub).",
            required_inputs=("target_tokens", "weights", "logprobs", "advantages"),
            family="rl",
        ),
        LossSpec(
            name=LossFn.DPO.value,
            description="Direct preference optimization (paired data).",
            required_inputs=("target_tokens", "weights"),
            family="preference",
        ),
    ]
    for spec in builtins:
        _REGISTRY[spec.name] = spec


_bootstrap_builtin()
