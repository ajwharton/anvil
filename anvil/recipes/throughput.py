"""Throughput defaults per job shape (Expert-v2 / P.Ops).

Opinionated starting knobs for dense text, VLM, and on-policy RL so operators
do not invent batch/seq/checkpoint policy each run. Recipes still accept
overrides; these are the documented defaults for scale jobs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from anvil.recipes.profiles import JobPattern, ModelShape


@dataclass(frozen=True, slots=True)
class ThroughputDefaults:
    """Per-shape training throughput profile."""

    shape: str
    pattern: str
    # Micro-batch / packing
    batch_size: int
    max_seq_len: int
    # Optimizer / adapter
    rank: int
    learning_rate: float
    # Long-job ops
    checkpoint_every: int
    # Rough wall-clock guidance (lab Spark-class single GPU; not a guarantee)
    notes: str
    tokens_per_step_hint: int | None = None
    steps_per_hour_hint: int | None = None

    def as_overrides(self) -> dict[str, Any]:
        """Knobs suitable for ``plan_recipe(..., overrides=...)`` / run_*."""
        return {
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "rank": self.rank,
            "learning_rate": self.learning_rate,
        }

    def to_public(self) -> dict[str, Any]:
        return asdict(self)


# Lab-oriented defaults (single GPU, bf16, LoRA). Tune on forge and promote
# into the personal recipe book when a family stabilizes.
_PROFILES: dict[tuple[str, str], ThroughputDefaults] = {
    ("dense_lm", "sft_chat"): ThroughputDefaults(
        shape="dense_lm",
        pattern="sft_chat",
        batch_size=4,
        max_seq_len=1024,
        rank=32,
        learning_rate=1e-4,
        checkpoint_every=50,
        notes="Text SFT 1.5B–7B: microbatch 4; raise batch via grad accum on multi-GPU later.",
        tokens_per_step_hint=4 * 1024,
        steps_per_hour_hint=800,
    ),
    ("dense_vlm", "vlm_sft"): ThroughputDefaults(
        shape="dense_vlm",
        pattern="vlm_sft",
        batch_size=1,
        max_seq_len=1024,
        rank=32,
        learning_rate=1e-4,
        checkpoint_every=25,
        notes=(
            "VLM 3B on Spark: batch 1 with pixels; freeze encoder; "
            "checkpoint more often — wall time dominated by vision forward."
        ),
        tokens_per_step_hint=1024,
        steps_per_hour_hint=120,
    ),
    ("edge_student", "vlm_sft"): ThroughputDefaults(
        shape="edge_student",
        pattern="vlm_sft",
        batch_size=2,
        max_seq_len=512,
        rank=16,
        learning_rate=2e-4,
        checkpoint_every=50,
        notes="Edge student VLM: smaller rank/seq; still freeze vision by default.",
        tokens_per_step_hint=2 * 512,
        steps_per_hour_hint=200,
    ),
    ("dense_lm", "rl_verifiable"): ThroughputDefaults(
        shape="dense_lm",
        pattern="rl_verifiable",
        batch_size=2,
        max_seq_len=1024,
        rank=16,
        learning_rate=5e-5,
        checkpoint_every=20,
        notes=(
            "GRPO: sample dominates wall-clock; use vLLM sample worker for multi-hour; "
            "group_size 8 lab / 64 paper-scale."
        ),
        tokens_per_step_hint=2 * 64,
        steps_per_hour_hint=60,
    ),
    ("dense_lm", "preference_dpo"): ThroughputDefaults(
        shape="dense_lm",
        pattern="preference_dpo",
        batch_size=2,
        max_seq_len=1024,
        rank=16,
        learning_rate=5e-5,
        checkpoint_every=50,
        notes="DPO: modest LR; monitor length_bias; one epoch preferred over huge step counts.",
        tokens_per_step_hint=2 * 1024,
        steps_per_hour_hint=400,
    ),
}


def throughput_defaults(
    *,
    shape: ModelShape | str = ModelShape.DENSE_LM,
    pattern: JobPattern | str = JobPattern.SFT_CHAT,
) -> ThroughputDefaults:
    """Return defaults for a shape×pattern pair (falls back to dense_lm sft_chat)."""
    s = shape.value if isinstance(shape, ModelShape) else str(shape)
    p = pattern.value if isinstance(pattern, JobPattern) else str(pattern)
    key = (s, p)
    if key in _PROFILES:
        return _PROFILES[key]
    # Soft fallbacks
    if p in {"vlm_sft", "vlm_classifier", "robot_offline"}:
        return _PROFILES.get(("dense_vlm", "vlm_sft"), _PROFILES[("dense_lm", "sft_chat")])
    if p in {"rl_verifiable", "grpo"}:
        return _PROFILES[("dense_lm", "rl_verifiable")]
    if p in {"preference_dpo", "dpo"}:
        return _PROFILES[("dense_lm", "preference_dpo")]
    return _PROFILES[("dense_lm", "sft_chat")]


def list_throughput_profiles() -> list[ThroughputDefaults]:
    return list(_PROFILES.values())


__all__ = [
    "ThroughputDefaults",
    "list_throughput_profiles",
    "throughput_defaults",
]
