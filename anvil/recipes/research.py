"""Research-backed recipe defaults (public sources, not proprietary runs).

We still fine-tune on our data; these encode *community-stable shapes*:
Tinker verb set, TRL/HF VLM cookbooks, GRPO-style on-policy RL, LoRA PEFT.

Citations are pointers for humans — not bit-compat claims with any vendor API.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Source:
    key: str
    title: str
    url: str
    takeaway: str


SOURCES: tuple[Source, ...] = (
    Source(
        key="tinker_verbs",
        title="Tinker API shape (Thinking Machines docs)",
        url="https://tinker-docs.thinkingmachines.ai/tinker/quickstart/",
        takeaway="SFT = forward_backward(cross_entropy)+optim_step; RL = sample → reward → IS/PPO loss.",
    ),
    Source(
        key="hf_vlm_trl",
        title="HF cookbook: Fine-tuning a VLM with TRL",
        url="https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl",
        takeaway="LoRA on language projections (often q/v); modest r (8–16); PEFT + SFTConfig.",
    ),
    Source(
        key="qwen25vl_card",
        title="Qwen2.5-VL model card (architecture + edge 3B)",
        url="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct",
        takeaway="image-text-to-text; dynamic res; agentic/tool use; 3B positioned for edge.",
    ),
    Source(
        key="grpo",
        title="Group Relative Policy Optimization (DeepSeekMath / open GRPO recipes)",
        url="https://huggingface.co/docs/trl/grpo_trainer",
        takeaway="Group-normalized advantages; no value head; natural fit for verifiable rewards.",
    ),
    Source(
        key="lora_posttrain",
        title="LoRA for post-training (TML research + PEFT practice)",
        url="https://thinkingmachines.ai/blog/lora/",
        takeaway="LoRA can match full FT for many post-train loads; small artifacts enable hot-swap/export.",
    ),
    Source(
        key="anderson_anatomy",
        title="Anatomy of a Modern Finetuning API (Anderson)",
        url="https://benanderson.work/blog/anatomy-of-finetuning-api/",
        takeaway="Low-level verbs + LoRA multi-tenancy explain the product shape.",
    ),
)


# Pattern → which research keys inform defaults
PATTERN_SOURCES: dict[str, tuple[str, ...]] = {
    "sft_chat": ("tinker_verbs", "lora_posttrain"),
    "vlm_sft": ("hf_vlm_trl", "qwen25vl_card", "lora_posttrain", "tinker_verbs"),
    "vlm_classifier": ("hf_vlm_trl", "qwen25vl_card"),
    "rl_verifiable": ("tinker_verbs", "grpo", "lora_posttrain"),
    "preference_dpo": ("tinker_verbs", "lora_posttrain"),
    "robot_offline": ("qwen25vl_card", "hf_vlm_trl", "lora_posttrain"),
}


def sources_for_pattern(pattern: str) -> list[dict[str, str]]:
    keys = PATTERN_SOURCES.get(pattern, ())
    by_key = {s.key: s for s in SOURCES}
    out = []
    for k in keys:
        s = by_key.get(k)
        if s:
            out.append(
                {
                    "key": s.key,
                    "title": s.title,
                    "url": s.url,
                    "takeaway": s.takeaway,
                }
            )
    return out


def research_notes(pattern: str) -> tuple[str, ...]:
    return tuple(f"{s['title']}: {s['takeaway']}" for s in sources_for_pattern(pattern))
