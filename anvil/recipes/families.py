"""Per-family post-training knowledge — the part that isn't knobs.

Model cards tell us *what kind of animal* a base model is (model_card.py);
this registry records *what we know about training that animal*: correct LoRA
target modules per architecture, template quirks, LR caps, RL behavior, and
the footguns we have seen fire in practice (Phi fused projections, Gemma-2
soft-capping, Qwen3 thinking modes, MoE expert explosion, …).

First match wins — keep entries specific → generic. Everything here is a
default: explicit user overrides still win downstream in plan_recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

_ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_ATTENTION_ONLY = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclass(frozen=True, slots=True)
class FamilyKnowledge:
    """Distilled training knowledge for one model family."""

    id: str
    display: str
    match: tuple[str, ...]  # lowercase substrings of the repo id / path
    model_types: tuple[str, ...] = ()  # HF config model_type values
    lora_targets: tuple[str, ...] = _ALL_LINEAR
    template_notes: str = ""
    sft_notes: str = ""
    rl_notes: str = ""
    cautions: tuple[str, ...] = ()
    sft_lr_cap: float | None = None  # hard ceiling for SFT lr on this family


FAMILIES: tuple[FamilyKnowledge, ...] = (
    # --- Qwen ----------------------------------------------------------------
    FamilyKnowledge(
        id="qwen3-moe",
        display="Qwen3 MoE (30B-A3B / 235B-A22B)",
        match=("a3b", "a22b", "qwen3-moe"),
        model_types=("qwen3_moe",),
        lora_targets=_ALL_LINEAR,
        template_notes="Ships {% generation %} markers → native assistant masks. Thinking-mode data like dense Qwen3.",
        sft_notes="Attention + dense-MLP projections only; the expert feed-forwards stay frozen.",
        rl_notes="MoE RL is stretch: expert routing shifts under policy updates. Warm with SFT first, small groups, watch KL.",
        cautions=(
            "NEVER LoRA the router/gate — per-expert LoRA explodes adapter size (many small experts) for no measured benefit.",
            "235B-A22B is far beyond lab scale; 30B-A3B (3B active) is the ceiling on one beefy GPU.",
        ),
        sft_lr_cap=1e-4,
    ),
    FamilyKnowledge(
        id="qwen3",
        display="Qwen3 dense (0.6B–32B, thinking)",
        match=("qwen3",),
        model_types=("qwen3",),
        lora_targets=_ALL_LINEAR,
        template_notes="Ships {% generation %} markers → native assistant masks. Thinking toggled via enable_thinking / /think /no_think.",
        sft_notes="Preserve <think> blocks IN the loss (mask the prompt only) — stripping them collapses the reasoning behavior.",
        rl_notes="GRPO on verifiable rewards is the Qwen3 sweet spot; rollout temp ~1.0. Thinking-length inflation is the main hack — cap completion length.",
        cautions=(
            "Do not mix thinking and non-thinking data without explicit mode tags — both modes degrade.",
        ),
    ),
    FamilyKnowledge(
        id="qwen2-vl",
        display="Qwen2/2.5-VL",
        match=("qwen2.5-vl", "qwen2-vl"),
        model_types=("qwen2_vl", "qwen2_5_vl"),
        lora_targets=_ALL_LINEAR,  # LLM-side; vision tower + merger handled separately
        template_notes="ChatML + generation markers. Image/video tokens must come from the native processor.",
        sft_notes="Freeze the ViT (675M on 7B); LoRA the LLM, train the merger (projector). Canonical freeze ladder applies.",
        rl_notes="Vision RL works (UI agents, visual puzzles) — reward must stay programmatically checkable.",
        cautions=(
            "M-RoPE + (2.5-VL) window attention: keep the native processor; never flatten image grids by hand.",
        ),
    ),
    FamilyKnowledge(
        id="qwen2.5",
        display="Qwen2.5 dense (0.5B–72B, incl. Coder + R1-distills)",
        match=("qwen2.5", "qwen2_5"),
        model_types=("qwen2",),
        lora_targets=_ALL_LINEAR,
        template_notes="Ships {% generation %} markers → native assistant masks.",
        sft_notes="DeepSeek-R1-Distill-Qwen-* are already RL'd — SFT lr ≤5e-5 or you overwrite the reasoning. Coder: include MLP targets, keep seq long.",
        rl_notes="Qwen2.5-Math + GRPO is the reference recipe lineage (DeepSeekMath → R1).",
        cautions=(),
    ),
    # --- Llama ---------------------------------------------------------------
    FamilyKnowledge(
        id="llama3",
        display="Llama 3.x dense (1B–70B)",
        match=("llama-3", "llama_3"),
        model_types=("llama",),
        lora_targets=_ALL_LINEAR,
        template_notes="Stock template usually LACKS {% generation %} markers → the renderer falls back to prefix-diff (exact). Do not hand-mask.",
        sft_notes="8B: lr 1e-4..2e-4, rank 16-64. 3.2 1B/3B are prune+distill — cap lr at 1e-4, they overfit fast.",
        rl_notes="8B Instruct + GRPO(math) is a solid single-GPU citizen. DPO: beta 0.1, LoRA lr ~1e-5.",
        cautions=("405B is not LoRA-on-lab material — hard size gates apply.",),
    ),
    # --- Gemma ---------------------------------------------------------------
    FamilyKnowledge(
        id="gemma3",
        display="Gemma 3 (1B text-only; 4B/12B/27B + SigLIP)",
        match=("gemma-3", "gemma3"),
        lora_targets=_ALL_LINEAR,
        template_notes="<start_of_turn> format; instruct templates in transformers ≥4.50 ship generation markers.",
        sft_notes="QK-norm (no soft-capping) — SDPA is fine for training. Rank 16-32 plenty; lr ≤1e-4.",
        rl_notes="GRPO works; watch verbose-style drift on the 4B.",
        cautions=(
            "4B+ carry a SigLIP vision tower — freeze it (PaliGemma-style). 1B is text-only.",
        ),
    ),
    FamilyKnowledge(
        id="gemma2",
        display="Gemma 2 (2B/9B/27B)",
        match=("gemma-2", "gemma2"),
        lora_targets=_ALL_LINEAR,
        template_notes="<start_of_turn> format; generation markers vary by checkpoint — prefix-diff is the safe default.",
        sft_notes="9B/27B punch above their weight; rank 16-32, lr ≤1e-4.",
        rl_notes="DPO behaves; keep beta ≥0.1 — Gemma-2 collapses fast at low beta.",
        cautions=(
            "Soft-capped attention logits — train with attn_implementation='eager'; FA2/SDPA paths are numerically unsafe for softcap.",
            "Interleaved local:global attention (4096 window) — long-seq SFT only reaches the global layers.",
            "Tied embeddings stay frozen under LoRA (they are also the LM head).",
        ),
    ),
    # --- Phi -----------------------------------------------------------------
    FamilyKnowledge(
        id="phi3",
        display="Phi-3 / Phi-4 / Phi-4-mini (fused projections)",
        match=("phi-3", "phi-4"),
        model_types=("phi3",),
        lora_targets=("qkv_proj", "o_proj", "gate_up_proj", "down_proj"),
        template_notes="ChatML (<|im_start|>); templates ship {% generation %} markers.",
        sft_notes="Synthetic-data trained — conservative lr (5e-5..1e-4). Small curated sets move it fast. Phi-4-mini (3.8B) is the Mia coach base.",
        rl_notes="DPO/GRPO at conservative lr; reasoning-style data suits the Phi-4 lineage.",
        cautions=(
            "FUSED projections: qkv_proj + gate_up_proj. Llama-style targets (q_proj…) match NOTHING — the zero-trainable-params gate fires.",
        ),
        sft_lr_cap=1e-4,
    ),
    # --- Mistral / Mixtral ---------------------------------------------------
    FamilyKnowledge(
        id="mistral",
        display="Mistral 7B (v0.1–v0.3)",
        match=("mistral",),
        model_types=("mistral",),
        lora_targets=_ALL_LINEAR,
        template_notes="Pre-v0.3 template has NO system role — fold system into the first user turn. v0.3 adds system + tool calls.",
        sft_notes="Standard dense recipe; v0.3 instruct is the better base.",
        rl_notes="GRPO/DPO unremarkable — behaves like a solid 7B.",
        cautions=(
            "v0.1 has a 4k sliding-window attention — long-context SFT misbehaves; v0.2+ is full 32k.",
        ),
    ),
    FamilyKnowledge(
        id="mixtral",
        display="Mixtral 8x7B / 8x22B (MoE)",
        match=("mixtral",),
        model_types=("mixtral",),
        lora_targets=_ATTENTION_ONLY,
        template_notes="[INST] blocks; no system role (v0.1).",
        sft_notes="Attention-only LoRA is the pragmatic recipe — 8 experts make per-expert LoRA 8× the adapter for no measured win.",
        rl_notes="RL on MoE is stretch (see qwen3-moe notes).",
        cautions=(
            "v0.1 sliding window; 8x22B needs multi-GPU or QLoRA even for LoRA SFT.",
        ),
        sft_lr_cap=5e-5,
    ),
    # --- Smol (smoke-test citizens) ------------------------------------------
    FamilyKnowledge(
        id="smollm",
        display="SmolLM2/3 (135M–3B)",
        match=("smollm",),
        lora_targets=_ALL_LINEAR,
        template_notes="ChatML; SmolLM3 ships generation markers.",
        sft_notes="Best pipeline-smoke citizens: real weights, hidden ≥576 (LoRA actually learns, unlike random tiny models).",
        rl_notes="SmolLM3 3B GRPOs arithmetic/format rewards on one GPU — good first RL loop.",
        cautions=("SmolLM3 uses NoPE hybrid layers — do not assume RoPE everywhere.",),
    ),
    FamilyKnowledge(
        id="smolvlm",
        display="SmolVLM (256M/500M/2.2B)",
        match=("smolvlm",),
        lora_targets=_ALL_LINEAR,
        template_notes="ChatML; image tokens via native processor.",
        sft_notes="Freeze SigLIP; pixel-shuffle projector trains with the LM. 256M/500M iterate in minutes on one GPU.",
        rl_notes="Edge-class VLM RL is possible but SFT usually suffices at this size.",
        cautions=(),
    ),
    FamilyKnowledge(
        id="paligemma",
        display="PaliGemma (3B, SigLIP + Gemma-2B)",
        match=("paligemma",),
        lora_targets=_ALL_LINEAR,
        template_notes="Prefix-LM: image tokens are a literal prefix of every example.",
        sft_notes="Freeze SigLIP; train Gemma + (gently) the projector. Strong at caption/VQA/detection-style short answers.",
        rl_notes="Not an RL citizen — use classifier/rubric patterns instead.",
        cautions=(
            "Image tokens are a PREFIX — never include them in the loss (mask everything up to the answer).",
        ),
    ),
    # --- DeepSeek -------------------------------------------------------------
    FamilyKnowledge(
        id="deepseek",
        display="DeepSeek V2/V3/R1 (MLA + heavy MoE)",
        match=("deepseek-v", "deepseek-moe", "deepseek-r1"),
        lora_targets=_ATTENTION_ONLY,  # MLA names differ; see cautions
        template_notes="ChatML-ish; R1 emits <think> blocks by default.",
        sft_notes="V2/V3/R1-671B are far beyond lab scale — hard block, not stretch. R1-DISTILLS (Qwen/Llama bases) are fine: see those families.",
        rl_notes="R1's own recipe (GRPO, KL→0, format+accuracy rewards) is the reference, but it ran on clusters.",
        cautions=(
            "MLA (multi-head latent attention) uses kv_a/kv_b/q_a projection names — standard q/k/v targets miss the attention entirely.",
        ),
        sft_lr_cap=5e-5,
    ),
)


def lookup_family(
    base_model: str,
    *,
    model_type: str | None = None,
    architectures: tuple[str, ...] = (),
) -> FamilyKnowledge | None:
    """First matching family for a repo id / path (config fields refine the match)."""
    name = base_model.lower().replace("_", "-")
    mt = (model_type or "").lower()
    arch_l = tuple(a.lower() for a in architectures)
    for fam in FAMILIES:
        if mt and mt in fam.model_types:
            return fam
        if any(m in name for m in fam.match):
            return fam
        if any(m.replace("-", "") in a.replace("-", "") for m in fam.match for a in arch_l):
            return fam
    return None
