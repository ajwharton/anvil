"""Derive architecture facts from Hugging Face model cards + config.json.

The overall recipe *shape* should come from the published model, not guesswork:
``architectures``, ``model_type``, ``vision_config``, param count, pipeline tag,
and card prose (multimodal / agentic / edge). Fine-tuning still needs our data
and loops — the card tells us *what kind of animal* we are adapting.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from anvil.recipes.profiles import ModelShape

HF_API = "https://huggingface.co/api/models/"
HF_RAW = "https://huggingface.co/{repo}/raw/main/{path}"


@dataclass(frozen=True, slots=True)
class ModelCardFacts:
    """Normalized view of a base model for recipe planning."""

    repo_id: str
    source: str  # local_path | hf_api | name_only
    model_type: str | None = None
    architectures: tuple[str, ...] = ()
    pipeline_tag: str | None = None
    tags: tuple[str, ...] = ()
    param_count: int | None = None  # total params if known
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    max_position_embeddings: int | None = None
    torch_dtype: str | None = None
    has_vision: bool = False
    has_video: bool = False
    is_moe: bool = False
    vision_hidden_size: int | None = None
    vision_depth: int | None = None
    image_token_id: int | None = None
    license: str | None = None
    card_summary: str | None = None
    local_path: str | None = None
    shape: ModelShape = ModelShape.UNKNOWN
    shape_confidence: str = "low"  # low | medium | high
    evidence: tuple[str, ...] = ()
    peft_target_modules: tuple[str, ...] = ()
    recommended_patterns: tuple[str, ...] = ()

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        d["shape"] = self.shape.value
        d["param_b"] = round(self.param_count / 1e9, 2) if self.param_count else None
        return d


def inspect_base_model(
    base_model: str,
    *,
    models_root: str | Path | None = None,
    fetch_remote: bool = True,
    timeout: float = 20.0,
) -> ModelCardFacts:
    """Inspect a base model id or local path → ModelCardFacts.

    Resolution order:
      1. Local directory (path or ``{models_root}/{basename}``)
      2. Hugging Face API + raw config.json (if ``fetch_remote``)
      3. Name-only heuristics (last resort)
    """
    evidence: list[str] = []
    local = _resolve_local(base_model, models_root)
    api: dict[str, Any] = {}
    cfg: dict[str, Any] = {}
    readme = ""
    source = "name_only"
    local_path = None
    repo_id = base_model

    if local is not None:
        source = "local_path"
        local_path = str(local)
        evidence.append(f"local snapshot {local}")
        cfg = _read_json(local / "config.json") or {}
        readme = _read_text(local / "README.md") or ""
        # Prefer HF-style repo id from folder name if base looks like a path
        if "/" not in base_model.replace("\\", "/").strip("/").split("/")[0] or base_model.startswith("/"):
            repo_id = local.name
        if (local / "model.safetensors.index.json").is_file():
            evidence.append("safetensors index present")

    if fetch_remote and _looks_like_hf_id(base_model):
        repo_id = base_model
        try:
            api = _http_json(f"{HF_API}{base_model}", timeout=timeout) or {}
            if source == "name_only":
                source = "hf_api"
            evidence.append("hf api model metadata")
            if not cfg:
                cfg = _http_json(
                    HF_RAW.format(repo=base_model, path="config.json"),
                    timeout=timeout,
                ) or {}
                if cfg:
                    evidence.append("hf raw config.json")
            if not readme:
                try:
                    readme = _http_text(
                        HF_RAW.format(repo=base_model, path="README.md"),
                        timeout=timeout,
                    ) or ""
                    if readme:
                        evidence.append("hf model card README")
                except Exception:
                    pass
        except Exception as e:
            evidence.append(f"hf fetch skipped: {type(e).__name__}")

    facts = _facts_from_parts(
        repo_id=repo_id,
        source=source,
        cfg=cfg,
        api=api,
        readme=readme,
        local_path=local_path,
        evidence=evidence,
    )
    return facts


def _facts_from_parts(
    *,
    repo_id: str,
    source: str,
    cfg: dict[str, Any],
    api: dict[str, Any],
    readme: str,
    local_path: str | None,
    evidence: list[str],
) -> ModelCardFacts:
    arch = tuple(cfg.get("architectures") or (api.get("config") or {}).get("architectures") or ())
    model_type = cfg.get("model_type") or (api.get("config") or {}).get("model_type")
    pipeline = api.get("pipeline_tag") or _frontmatter_val(readme, "pipeline_tag")
    tags = tuple(api.get("tags") or [])
    if not tags:
        tags = tuple(_frontmatter_list(readme, "tags") or [])

    vision_cfg = cfg.get("vision_config") or {}
    name_l = repo_id.lower()
    has_vision = bool(vision_cfg) or bool(cfg.get("image_token_id")) or (
        pipeline == "image-text-to-text"
    )
    has_vision = has_vision or "vl" in (model_type or "").lower()
    has_vision = has_vision or any("vl" in a.lower() for a in arch)
    # name cue when card/config not fetched yet
    has_vision = has_vision or any(m in name_l for m in ("-vl", "vl-", "vision", "vlm"))
    has_video = bool(cfg.get("video_token_id")) or "video" in (readme[:2000].lower())
    is_moe = bool(cfg.get("num_experts") or cfg.get("num_local_experts")) or any(
        "moe" in t.lower() for t in tags
    ) or "moe" in (model_type or "").lower()

    params = None
    st = api.get("safetensors") or {}
    if isinstance(st, dict) and st.get("total"):
        params = int(st["total"])
        evidence.append(f"param_count from safetensors.total={params}")
    elif isinstance(st.get("parameters"), dict):
        params = int(sum(st["parameters"].values()))

    # Rough param estimate from config if needed
    if params is None and cfg.get("hidden_size") and cfg.get("num_hidden_layers"):
        # very rough — not used for exact size class alone
        pass

    shape, conf, shape_ev = _shape_from_card(
        repo_id=repo_id,
        model_type=model_type,
        arch=arch,
        has_vision=has_vision,
        has_vision_config=bool(vision_cfg),
        is_moe=is_moe,
        params=params,
        pipeline=pipeline,
        readme=readme,
    )
    evidence.extend(shape_ev)

    peft_targets = _default_peft_targets(model_type, has_vision, arch)
    patterns = _patterns_for_shape(shape, has_vision, readme)

    summary = _card_blurb(readme)
    license_ = None
    card = api.get("cardData") or {}
    if isinstance(card, dict):
        license_ = card.get("license_name") or card.get("license")
    license_ = license_ or _frontmatter_val(readme, "license_name") or _frontmatter_val(readme, "license")

    return ModelCardFacts(
        repo_id=repo_id,
        source=source,
        model_type=model_type,
        architectures=arch,
        pipeline_tag=pipeline,
        tags=tags,
        param_count=params,
        hidden_size=_as_int(cfg.get("hidden_size")),
        num_hidden_layers=_as_int(cfg.get("num_hidden_layers")),
        num_attention_heads=_as_int(cfg.get("num_attention_heads")),
        max_position_embeddings=_as_int(cfg.get("max_position_embeddings")),
        torch_dtype=cfg.get("torch_dtype") or (api.get("config") or {}).get("torch_dtype"),
        has_vision=has_vision,
        has_video=has_video,
        is_moe=is_moe,
        vision_hidden_size=_as_int(vision_cfg.get("hidden_size")),
        vision_depth=_as_int(vision_cfg.get("depth")),
        image_token_id=_as_int(cfg.get("image_token_id")),
        license=license_,
        card_summary=summary,
        local_path=local_path,
        shape=shape,
        shape_confidence=conf,
        evidence=tuple(evidence),
        peft_target_modules=peft_targets,
        recommended_patterns=patterns,
    )


def _shape_from_card(
    *,
    repo_id: str,
    model_type: str | None,
    arch: tuple[str, ...],
    has_vision: bool,
    has_vision_config: bool = False,
    is_moe: bool,
    params: int | None,
    pipeline: str | None,
    readme: str,
) -> tuple[ModelShape, str, list[str]]:
    ev: list[str] = []
    name = repo_id.lower()
    param_b = (params / 1e9) if params else None

    if is_moe and not has_vision:
        ev.append("moe signals in config/tags")
        return ModelShape.MOE_LM, "high" if params or model_type else "medium", ev

    if has_vision or pipeline == "image-text-to-text":
        if has_vision_config or pipeline == "image-text-to-text" or model_type:
            ev.append("vision from config/pipeline/model_type")
        else:
            ev.append("vision from model id / card name")
        small = False
        if param_b is not None:
            small = param_b <= 4.5
            ev.append(f"params≈{param_b:.2f}B")
        else:
            # name size cue only when no param count (normalize 2.5 → 25 so 3b still matches)
            compact = re.sub(r"(\d)\.(\d)", r"\1\2", name)
            small = bool(re.search(r"(^|[^0-9])([123])b([^a-z0-9]|$)", compact))
            if small:
                ev.append("name size cue ≤3B")
        # card prose: Qwen markets 3B as edge
        if "edge" in readme[:3000].lower() or "edge ai" in readme[:3000].lower():
            small = True
            ev.append("card mentions edge")
        if small:
            return ModelShape.EDGE_STUDENT, ("high" if param_b else "medium"), ev
        return ModelShape.DENSE_VLM, "high" if (model_type or arch or param_b) else "medium", ev

    if param_b is not None and param_b <= 4.5:
        ev.append(f"dense LM params≈{param_b:.2f}B")
        return ModelShape.DENSE_LM, "high", ev

    if model_type or arch:
        ev.append(f"model_type={model_type!r} arch={arch[:1]}")
        return ModelShape.DENSE_LM, "medium", ev

    # name fallback
    if any(x in name for x in ("-vl", "vl-", "vision")):
        return ModelShape.DENSE_VLM, "low", ["name-only vision cue"]
    return ModelShape.UNKNOWN, "low", ["insufficient card/config"]


def _default_peft_targets(
    model_type: str | None,
    has_vision: bool,
    arch: tuple[str, ...],
) -> tuple[str, ...]:
    """Research-common LoRA targets (TRL / HF cookbook style).

    Language: attention projections. Vision encoder omitted by default
    (freeze until data proves need) — matches Anvil + common VLM recipes.
    """
    # HF TRL VLM cookbook often uses q_proj, v_proj; broader recipes use all-linear attention
    lang = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    mt = (model_type or "").lower()
    if "qwen2" in mt or "qwen3" in mt or any("Qwen" in a for a in arch):
        return lang
    if has_vision:
        return ("q_proj", "v_proj")  # conservative cookbook default
    return lang


def _patterns_for_shape(
    shape: ModelShape, has_vision: bool, readme: str
) -> tuple[str, ...]:
    agentic = "agent" in readme[:4000].lower()
    if shape in {ModelShape.DENSE_VLM, ModelShape.EDGE_STUDENT} or has_vision:
        out = ["vlm_sft", "vlm_classifier", "robot_offline", "sft_chat"]
        if agentic:
            out = ["vlm_sft", "robot_offline", "vlm_classifier", "rl_verifiable"]
        return tuple(out)
    if shape == ModelShape.MOE_LM:
        return ("sft_chat", "rl_verifiable", "preference_dpo")
    return ("sft_chat", "rl_verifiable", "preference_dpo")


def _card_blurb(readme: str) -> str | None:
    if not readme:
        return None
    # drop yaml frontmatter
    body = readme
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) >= 3:
            body = parts[2]
    lines = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<") or line.startswith("!["):
            if line.startswith("#") and lines:
                break
            continue
        lines.append(line)
        if sum(len(x) for x in lines) > 280:
            break
    if not lines:
        return None
    return " ".join(lines)[:400]


def _resolve_local(base_model: str, models_root: str | Path | None) -> Path | None:
    p = Path(base_model).expanduser()
    if p.is_dir() and (p / "config.json").is_file():
        return p
    root = Path(models_root or os.environ.get("ANVIL_MODELS_ROOT", "/mnt/data/models"))
    candidates = [
        root / base_model,
        root / base_model.split("/")[-1],
    ]
    for c in candidates:
        if c.is_dir() and (c / "config.json").is_file():
            return c
    return None


def _looks_like_hf_id(s: str) -> bool:
    if s.startswith("/") or s.startswith("."):
        return False
    parts = s.split("/")
    return len(parts) == 2 and all(parts)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _http_json(url: str, *, timeout: float) -> dict[str, Any] | None:
    text = _http_text(url, timeout=timeout)
    if not text:
        return None
    return json.loads(text)


def _http_text(url: str, *, timeout: float) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": "anvil-recipes/0.0.2"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — intentional HF fetch
        return resp.read().decode("utf-8", errors="replace")


def _as_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _frontmatter_val(md: str, key: str) -> str | None:
    if not md.startswith("---"):
        return None
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", md, re.M)
    if not m:
        return None
    return m.group(1).strip().strip("\"'")


def _frontmatter_list(md: str, key: str) -> list[str] | None:
    if not md.startswith("---"):
        return None
    # simple "- item" list under key
    m = re.search(rf"^{re.escape(key)}:\s*\n((?:\s*-\s+.+\n)+)", md, re.M)
    if not m:
        return None
    return re.findall(r"-\s+(.+)", m.group(1))
