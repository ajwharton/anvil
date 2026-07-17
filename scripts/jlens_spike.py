#!/usr/bin/env python3
"""J0 forge spike — fit/apply Jacobian lens on a small dense model (math order gate).

Phase 2.5 last gate (see docs/roadmap.md, docs/spikes/jlens-math.md):

  Reproduce "intermediate steps light up in order" on multi-step arithmetic
  before any permanent anvil-web J-Lens panel lands.

**Where to run:** forge (or hammer) with GPU + lab model path. Not on the Mac
for fit/apply of multi‑GB bases. Mac may run ``--check`` / dry docs only.

Examples (on forge)::

  # deps + paths
  python scripts/jlens_spike.py check \\
    --model-path /mnt/data/models/Qwen2.5-1.5B-Instruct

  # fit once (writes lens under /mnt/data/models/lenses/<name>/)
  python scripts/jlens_spike.py fit \\
    --model-path /mnt/data/models/Qwen2.5-1.5B-Instruct \\
    --device cuda

  # apply math probes + order score → JSON + markdown stub
  python scripts/jlens_spike.py apply \\
    --model-path /mnt/data/models/Qwen2.5-1.5B-Instruct \\
    --lens-path /mnt/data/models/lenses/Qwen2.5-1.5B-Instruct/jacobian_lens.pt \\
    --out /mnt/data/anvil-runs/jlens-spike-$(date +%Y%m%d)

  # fit if missing, then apply
  python scripts/jlens_spike.py all \\
    --model-path /mnt/data/models/Qwen2.5-1.5B-Instruct \\
    --out /mnt/data/anvil-runs/jlens-spike

Requires optional deps (lab venv)::

  pip install 'git+https://github.com/anthropics/jacobian-lens.git' torch transformers
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Math probe set — intermediate concepts should light before the answer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MathProbe:
    """One multi-step prompt with ordered concept stages (strings to match in tops)."""

    id: str
    prompt: str
    # stages earliest → latest; each stage is a list of acceptable surface forms
    stages: tuple[tuple[str, ...], ...]
    answer: str


# Stages are deliberately short token fragments the unembedding might surface.
DEFAULT_PROBES: tuple[MathProbe, ...] = (
    MathProbe(
        id="add_then_mul",
        prompt=(
            "Solve step by step. First add 3 and 4, then multiply the sum by 2. "
            "Final answer only after the steps.\n"
        ),
        stages=(
            ("3", "three"),
            ("4", "four"),
            ("7", "sum"),
            ("2", "times", "*"),
            ("14",),
        ),
        answer="14",
    ),
    MathProbe(
        id="sub_chain",
        prompt=(
            "Solve step by step. Start with 20, subtract 5, then subtract 3. "
            "Final answer only after the steps.\n"
        ),
        stages=(
            ("20",),
            ("5",),
            ("15",),
            ("3",),
            ("12",),
        ),
        answer="12",
    ),
    MathProbe(
        id="double_plus",
        prompt=(
            "Solve step by step. Double 6, then add 1. "
            "Final answer only after the steps.\n"
        ),
        stages=(
            ("6", "six"),
            ("2", "double", "times"),
            ("12",),
            ("1",),
            ("13",),
        ),
        answer="13",
    ),
)


DEFAULT_MODEL_PATH = "/mnt/data/models/Qwen2.5-1.5B-Instruct"
DEFAULT_LENS_ROOT = "/mnt/data/models/lenses"
DEFAULT_FIT_N = 64  # paper uses more; 64 is a usable forge smoke
DEFAULT_SEQ_LEN = 128
DEFAULT_TOP_K = 8


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _norm_tok(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().lower())


def top_tokens_contain(top_strings: Sequence[str], candidates: Sequence[str]) -> bool:
    tops = {_norm_tok(t) for t in top_strings}
    for c in candidates:
        nc = _norm_tok(c)
        if not nc:
            continue
        if nc in tops:
            return True
        # substring match for BPE pieces ("14" in "14\n")
        if any(nc in t or t in nc for t in tops if t):
            return True
    return False


def earliest_stage_layers(
    layer_tops: dict[int, list[str]],
    stages: Sequence[Sequence[str]],
) -> list[int | None]:
    """For each stage, earliest layer (numeric) where any candidate appears in top-k."""
    layers_sorted = sorted(layer_tops.keys())
    out: list[int | None] = []
    for stage in stages:
        hit: int | None = None
        for L in layers_sorted:
            if top_tokens_contain(layer_tops[L], stage):
                hit = L
                break
        out.append(hit)
    return out


def intermediate_order_score(stage_layers: Sequence[int | None]) -> float | None:
    """Fraction of consecutive stage pairs that are non-decreasing in layer index.

    None if fewer than 2 stages have hits (cannot score order).
    """
    known = [(i, L) for i, L in enumerate(stage_layers) if L is not None]
    if len(known) < 2:
        return None
    ok = 0
    total = 0
    for (_, la), (_, lb) in zip(known, known[1:]):
        total += 1
        if lb >= la:
            ok += 1
    return ok / total if total else None


def answer_min_rank(layer_tops: dict[int, list[str]], answer: str) -> int | None:
    """Best (lowest) 1-based rank of answer string across layers; None if never in top-k."""
    best: int | None = None
    for tops in layer_tops.values():
        for i, t in enumerate(tops):
            if _norm_tok(answer) in _norm_tok(t) or _norm_tok(t) in _norm_tok(answer):
                rank = i + 1
                best = rank if best is None else min(best, rank)
                break
    return best


# ---------------------------------------------------------------------------
# jlens backend (optional import)
# ---------------------------------------------------------------------------


def _import_jlens():
    try:
        import jlens  # type: ignore
        import torch
        import transformers
    except ImportError as e:
        raise SystemExit(
            "Missing jlens/torch/transformers. On forge:\n"
            "  pip install torch transformers "
            "'git+https://github.com/anthropics/jacobian-lens.git'\n"
            f"Original error: {e}"
        ) from e
    return jlens, torch, transformers


def resolve_model_path(path: str | None, hf_id: str | None) -> str:
    if path and Path(path).is_dir():
        return path
    if hf_id:
        return hf_id
    if path:
        return path
    return DEFAULT_MODEL_PATH


def default_lens_path(model_path: str) -> Path:
    name = Path(model_path.rstrip("/")).name
    return Path(DEFAULT_LENS_ROOT) / name / "jacobian_lens.pt"


def load_hf_model(model_path: str, device: str):
    jlens, torch, transformers = _import_jlens()
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"loading model {model_path!r} device={device} dtype={dtype}", flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    hf.to(device)
    hf.eval()
    model = jlens.from_hf(hf, tok)
    return jlens, model, tok, hf


def cmd_check(args: argparse.Namespace) -> int:
    print("jlens_spike check")
    model_path = resolve_model_path(args.model_path, args.hf_id)
    p = Path(model_path)
    exists = p.is_dir()
    print(f"  model_path: {model_path}  local_dir={exists}")
    if exists:
        print(f"  entries: {len(list(p.iterdir()))}")
    lens = Path(args.lens_path) if args.lens_path else default_lens_path(model_path)
    print(f"  lens_path: {lens}  exists={lens.is_file()}")
    try:
        import jlens  # noqa: F401
        import torch
        import transformers

        print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()}")
        print(f"  transformers {transformers.__version__}")
        print("  jlens OK")
    except ImportError as e:
        print(f"  deps MISSING: {e}")
        return 1
    print("  probes:", len(DEFAULT_PROBES))
    for pr in DEFAULT_PROBES:
        print(f"    - {pr.id}: {len(pr.stages)} stages → answer {pr.answer}")
    return 0


def _fit_prompts(n: int, seq_len: int) -> list[str]:
    """Synthetic fit corpus — no external download required for the smoke."""
    seeds = [
        "The capital of France is Paris and the river is the Seine.",
        "In arithmetic, two plus two equals four.",
        "A recipe calls for flour, water, yeast, and salt.",
        "The speed of light is approximately three times ten to the eight meters per second.",
        "Once upon a time there was a small workshop under a mountain.",
    ]
    out: list[str] = []
    i = 0
    while len(out) < n:
        s = seeds[i % len(seeds)] + f" Example {i}. " + ("lorem " * 20)
        out.append(s[: seq_len * 4])  # rough char budget; tokenizer truncates in fit
        i += 1
    return out


def cmd_fit(args: argparse.Namespace) -> int:
    jlens, model, tok, _hf = load_hf_model(
        resolve_model_path(args.model_path, args.hf_id), args.device
    )
    out = Path(args.lens_path) if args.lens_path else default_lens_path(
        resolve_model_path(args.model_path, args.hf_id)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    prompts = _fit_prompts(args.fit_n, args.seq_len)
    print(f"fitting lens on {len(prompts)} prompts → {out}", flush=True)
    t0 = time.monotonic()
    # API: jlens.fit(model, prompts=..., checkpoint_path=...)
    lens = jlens.fit(
        model,
        prompts=prompts,
        checkpoint_path=str(out.with_suffix(".ckpt.pt")),
    )
    lens.save(str(out))
    meta = {
        "model_path": resolve_model_path(args.model_path, args.hf_id),
        "fit_n": args.fit_n,
        "seq_len": args.seq_len,
        "wall_time_s": time.monotonic() - t0,
        "lens_path": str(out),
        "note": "forge smoke fit; paper uses larger corpus — re-fit for quality",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out} in {meta['wall_time_s']:.1f}s", flush=True)
    return 0


def _layer_index(key: Any) -> int:
    if isinstance(key, int):
        return key
    s = str(key)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


def apply_one(
    jlens_mod: Any,
    lens: Any,
    model: Any,
    tok: Any,
    probe: MathProbe,
    *,
    top_k: int,
    positions: Sequence[int],
) -> dict[str, Any]:
    """Apply lens; return layer→top token strings + scores."""
    t0 = time.monotonic()
    lens_logits, _model_logits, _extra = lens.apply(
        model, probe.prompt, positions=list(positions)
    )
    # lens_logits: layer → tensor [n_pos, vocab] or similar
    layer_tops: dict[int, list[str]] = {}
    for layer_key, logits in lens_logits.items():
        L = _layer_index(layer_key)
        # take last position in the returned tensor
        t = logits
        if hasattr(t, "ndim") and t.ndim >= 2:
            row = t[0] if t.shape[0] <= t.shape[-1] else t[-1]
            if row.ndim > 1:
                row = row[-1]
        else:
            row = t
        k = min(top_k, int(row.numel()) if hasattr(row, "numel") else top_k)
        vals, idx = row.topk(k)
        ids = idx.detach().cpu().tolist()
        if isinstance(ids, int):
            ids = [ids]
        # flatten nested
        flat: list[int] = []
        for x in ids:
            if isinstance(x, list):
                flat.extend(int(y) for y in x)
            else:
                flat.append(int(x))
        tops = [tok.decode([i], skip_special_tokens=False) for i in flat[:top_k]]
        layer_tops[L] = tops

    stage_layers = earliest_stage_layers(layer_tops, probe.stages)
    order = intermediate_order_score(stage_layers)
    ans_rank = answer_min_rank(layer_tops, probe.answer)
    return {
        "probe_id": probe.id,
        "prompt": probe.prompt,
        "answer": probe.answer,
        "stages": [list(s) for s in probe.stages],
        "stage_layers": stage_layers,
        "intermediate_order_score": order,
        "answer_min_rank": ans_rank,
        "layer_tops": {str(k): v for k, v in sorted(layer_tops.items())},
        "wall_time_s": time.monotonic() - t0,
    }


def _gate_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Binary go/no-go from order scores + answer ranks."""
    orders = [
        r["intermediate_order_score"]
        for r in results
        if r.get("intermediate_order_score") is not None
    ]
    ans_hits = sum(1 for r in results if r.get("answer_min_rank") is not None)
    mean_order = sum(orders) / len(orders) if orders else None
    # Gate (documented in docs/spikes/jlens-math.md):
    #  - mean intermediate_order_score >= 0.6 over scored probes
    #  - answer appears in top-k on >= half of probes
    go = (
        mean_order is not None
        and mean_order >= 0.6
        and ans_hits >= max(1, (len(results) + 1) // 2)
    )
    return {
        "go": go,
        "mean_intermediate_order_score": mean_order,
        "n_probes": len(results),
        "n_order_scored": len(orders),
        "n_answer_in_topk": ans_hits,
        "thresholds": {
            "mean_order_min": 0.6,
            "answer_hit_fraction_min": 0.5,
        },
        "decision": "GO — proceed to permanent observe panel"
        if go
        else "NO-GO — keep CLI-only; try larger model or more fit data",
    }


def cmd_apply(args: argparse.Namespace) -> int:
    model_path = resolve_model_path(args.model_path, args.hf_id)
    lens_path = Path(args.lens_path) if args.lens_path else default_lens_path(model_path)
    if not lens_path.is_file():
        raise SystemExit(f"lens not found: {lens_path} — run fit first")

    jlens, model, tok, _hf = load_hf_model(model_path, args.device)
    print(f"loading lens {lens_path}", flush=True)
    # Prefer from_pretrained-style if path is a dir; else load local pt
    if hasattr(jlens.JacobianLens, "load"):
        lens = jlens.JacobianLens.load(str(lens_path))
    elif hasattr(jlens.JacobianLens, "from_pretrained") and lens_path.is_dir():
        lens = jlens.JacobianLens.from_pretrained(str(lens_path))
    else:
        # torch load fallback via jlens if available
        import torch

        obj = torch.load(lens_path, map_location="cpu", weights_only=False)
        if isinstance(obj, jlens.JacobianLens):
            lens = obj
        elif hasattr(jlens.JacobianLens, "from_state_dict"):
            lens = jlens.JacobianLens.from_state_dict(obj)
        else:
            lens = obj  # hope it has .apply

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    positions = [int(x) for x in args.positions.split(",")]

    for probe in DEFAULT_PROBES:
        print(f"apply {probe.id}…", flush=True)
        try:
            rec = apply_one(
                jlens,
                lens,
                model,
                tok,
                probe,
                top_k=args.top_k,
                positions=positions,
            )
        except Exception as e:
            rec = {
                "probe_id": probe.id,
                "error": f"{type(e).__name__}: {e}",
                "intermediate_order_score": None,
                "answer_min_rank": None,
            }
            print(f"  ERROR {rec['error']}", flush=True)
        results.append(rec)
        if rec.get("intermediate_order_score") is not None:
            print(
                f"  order={rec['intermediate_order_score']:.2f} "
                f"answer_rank={rec.get('answer_min_rank')} "
                f"stages={rec.get('stage_layers')}",
                flush=True,
            )

    gate = _gate_decision(results)
    payload = {
        "schema_version": 1,
        "spike": "jlens-math-j0",
        "model_path": model_path,
        "lens_path": str(lens_path),
        "device": args.device,
        "top_k": args.top_k,
        "positions": positions,
        "results": results,
        "gate": gate,
        "ts": time.time(),
    }
    json_path = out_dir / "jlens_spike_results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / "jlens_spike_results.md"
    md_path.write_text(_results_markdown(payload), encoding="utf-8")
    print(json.dumps(gate, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print("Fill docs/spikes/jlens-math.md from this run (paste gate + notes).")
    return 0 if gate["go"] else 2  # exit 2 = NO-GO but ran successfully


def _results_markdown(payload: dict[str, Any]) -> str:
    g = payload["gate"]
    lines = [
        "# J-Lens spike results",
        "",
        f"- model: `{payload['model_path']}`",
        f"- lens: `{payload['lens_path']}`",
        f"- device: `{payload['device']}`",
        "",
        f"## Gate: **{'GO' if g['go'] else 'NO-GO'}**",
        "",
        f"- mean intermediate_order_score: `{g['mean_intermediate_order_score']}`",
        f"- answer in top-k: `{g['n_answer_in_topk']}/{g['n_probes']}`",
        f"- decision: {g['decision']}",
        "",
        "## Per probe",
        "",
    ]
    for r in payload["results"]:
        if r.get("error"):
            lines.append(f"### {r['probe_id']} — ERROR\n\n`{r['error']}`\n")
            continue
        lines.append(f"### {r['probe_id']}")
        lines.append("")
        lines.append(f"- order score: `{r.get('intermediate_order_score')}`")
        lines.append(f"- stage layers: `{r.get('stage_layers')}`")
        lines.append(f"- answer min rank: `{r.get('answer_min_rank')}`")
        lines.append("")
        # compact mid-layer tops
        tops = r.get("layer_tops") or {}
        keys = sorted(tops.keys(), key=lambda x: int(x))
        mid = keys[len(keys) // 3 : 2 * len(keys) // 3] if keys else []
        for L in mid[:6]:
            lines.append(f"  - L{L}: {tops[L][:5]}")
        lines.append("")
    return "\n".join(lines) + "\n"


def cmd_all(args: argparse.Namespace) -> int:
    model_path = resolve_model_path(args.model_path, args.hf_id)
    lens_path = Path(args.lens_path) if args.lens_path else default_lens_path(model_path)
    if not lens_path.is_file():
        print("lens missing — running fit", flush=True)
        rc = cmd_fit(args)
        if rc != 0:
            return rc
    else:
        print(f"using existing lens {lens_path}", flush=True)
    return cmd_apply(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "command",
        choices=("check", "fit", "apply", "all"),
        help="check deps/paths | fit lens | apply math probes | fit-if-needed+apply",
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="local HF dir on lab NVMe")
    p.add_argument("--hf-id", default=None, help="optional HF hub id instead of local path")
    p.add_argument("--lens-path", default=None, help="path to jacobian_lens.pt")
    p.add_argument("--device", default="cuda", help="cuda | cuda:0 | cpu")
    p.add_argument("--fit-n", type=int, default=DEFAULT_FIT_N, help="number of fit prompts")
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument(
        "--positions",
        default="-1",
        help="comma-separated source positions (default: last token of prompt)",
    )
    p.add_argument(
        "--out",
        default="/mnt/data/anvil-runs/jlens-spike",
        help="directory for apply JSON/MD results",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return cmd_check(args)
    if args.command == "fit":
        return cmd_fit(args)
    if args.command == "apply":
        return cmd_apply(args)
    if args.command == "all":
        return cmd_all(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
