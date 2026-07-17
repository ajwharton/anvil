#!/usr/bin/env python3
"""J0 forge spike — fit/apply Jacobian lens on a small dense model (math order gate).

Protocols (``--protocol``):

  last_prompt   v1 — apply at last prompt token only (known weak for math)
  cot_in_prompt v2 — prompt already contains intermediate steps; multi-position
  generate      v2 — greedy-generate a short CoT, then apply on full sequence

Phase 2.5 gate: reproduce "intermediate steps light up in order" before a
permanent anvil-web J-Lens panel. See docs/spikes/jlens-math.md.

Examples (forge)::

  python scripts/jlens_spike.py apply \\
    --model-path /mnt/data/models/qwen2.5-1.5b-instruct \\
    --protocol cot_in_prompt,generate \\
    --out /mnt/data/anvil-runs/jlens-spike-v2
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
# Math probe set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MathProbe:
    """Multi-step probe with ordered concept stages + optional CoT body."""

    id: str
    prompt: str
    stages: tuple[tuple[str, ...], ...]
    answer: str
    # Partial solution already written (protocol cot_in_prompt)
    cot_prompt: str = ""


DEFAULT_PROBES: tuple[MathProbe, ...] = (
    MathProbe(
        id="add_then_mul",
        prompt=(
            "Solve step by step. First add 3 and 4, then multiply the sum by 2. "
            "Final answer only after the steps.\n"
        ),
        cot_prompt=(
            "Solve step by step.\n"
            "Problem: First add 3 and 4, then multiply the sum by 2.\n"
            "Step 1: 3 + 4 = 7\n"
            "Step 2: 7 * 2 = "
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
        cot_prompt=(
            "Solve step by step.\n"
            "Problem: Start with 20, subtract 5, then subtract 3.\n"
            "Step 1: 20 - 5 = 15\n"
            "Step 2: 15 - 3 = "
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
        cot_prompt=(
            "Solve step by step.\n"
            "Problem: Double 6, then add 1.\n"
            "Step 1: 6 * 2 = 12\n"
            "Step 2: 12 + 1 = "
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


DEFAULT_MODEL_PATH = "/mnt/data/models/qwen2.5-1.5b-instruct"
DEFAULT_LENS_ROOT = "/mnt/data/models/lenses"
DEFAULT_FIT_N = 64
DEFAULT_SEQ_LEN = 128
DEFAULT_TOP_K = 8
DEFAULT_PROTOCOLS = ("cot_in_prompt", "generate", "last_prompt")


# ---------------------------------------------------------------------------
# Scoring — package SSOT with local fallback
# ---------------------------------------------------------------------------

try:
    from anvil.observe.jlens import (  # type: ignore
        answer_min_rank,
        earliest_stage_layers,
        intermediate_order_score,
        token_matches,
        top_tokens_contain,
    )
except ImportError:  # pragma: no cover

    def _norm_tok(s: str) -> str:
        return re.sub(r"\s+", "", s.strip().lower())

    def _alnum(s: str) -> str:
        return re.sub(r"[^0-9a-z]+", "", _norm_tok(s))

    def token_matches(surface: str, candidate: str) -> bool:
        s, c = _norm_tok(surface), _norm_tok(candidate)
        if not s or not c:
            return False
        if s == c:
            return True
        sa, ca = _alnum(surface), _alnum(candidate)
        if not sa or not ca:
            return False
        if ca.isdigit() or sa.isdigit():
            return sa == ca
        if len(ca) >= 2 and (ca in sa or sa in ca):
            return True
        return False

    def top_tokens_contain(top_strings: Sequence[str], candidates: Sequence[str]) -> bool:
        return any(token_matches(t, c) for t in top_strings if t for c in candidates)

    def earliest_stage_layers(
        layer_tops: dict[int, list[str]],
        stages: Sequence[Sequence[str]],
    ) -> list[int | None]:
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
        known = [(i, L) for i, L in enumerate(stage_layers) if L is not None]
        if len(known) < 2:
            return None
        ok = total = 0
        for (_, la), (_, lb) in zip(known, known[1:]):
            total += 1
            if lb >= la:
                ok += 1
        return ok / total if total else None

    def answer_min_rank(layer_tops: dict[int, list[str]], answer: str) -> int | None:
        best: int | None = None
        for tops in layer_tops.values():
            for i, t in enumerate(tops):
                if token_matches(t, answer):
                    rank = i + 1
                    best = rank if best is None else min(best, rank)
                    break
        return best


def position_order_score(
    pos_tops: dict[int, list[str]],
    stages: Sequence[Sequence[str]],
) -> tuple[float | None, list[int | None]]:
    """Earliest *sequence position index* where each stage hits (mid-layer tops).

    pos_tops maps position index 0..n-1 → top token strings at a fixed layer.
    """
    order_idx = sorted(pos_tops.keys())
    stage_pos: list[int | None] = []
    for stage in stages:
        hit: int | None = None
        for p in order_idx:
            if top_tokens_contain(pos_tops[p], stage):
                hit = p
                break
        stage_pos.append(hit)
    return intermediate_order_score(stage_pos), stage_pos


# ---------------------------------------------------------------------------
# jlens backend
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
    try:
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=True
        )
    except TypeError:
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        )
    hf.to(device)
    hf.eval()
    model = jlens.from_hf(hf, tok)
    return jlens, model, tok, hf


def load_lens(jlens: Any, lens_path: Path) -> Any:
    if hasattr(jlens.JacobianLens, "load"):
        return jlens.JacobianLens.load(str(lens_path))
    if hasattr(jlens.JacobianLens, "from_pretrained") and lens_path.is_dir():
        return jlens.JacobianLens.from_pretrained(str(lens_path))
    import torch

    obj = torch.load(lens_path, map_location="cpu", weights_only=False)
    if isinstance(obj, jlens.JacobianLens):
        return obj
    if hasattr(jlens.JacobianLens, "from_state_dict"):
        return jlens.JacobianLens.from_state_dict(obj)
    return obj


def cmd_check(args: argparse.Namespace) -> int:
    print("jlens_spike check")
    model_path = resolve_model_path(args.model_path, args.hf_id)
    p = Path(model_path)
    print(f"  model_path: {model_path}  local_dir={p.is_dir()}")
    lens = Path(args.lens_path) if args.lens_path else default_lens_path(model_path)
    print(f"  lens_path: {lens}  exists={lens.is_file()}")
    try:
        import torch
        import transformers

        print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()}")
        print(f"  transformers {transformers.__version__}")
        import jlens  # noqa: F401

        print("  jlens OK")
    except ImportError as e:
        print(f"  deps MISSING: {e}")
        return 1
    print("  probes:", len(DEFAULT_PROBES))
    for pr in DEFAULT_PROBES:
        print(f"    - {pr.id}: stages={len(pr.stages)} answer={pr.answer} cot={bool(pr.cot_prompt)}")
    print("  protocols:", ",".join(args.protocol) if hasattr(args, "protocol") else DEFAULT_PROTOCOLS)
    return 0


def _fit_prompts(n: int, seq_len: int) -> list[str]:
    seeds = [
        "The capital of France is Paris and the river is the Seine.",
        "In arithmetic, two plus two equals four. Three plus four equals seven.",
        "Twenty minus five is fifteen. Fifteen minus three is twelve.",
        "Double six is twelve. Twelve plus one is thirteen.",
        "A recipe calls for flour, water, yeast, and salt.",
    ]
    out: list[str] = []
    i = 0
    while len(out) < n:
        s = seeds[i % len(seeds)] + f" Example {i}. " + ("lorem " * 20)
        out.append(s[: seq_len * 4])
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
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out} in {meta['wall_time_s']:.1f}s", flush=True)
    return 0


def _layer_index(key: Any) -> int:
    if isinstance(key, int):
        return key
    m = re.search(r"(\d+)", str(key))
    return int(m.group(1)) if m else 0


def _row_topk(logits: Any, pos_i: int, top_k: int, tok: Any) -> list[str]:
    t = logits
    if hasattr(t, "ndim") and t.ndim == 3:
        t = t[0]
    if hasattr(t, "ndim") and t.ndim >= 2:
        # [n_pos, vocab]
        n_pos = t.shape[0]
        idx = pos_i if pos_i >= 0 else n_pos + pos_i
        idx = max(0, min(n_pos - 1, idx))
        row = t[idx]
    else:
        row = t
    k = min(top_k, int(row.numel()) if hasattr(row, "numel") else top_k)
    _, ids = row.topk(k)
    flat = ids.detach().cpu().tolist()
    if isinstance(flat, int):
        flat = [flat]
    out: list[int] = []
    for x in flat:
        if isinstance(x, list):
            out.extend(int(y) for y in x)
        else:
            out.append(int(x))
    return [tok.decode([i], skip_special_tokens=False) for i in out[:top_k]]


def _layer_tops_pooled(
    lens_logits: dict[Any, Any], tok: Any, top_k: int, pos_indices: Sequence[int]
) -> dict[int, list[str]]:
    """Union top-k strings across positions, per layer (order preserved, deduped)."""
    layer_tops: dict[int, list[str]] = {}
    for layer_key, logits in lens_logits.items():
        L = _layer_index(layer_key)
        seen: list[str] = []
        for pi in range(len(pos_indices)):
            for t in _row_topk(logits, pi, top_k, tok):
                if t not in seen:
                    seen.append(t)
        layer_tops[L] = seen[: max(top_k * 2, top_k)]
    return layer_tops


def _mid_layer_pos_tops(
    lens_logits: dict[Any, Any], tok: Any, top_k: int, n_pos: int
) -> dict[int, list[str]]:
    layers = sorted(lens_logits.keys(), key=_layer_index)
    mid = layers[len(layers) // 2]
    logits = lens_logits[mid]
    return {pi: _row_topk(logits, pi, top_k, tok) for pi in range(n_pos)}


def _greedy_complete(hf: Any, tok: Any, prompt: str, device: str, max_new: int) -> str:
    import torch

    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = hf.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)


def apply_text(
    lens: Any,
    model: Any,
    tok: Any,
    text: str,
    *,
    positions: Sequence[int],
    top_k: int,
    probe: MathProbe,
    protocol: str,
) -> dict[str, Any]:
    t0 = time.monotonic()
    lens_logits, _model_logits, _extra = lens.apply(model, text, positions=list(positions))
    n_pos = len(positions)
    layer_tops = _layer_tops_pooled(lens_logits, tok, top_k, positions)
    # also last-position-only for comparison
    last_only: dict[int, list[str]] = {}
    for lk, logits in lens_logits.items():
        last_only[_layer_index(lk)] = _row_topk(logits, n_pos - 1, top_k, tok)

    stage_layers = earliest_stage_layers(layer_tops, probe.stages)
    order = intermediate_order_score(stage_layers)
    ans_rank = answer_min_rank(layer_tops, probe.answer)
    stage_layers_last = earliest_stage_layers(last_only, probe.stages)
    order_last = intermediate_order_score(stage_layers_last)

    pos_tops = _mid_layer_pos_tops(lens_logits, tok, top_k, n_pos)
    pos_order, stage_pos = position_order_score(pos_tops, probe.stages)

    return {
        "probe_id": probe.id,
        "protocol": protocol,
        "text_preview": text[:240],
        "answer": probe.answer,
        "stages": [list(s) for s in probe.stages],
        "positions": list(positions),
        "stage_layers": stage_layers,
        "intermediate_order_score": order,
        "answer_min_rank": ans_rank,
        "stage_layers_last_pos": stage_layers_last,
        "order_score_last_pos": order_last,
        "position_order_score": pos_order,
        "stage_positions_mid_layer": stage_pos,
        "layer_tops_pooled": {str(k): v for k, v in sorted(layer_tops.items())},
        "layer_tops_last": {str(k): v for k, v in sorted(last_only.items())},
        "mid_layer_pos_tops": {str(k): v for k, v in sorted(pos_tops.items())},
        "wall_time_s": time.monotonic() - t0,
    }


def _positions_for(text: str, tok: Any, protocol: str, explicit: Sequence[int] | None) -> list[int]:
    if explicit:
        return list(explicit)
    n = len(tok.encode(text))
    if protocol == "last_prompt":
        return [-1]
    # multi-position: last min(12, n) tokens
    span = min(12, max(1, n))
    return list(range(-span, 0))


def apply_probe(
    lens: Any,
    model: Any,
    tok: Any,
    hf: Any,
    probe: MathProbe,
    *,
    protocol: str,
    top_k: int,
    device: str,
    max_new: int,
    explicit_positions: Sequence[int] | None,
) -> dict[str, Any]:
    if protocol == "last_prompt":
        text = probe.prompt
    elif protocol == "cot_in_prompt":
        text = probe.cot_prompt or probe.prompt
    elif protocol == "generate":
        text = _greedy_complete(hf, tok, probe.prompt, device, max_new)
    else:
        raise ValueError(f"unknown protocol {protocol!r}")

    positions = _positions_for(text, tok, protocol, explicit_positions)
    rec = apply_text(
        lens, model, tok, text, positions=positions, top_k=top_k, probe=probe, protocol=protocol
    )
    if protocol == "generate":
        rec["generated_full"] = text
    return rec


def _primary_order(rec: dict[str, Any]) -> float | None:
    """Gate uses best of pooled-layer order and position order (v2)."""
    scores = [
        rec.get("intermediate_order_score"),
        rec.get("position_order_score"),
    ]
    nums = [float(s) for s in scores if s is not None]
    return max(nums) if nums else None


def _gate_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    # Prefer primary_order if set by apply loop
    orders = []
    for r in results:
        if r.get("error"):
            continue
        o = r.get("primary_order_score")
        if o is None:
            o = _primary_order(r)
        if o is not None:
            orders.append(float(o))
    ans_hits = sum(1 for r in results if r.get("answer_min_rank") is not None and not r.get("error"))
    n_ok = sum(1 for r in results if not r.get("error"))
    mean_order = sum(orders) / len(orders) if orders else None
    go = (
        mean_order is not None
        and mean_order >= 0.6
        and ans_hits >= max(1, (n_ok + 1) // 2)
        and n_ok > 0
    )
    return {
        "go": go,
        "mean_intermediate_order_score": mean_order,
        "n_probes": n_ok,
        "n_order_scored": len(orders),
        "n_answer_in_topk": ans_hits,
        "thresholds": {"mean_order_min": 0.6, "answer_hit_fraction_min": 0.5},
        "decision": (
            "GO — intermediate-order signal supports product panel work"
            if go
            else "NO-GO — signal too weak; deprioritize permanent J-Lens panel"
        ),
    }


def cmd_apply(args: argparse.Namespace) -> int:
    model_path = resolve_model_path(args.model_path, args.hf_id)
    lens_path = Path(args.lens_path) if args.lens_path else default_lens_path(model_path)
    if not lens_path.is_file():
        raise SystemExit(f"lens not found: {lens_path} — run fit first")

    jlens, model, tok, hf = load_hf_model(model_path, args.device)
    print(f"loading lens {lens_path}", flush=True)
    lens = load_lens(jlens, lens_path)

    protocols = list(args.protocol)
    explicit = None
    if args.positions.strip() and args.positions.strip() != "auto":
        explicit = [int(x) for x in args.positions.split(",")]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    by_protocol: dict[str, list[dict[str, Any]]] = {}

    for protocol in protocols:
        print(f"\n=== protocol={protocol} ===", flush=True)
        prec: list[dict[str, Any]] = []
        for probe in DEFAULT_PROBES:
            print(f"apply {probe.id}…", flush=True)
            try:
                rec = apply_probe(
                    lens,
                    model,
                    tok,
                    hf,
                    probe,
                    protocol=protocol,
                    top_k=args.top_k,
                    device=args.device,
                    max_new=args.max_new_tokens,
                    explicit_positions=explicit,
                )
                rec["primary_order_score"] = _primary_order(rec)
                print(
                    f"  layer_order={rec.get('intermediate_order_score')} "
                    f"pos_order={rec.get('position_order_score')} "
                    f"ans_rank={rec.get('answer_min_rank')} "
                    f"stages_L={rec.get('stage_layers')}",
                    flush=True,
                )
            except Exception as e:
                rec = {
                    "probe_id": probe.id,
                    "protocol": protocol,
                    "error": f"{type(e).__name__}: {e}",
                    "intermediate_order_score": None,
                    "answer_min_rank": None,
                    "primary_order_score": None,
                }
                print(f"  ERROR {rec['error']}", flush=True)
            prec.append(rec)
            all_results.append(rec)
        by_protocol[protocol] = prec

    # Per-protocol gates + overall (best protocol mean used for decision summary)
    protocol_gates = {p: _gate_decision(rs) for p, rs in by_protocol.items()}
    # Overall: GO if any protocol is GO
    any_go = any(g["go"] for g in protocol_gates.values())
    # Aggregate for primary decision: use best mean among protocols with scores
    means = [
        (p, g["mean_intermediate_order_score"])
        for p, g in protocol_gates.items()
        if g["mean_intermediate_order_score"] is not None
    ]
    best = max(means, key=lambda x: x[1]) if means else (None, None)
    overall = {
        "go": any_go,
        "best_protocol": best[0],
        "best_mean_order": best[1],
        "protocol_gates": protocol_gates,
        "decision": (
            f"GO via protocol={best[0]}"
            if any_go
            else "NO-GO across all protocols — weak pursue signal for permanent panel"
        ),
    }

    payload = {
        "schema_version": 2,
        "spike": "jlens-math-j0-v2",
        "model_path": model_path,
        "lens_path": str(lens_path),
        "device": args.device,
        "top_k": args.top_k,
        "protocols": protocols,
        "results": all_results,
        "gate": overall,
        "ts": time.time(),
    }
    json_path = out_dir / "jlens_spike_results.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / "jlens_spike_results.md"
    md_path.write_text(_results_markdown(payload), encoding="utf-8")
    print(json.dumps(overall, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0 if any_go else 2


def _results_markdown(payload: dict[str, Any]) -> str:
    g = payload["gate"]
    lines = [
        "# J-Lens spike results (protocol v2)",
        "",
        f"- model: `{payload['model_path']}`",
        f"- lens: `{payload['lens_path']}`",
        f"- device: `{payload['device']}`",
        f"- protocols: `{payload.get('protocols')}`",
        "",
        f"## Gate: **{'GO' if g['go'] else 'NO-GO'}**",
        "",
        f"- best protocol: `{g.get('best_protocol')}` mean_order=`{g.get('best_mean_order')}`",
        f"- decision: {g.get('decision')}",
        "",
    ]
    for p, pg in (g.get("protocol_gates") or {}).items():
        lines.append(
            f"### protocol `{p}`: {'GO' if pg['go'] else 'NO-GO'} "
            f"(mean={pg['mean_intermediate_order_score']}, "
            f"ans_hits={pg['n_answer_in_topk']}/{pg['n_probes']})"
        )
        lines.append("")
    lines.append("## Per probe")
    lines.append("")
    for r in payload["results"]:
        if r.get("error"):
            lines.append(f"### {r.get('protocol')}/{r.get('probe_id')} — ERROR\n\n`{r['error']}`\n")
            continue
        lines.append(f"### {r.get('protocol')}/{r.get('probe_id')}")
        lines.append("")
        lines.append(f"- primary_order: `{r.get('primary_order_score')}`")
        lines.append(f"- layer_order (pooled pos): `{r.get('intermediate_order_score')}` stages `{r.get('stage_layers')}`")
        lines.append(f"- pos_order (mid layer): `{r.get('position_order_score')}` stages `{r.get('stage_positions_mid_layer')}`")
        lines.append(f"- answer_min_rank: `{r.get('answer_min_rank')}`")
        lines.append(f"- text: `{r.get('text_preview', '')[:120]}…`")
        lines.append("")
        tops = r.get("layer_tops_last") or {}
        keys = sorted(tops.keys(), key=lambda x: int(x))
        if keys:
            mid = keys[len(keys) // 2]
            lines.append(f"  - last-pos mid L{mid}: {tops[mid][:6]}")
            lines.append(f"  - last-pos final L{keys[-1]}: {tops[keys[-1]][:6]}")
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
    p.add_argument("command", choices=("check", "fit", "apply", "all"))
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--hf-id", default=None)
    p.add_argument("--lens-path", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fit-n", type=int, default=DEFAULT_FIT_N)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument(
        "--protocol",
        default=",".join(DEFAULT_PROTOCOLS),
        help="comma list: last_prompt,cot_in_prompt,generate",
    )
    p.add_argument(
        "--positions",
        default="auto",
        help="'auto' or comma-separated positions (e.g. -1 or -8,-7,...,-1)",
    )
    p.add_argument("--max-new-tokens", type=int, default=48, help="for protocol=generate")
    p.add_argument("--out", default="/mnt/data/anvil-runs/jlens-spike-v2")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # normalize protocol list
    args.protocol = [x.strip() for x in str(args.protocol).split(",") if x.strip()]
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
