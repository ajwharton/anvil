#!/usr/bin/env python3
"""Phase 4.C edge export package smoke (fake://).

Trains a tiny adapter, then builds PEFT / GGUF / ONNX edge packages with
``edge_manifest.json`` (converters optional via env).

Examples::

  python scripts/edge_export_demo.py
  ANVIL_GGUF_CONVERTER='touch {dst}' python scripts/edge_export_demo.py --format gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Edge export package demo")
    p.add_argument(
        "--format",
        default="gguf",
        choices=("peft", "merged_hf", "gguf", "onnx", "trt"),
    )
    p.add_argument("--out", default="/tmp/anvil-edge-export")
    args = p.parse_args(argv)

    from anvil.backends.jetson import JetsonSampleBackend, JetsonSampleConfig
    from anvil.client.service import ServiceClient
    from anvil.export.edge import package_edge_export
    from anvil.protocol.types import (
        AdamParams,
        Datum,
        ExportFormat,
        LoraConfig,
        LoraTargets,
        ModelInput,
        SamplingParams,
        TrainConfig,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fmt = ExportFormat(args.format)

    svc = ServiceClient(endpoint="fake://")
    tc = svc.create_lora_training_client(
        base_model="toy/SmolEdge-256M",
        rank=8,
        modalities=["text"],
    )
    tokens = list(range(10, 30))
    datum = Datum(
        model_input=ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": [0.0] * 5 + [1.0] * (len(tokens) - 6),
        },
    )
    tc.forward_backward([datum], loss_fn="cross_entropy").result()
    tc.optim_step(AdamParams(learning_rate=1e-3)).result()

    exp_path = out / args.format
    result = tc.export_adapter(str(exp_path), format=args.format)

    # Ensure edge package layout for formats that only write flat files on fake
    man_path = Path(result.path) / "edge_manifest.json"
    if not man_path.is_file() and fmt != ExportFormat.PEFT:
        bundle = package_edge_export(
            fmt=fmt,
            root=out / f"{args.format}_pkg",
            adapter_id=result.adapter_id,
            base_model="toy/SmolEdge-256M",
            save_peft=lambda d: (
                d.mkdir(parents=True, exist_ok=True),
                (d / "adapter_config.json").write_text("{}", encoding="utf-8"),
            )
            and None,
        )
        result_path = bundle.path
        man = bundle.manifest.to_public()
    elif man_path.is_file():
        result_path = result.path
        man = json.loads(man_path.read_text(encoding="utf-8"))
    else:
        # PEFT: still emit a package for operators
        bundle = package_edge_export(
            fmt=ExportFormat.PEFT,
            root=out / "peft_pkg",
            adapter_id=result.adapter_id,
            base_model="toy/SmolEdge-256M",
            save_peft=lambda d: (
                d.mkdir(parents=True, exist_ok=True),
                (d / "adapter_config.json").write_text(
                    json.dumps({"r": 8}), encoding="utf-8"
                ),
            )
            and None,
        )
        result_path = bundle.path
        man = bundle.manifest.to_public()

    jet = JetsonSampleBackend(JetsonSampleConfig(dry_run=True))
    jid = jet.create_lora_session(
        TrainConfig(
            base_model="smolvlm-256m",
            lora=LoraConfig(rank=8, targets=LoraTargets()),
        )
    )
    sample = jet.sample(
        base_model="smolvlm-256m",
        adapter_id=jid,
        prompt=ModelInput.from_ints([10, 11, 12]),
        sampling_params=SamplingParams(max_tokens=16, temperature=0.2),
    )

    print(
        json.dumps(
            {
                "export": {
                    "format": args.format,
                    "path": result_path,
                    "adapter_id": result.adapter_id,
                },
                "manifest": man,
                "jetson_dry_run_n_tokens": len(sample.sequences[0].tokens),
                "jetson_last_text": jet.last_text,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
