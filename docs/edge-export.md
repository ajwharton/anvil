# Edge export (Phase 4.C)

Lab trains **LoRA** on forge/hammer; the house robot loads a **small student**
(default **SmolVLM-256M**). Anvil materializes an **edge package**; GGUF/ONNX/TRT
**converters** are host tools (not vendored).

## Formats

| Format | What Anvil writes | Host follow-up |
|--------|-------------------|----------------|
| `peft` | PEFT adapter dir | load base+adapter on lab |
| `merged_hf` | full HF weights (+ `edge_manifest.json`) | convert / serve |
| `gguf` | `peft/` + optional `merged/` + `edge_manifest.json` + optional `model.gguf` | llama.cpp / Ollama |
| `onnx` | same layout + optional `model.onnx` | ORT / TRT |
| `trt` | recipe package (engine built **on Jetson**) | `trtexec` |

## Python

```python
tc.export_adapter("/path/to/out", format="gguf")
# → out/peft/, out/merged/, out/edge_manifest.json, out/README.md
# → out/model.gguf if ANVIL_GGUF_CONVERTER succeeds
```

Optional env hooks (placeholders `{src}` `{dst}`):

```bash
export ANVIL_GGUF_CONVERTER='python /path/to/convert_hf_to_gguf.py {src} --outfile {dst}'
export ANVIL_ONNX_CONVERTER='optimum-cli export onnx --model {src} {dst}.dir && cp {dst}.dir/model.onnx {dst}'
```

## On-robot sample stub

`JetsonSampleBackend` is **sample-only** (no train verbs). Point it at Ollama:

```bash
export ANVIL_JETSON_URL=http://127.0.0.1:11434
export ANVIL_JETSON_MODEL=smolvlm-256m
```

CI uses `JetsonSampleConfig(dry_run=True)`.

## Edge device storage (j30 / Orin)

| Do on robot | Do on lab (forge / Mac) |
|-------------|-------------------------|
| Serve small GGUF (smol ~256M) | Train LoRA, build packs, write `run_dir` |
| Short vision bursts under **robotics** retention | Hold episode packs + media CAS |
| Prune `~/vision/out` after pull | Convert `vision/out` → house pack |

The j30 is **ops-owned by robotics**, not Anvil. Anvil agents must not invent
long-running capture jobs or leave multi-GB logs on device. SDKs (Orbbec,
etc.) already dominate disk; treat free space as scarce even when `df` looks
comfortable.

## Smoke

```bash
python scripts/edge_export_demo.py --format gguf
python scripts/robot_pack_smoke.py --steps 2   # pack → offline → (optional export)

# Edge sample FPS (lab-side; dry-run default — no robot writes)
python scripts/j30_edge_fps_smoke.py --dry-run --n 3
# Live: ANVIL_JETSON_URL=http://<robot>:11434 python scripts/j30_edge_fps_smoke.py --n 5 --image ./frame.jpg
```

## Safety

Never actuate motors from raw `sample` text without a supervisor gate.
See `prompts/agent/safety_policy.md`.
