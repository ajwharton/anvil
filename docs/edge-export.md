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

## Smoke

```bash
python scripts/edge_export_demo.py --format gguf
python scripts/robot_pack_smoke.py --steps 2   # pack → offline → (optional export)
```

## Safety

Never actuate motors from raw `sample` text without a supervisor gate.
See `prompts/agent/safety_policy.md`.
