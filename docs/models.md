# Reference models

**Where weights live:** lab NVMe on **forge** / **hammer**, not the Mac and not this git tree.

| Host | Role | Paths |
|------|------|--------|
| **forge** | Primary Anvil train/sample (default) | `/mnt/data/models/…`, HF cache `/mnt/data/hf_cache` |
| **hammer** | Peer Spark (sample/train split later) | same layout when mirrored |
| **Mac client** | Control-plane client only | no multi‑GB bases |

Pull:

```bash
# from Anvil repo on the Mac — SSHes to forge and downloads there
python scripts/pull_base_model.py
python scripts/pull_base_model.py --host hammer
python scripts/pull_base_model.py --preset qwen2.5-vl-7b --host forge
```

Red line (same as `start.md`): never commit weights, datasets, or media blobs.

## Memory-constrained robot (default for `robot_offline`)

| Field | Value |
|-------|--------|
| **Repo** | [`HuggingFaceTB/SmolVLM-256M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) |
| **Class** | Edge student VLM (SmolVLM) |
| **Size** | ~256M params · ~0.5 GB class on disk (GGUF/Q8 smaller still) |
| **Why** | Fits severe on-robot memory; families note 256M/500M iterate in minutes |
| **Anvil entry** | `anvil.recipes.robot_offline.DEFAULT_ROBOT_BASE` / catalog `robot_offline_edge` |

Text-only fallback: `HuggingFaceTB/SmolLM2-135M-Instruct` with `run_robot_offline(..., text_only=True)`.

## Lab vision / Jetson student (larger)

| Field | Value |
|-------|--------|
| **Repo** | [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| **Lab path (forge)** | `/mnt/data/models/Qwen2.5-VL-3B-Instruct` |
| **Class** | Dense VLM (Qwen2.5-VL), instruct |
| **Size** | ~3.75B params · ~7.5 GB BF16 on disk |
| **Why** | Lab student / teacher; LoRA on a Spark when the robot cannot hold 3B |
| **License** | Check the HF card before redistribution |

**Anvil `base_model` string (lab):** `Qwen/Qwen2.5-VL-3B-Instruct`  
Workers resolve that id via lab paths / HF cache on the train host — clients do not need local weights.

### Jetson / on-robot notes

- Prefer **SmolVLM-256M** (or 500M) onboard when RAM is tight; use **3B** on forge as teacher if you distill later.
- Edge export: PEFT/merged → AWQ / ONNX / TRT (Phase 4.C).
- Same Anvil multimodal message schema lab ↔ edge; action targets are **text tokens** (bins).

## Optional lab bases

| Preset | HF repo | Notes |
|--------|---------|--------|
| `smolvlm-256m` | `HuggingFaceTB/SmolVLM-256M-Instruct` | **robot_offline default** |
| `smollm2-135m` | `HuggingFaceTB/SmolLM2-135M-Instruct` | text-only robot / pipeline smoke |
| `qwen2.5-vl-3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | lab VLM default |
| `qwen2.5-vl-3b-awq` | `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` | quant sample/edge experiments |
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | stronger lab VLM |
| `qwen2.5-vl-7b-awq` | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | quant 7B |

## Checklist after pull

```bash
ssh forge 'du -sh /mnt/data/models/Qwen2.5-VL-3B-Instruct && ls /mnt/data/models/Qwen2.5-VL-3B-Instruct | head'
```

1. Confirm snapshot on forge (or hammer).
2. Optional smoke load **on the lab host** (transformers + vision deps).
3. Phase 1+ workers use that path; Mac only holds the client + `base_model` id.
