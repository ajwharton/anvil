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

## Default vision / Jetson student

| Field | Value |
|-------|--------|
| **Repo** | [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| **Lab path (forge)** | `/mnt/data/models/Qwen2.5-VL-3B-Instruct` |
| **Class** | Dense VLM (Qwen2.5-VL), instruct |
| **Size** | ~3.75B params · ~7.5 GB BF16 on disk |
| **Why** | Edge-sized student for robot/Jetson; still large enough to LoRA on a Spark |
| **License** | Check the HF card before redistribution |

**Anvil `base_model` string:** `Qwen/Qwen2.5-VL-3B-Instruct`  
Workers resolve that id via lab paths / HF cache on the train host — clients do not need local weights.

### Jetson notes

- Prefer **3B** onboard; use **7B** on forge as teacher if you distill later.
- Edge export: PEFT/merged → AWQ / ONNX / TRT when Phase 4 lands.
- Same Anvil multimodal message schema lab ↔ edge.

## Optional lab bases

| Preset | HF repo | Notes |
|--------|---------|--------|
| `qwen2.5-vl-3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | **default** |
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
