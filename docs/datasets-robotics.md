# Robotics & vision datasets (product testing)

Anvil’s vision contract is multimodal **Examples** (text + `cas://` image refs)
and optional **Trajectories** (obs refs + instruction + action + reward).
Large public robotics corpora are **not** vendored in-repo — pull to lab NVMe
and convert into that contract.

## How they map to Anvil

```text
External corpus (OXE / Bridge / LeRobot / …)
        │  convert / subsample
        ▼
LocalMediaStore  ← frames as cas://sha256/…
        │
        ▼
Example JSONL  or  Trajectory JSON
        │
        ▼
recipes: vlm_sft / (later) robot_offline RL
```

Helpers: `anvil.data.ingest` (`examples_from_vlm_jsonl`, `put_images_from_paths`),
`anvil.protocol.trajectory.Trajectory.to_vlm_sft_examples`.

## Recommended open corpora

| Dataset | Scale | Format notes | Anvil use | Links |
|---------|-------|--------------|-----------|--------|
| **Open X-Embodiment (OXE)** | 1M+ real trajectories, 22 embodiments | RLDS / TFRecord; multi-lab pool | Pretrain-scale mixtures; subsample for smoke | [site](https://robotics-transformer-x.github.io/), [paper](https://arxiv.org/abs/2310.08864) |
| **BridgeData V2** | ~60k trajectories, WidowX tabletop | Common OXE constituent; language-conditioned | **Best first fine-tune / LoRA target** for tabletop VLM-SFT | [RAIL](https://rail-berkeley.github.io/bridgedata/) |
| **OpenVLA training mix** | ~970k OXE episodes | Used to train OpenVLA-7B; LoRA FT recipes published | Align recipes with community LoRA practice | [GitHub](https://github.com/openvla/openvla) |
| **DROID** | Large in-the-wild Franka | Diverse scenes; harder action tokenization | Stretch after Bridge works | OpenVLA / OXE docs |
| **LeRobot datasets** | Growing HF hub sets | Hugging Face `lerobot` format | Friendly download path for small smokes | [HF LeRobot](https://huggingface.co/lerobot) |
| **Robo2VLM-1** | ~685k VQA from OXE | Language Q/A over robot scenes | **VLM SFT / classifier** without action head | NeurIPS 2025 Datasets track |

Licenses vary by subset (Apache / CC-BY / non-commercial). Check each source before redistribution.

## Practical lab layout

```text
/mnt/data/datasets/
  bridge_v2/           # raw or RLDS
  oxe_subsample/       # small TFRecord or extracted frames
  anvil_jsonl/         # converted Example JSONL (refs only)
/mnt/data/anvil-media/ # LocalMediaStore root (cas:// blobs)
/mnt/data/models/
  Qwen2.5-VL-3B-Instruct/
```

Never commit frames or multi‑GB shards to the Anvil git tree.

## Suggested conversion path (Bridge / OXE smoke)

1. Extract a **tiny** episode set (e.g. 10–100 trajectories) with RGB frames + language instruction + action summary text.  
2. `LocalMediaStore("/mnt/data/anvil-media").put_path(frame.jpg)` per frame.  
3. Emit JSONL:

```json
{"instruction": "pick up the blue cube", "images": ["cas://sha256/…"], "response": "close gripper; lift", "dataset": "bridge_v2", "episode_id": "ep0"}
```

Or build `Trajectory` objects in Python and call `to_vlm_sft_examples()`.

4. `examples_from_vlm_jsonl(...)` → `run_vlm_sft(..., examples=...)` on forge with Qwen2.5-VL-3B + LoRA (encoder frozen by default).

## RL vs SFT

| Goal | Format | Recipe |
|------|--------|--------|
| Instruction-following / caption / grasp rubric | Example (image + text → text) | `vlm_sft` / classifier-style |
| Offline trajectory policy learning | Trajectory + rewards | Phase 4 `robot_offline` (schema now; trainer later) |
| On-policy vision RL | live sample + reward | Phase 2 GRPO loop + multimodal sample (later) |

OpenVLA-style **action tokenization** is a recipe concern (how `response` is spelled), not a change to the four verbs.

## Product goals

Robotics data is one **application** of Anvil’s **live sufficiency** thesis
(`docs/product.md`: instrument while data is applied; decide “enough” mid-run).
Observe/ops goals are **platform-wide** (roadmap §P.Sufficiency / §P.Ops);
vision must not lag text GRPO.

| Goal | Roadmap |
|------|---------|
| Lab corpus on NVMe in Anvil shape (Bridge / OXE subsample / Robo2VLM) | 3.B |
| Production convert pipeline (RLDS/LeRobot → CAS + JSONL, resumable) | 3.B |
| Scale ladder 1k → 5k → 50k+ exercised on forge | 3.B |
| VLM/SFT `metrics.jsonl` + live `/observe` (**done**); vision probes still open | 3.C / P.Sufficiency |
| Checkpoint/resume + multi-hour VLM job ops | 3.D / P.Ops |
| Offline robot RL + action tokenization + vision on-policy RL | Phase 4 |

## Smoke checklist

### Platform (toy / synthetic)

- [x] Media store put/get for PNG/JPEG on forge (lab demos)  
- [x] 1 JSONL/synthetic row → `run_vlm_sft` fake:// then local://  
- [x] Qwen2.5-VL-3B LoRA smoke, encoder frozen by default  
- [x] `run_vlm_sft(run_dir=…)` → `metrics.jsonl` + `/observe` loss curve (toy)  
- [~] Export PEFT; qualitative sample (smoke only; real corpus TBD)  

### Product corpus (required)

- [ ] Bridge/Robo2VLM slice → `anvil_jsonl` + CAS on lab NVMe  
- [ ] Documented converter CLI (subsample, resume, license)  
- [ ] `run_vlm_sft` on ≥1k real rows with **observe** loss curve  
- [ ] Held-out qualitative sample after export  
- [ ] Multi-hour resume test  

## Out of scope (for now)

- Full OXE download in CI  
- Hosting third-party weights/datasets in this repo  
- Claiming OpenVLA wire compatibility  
