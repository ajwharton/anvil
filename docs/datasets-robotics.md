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

Helpers:

- `anvil.data.convert` / `scripts/convert_robotics_corpus.py` — **episode pack or path JSONL → CAS + Anvil JSONL** (resumable, subsample)
- `anvil.data.ingest` (`examples_from_vlm_jsonl`, `put_images_from_paths`)
- `anvil.protocol.trajectory.Trajectory.to_vlm_sft_examples`

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

## Conversion pipeline (3.B)

### Episode pack intermediate (recommended)

Raw Bridge/OXE/RLDS is multi‑GB and TF-heavy. On the lab host, materialize a
small **episode pack** (or use an existing path JSONL), then run Anvil’s
converter (no TensorFlow required):

```text
source/
  ep_0001/
    meta.json          # language_instruction, actions[], optional license
    frames/0000.jpg
  ep_0002/
    ...
```

```bash
# synthetic smoke (CI / laptop)
python scripts/convert_robotics_corpus.py --demo --max-rows 100 \
  --media-root /tmp/anvil-media \
  --output /tmp/anvil_jsonl/demo.jsonl

# lab — Bridge-like pack → 1k rows (resume-safe)
python scripts/convert_robotics_corpus.py \
  --source /mnt/data/datasets/bridge_v2/episode_pack \
  --media-root /mnt/data/anvil-media \
  --output /mnt/data/datasets/anvil_jsonl/bridge_1k.jsonl \
  --dataset bridge_v2 \
  --license "BridgeData V2 — check RAIL terms before redistribute" \
  --max-rows 1000 \
  --frames-per-episode 4

# train
python scripts/robot_vlm_sft_demo.py --source jsonl \
  --jsonl /mnt/data/datasets/anvil_jsonl/bridge_1k.jsonl \
  --media-root /mnt/data/anvil-media \
  --run-id bridge-1k-sft --steps 50
```

Output row shape:

```json
{"instruction": "pick up the blue cube", "images": ["cas://sha256/…"], "response": "0.10 0.00 …", "dataset": "bridge_v2", "episode_id": "ep0", "license": "…"}
```

Knobs: `--max-rows`, `--max-episodes`, `--frames-per-episode`, `--row-mode per_frame|keyframe`,
`--no-resume` (default **resumes** via `<output>.state.json`).

Path JSONL with local image paths: `--kind path_jsonl --source rows.jsonl`.

Or build `Trajectory` objects in Python and call `to_vlm_sft_examples()`.

`examples_from_vlm_jsonl(...)` → `run_vlm_sft(..., examples=...)` on forge with Qwen2.5-VL-3B + LoRA (encoder frozen by default).

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

| Goal | Roadmap (Expert ladder) |
|------|-------------------------|
| Lab corpus on NVMe in Anvil shape (Bridge / OXE subsample / Robo2VLM) | **Expert-v0** (converter shipped; operator extract) |
| Production convert pipeline (episode_pack / path JSONL → CAS + JSONL, resumable) | **done** (historical 3.B) |
| Scale ladder 1k → 5k → 50k+ tooling + forge runbook | **done** (`docs/scale-ladder.md`, `scripts/scale_ladder.py`) |
| VLM/SFT `metrics.jsonl` + live `/observe` + probes | **done** (Expert-v0/v1) |
| Checkpoint/resume + multi-hour lab profile | **done** (`checkpoint` + `lab_smokes --profile multi_hour`) |
| Offline robot RL + action tokenization + vision on-policy RL | **Path: robotics** (historical Phase 4) |

## Smoke checklist

### Platform (toy / synthetic)

- [x] Media store put/get for PNG/JPEG on forge (lab demos)  
- [x] 1 JSONL/synthetic row → `run_vlm_sft` fake:// then local://  
- [x] Qwen2.5-VL-3B LoRA smoke, encoder frozen by default  
- [x] `run_vlm_sft(run_dir=…)` → `metrics.jsonl` + `/observe` loss curve (toy)  
- [~] Export PEFT; qualitative sample (smoke only; real corpus TBD)  

### Product corpus (required)

- [x] Documented converter CLI (`scripts/convert_robotics_corpus.py`: subsample, resume, license)  
- [x] Synthetic episode_pack → `anvil_jsonl` + CAS (CI + `--demo`)  
- [ ] Bridge/Robo2VLM slice extracted to episode_pack on lab NVMe  
- [ ] `run_vlm_sft` on ≥1k **real** Bridge rows with observe (operator; ladder tooling ready)  
- [ ] Held-out qualitative sample after export  
- [x] Multi-hour resume path + lab smoke — `checkpoint_every`/`resume` + `fake_multi_hour_resume` / `multi_hour` profile  
- [x] Scale ladder demo smoke (`fake_scale_ladder`); forge: `scale_ladder.py --no-demo`


## Out of scope (for now)

- Full OXE download in CI  
- Hosting third-party weights/datasets in this repo  
- Claiming OpenVLA wire compatibility  
