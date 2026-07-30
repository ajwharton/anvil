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
recipes: vlm_sft / robot_offline (text-tokenized actions)
```

Helpers:

- `anvil.data.convert` / `scripts/convert_robotics_corpus.py` — **episode pack or path JSONL → CAS + Anvil JSONL** (resumable, subsample)
- `anvil.data.ingest` (`examples_from_vlm_jsonl`, `put_images_from_paths`)
- `anvil.protocol.trajectory.Trajectory.to_vlm_sft_examples` (+ optional `ActionTokenizer`)
- `anvil.protocol.action_tokens.ActionTokenizer` — OpenVLA-style **bins** (default 256 over [-1,1]) or continuous decimal text
- `anvil.recipes.robot_offline.run_robot_offline` — trajectories → tokenized targets → LoRA CE; default base **SmolVLM-256M**

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
| Offline trajectory policy learning | Trajectory + rewards | `run_robot_offline` + `ActionTokenizer` (bins v1) |
| On-policy vision RL | live sample + reward | Phase 2 GRPO loop + multimodal sample (later) |

OpenVLA-style **action tokenization** is a recipe concern (how `response` is spelled), not a change to the four verbs.

### Memory-constrained robot (smol ~250M)

In-house robots that only fit **~256M** should use:

| Piece | Choice |
|-------|--------|
| Base | `HuggingFaceTB/SmolVLM-256M-Instruct` (or SmolLM text-only via `text_only=True`) |
| Recipe | `robot_offline_edge` / `run_robot_offline` |
| Actions | `ActionTokenizer(scheme="bins", n_bins=256)` |
| Knobs | rank 8, freeze vision, short seq (see `throughput` edge×robot_offline) |

```python
from anvil.recipes.robot_offline import run_robot_offline, toy_robot_trajectories

res = run_robot_offline(
    trajectories=toy_robot_trajectories(),  # or your Trajectory list
    steps=50,
    endpoint="local://",  # lab / on-robot backend
    run_dir="/path/to/run",
)
```

### In-house (j30-style) pack

```text
house_pack/
  ep_0001/
    meta.json     # language_instruction, actions[], captions[], detections[]
    frames/0000.jpg
```

```bash
python scripts/robot_pack_smoke.py --steps 2
python scripts/robot_pack_smoke.py --pack /path/to/house_pack \
  --endpoint local:// --model HuggingFaceTB/SmolVLM-256M-Instruct --steps 50
```

Python: `anvil.data.robot_pack.house_pack_to_trajectories` / `house_pack_to_jsonl`.

#### j30 `vision/out` → house pack

**Ops ownership:** the reComputer j30 (and `~/vision`) is run by the
**robotics** project. Anvil does not own on-device logging policy.

**Storage:** Orin Nano-class devices fill up fast if frame dumps are left
running. Anvil’s contract is **pull a small snapshot → convert on lab → train
on lab**. Do not use the robot as an Anvil run dir, HF cache, or multi-hour
JPEG ring buffer unless robotics explicitly budgets the space.

If a **short** capture already exists under `~/vision/out` (llm RGB+see text,
loop escalations, optional live_pull):

```bash
# lab machine only — no secrets/IPs in git
# Prefer size-capped rsync; prune on-device after pull (robotics policy).
rsync -avz --max-size=5m <robot-user>@<robot-host>:~/vision/out/ ./j30-out/

python scripts/j30_vision_out_to_pack.py --source ./j30-out --out ./house_pack
python scripts/robot_pack_smoke.py --pack ./house_pack --steps 20
# train/export on forge; ship a small GGUF/adapter back — not the reverse
```

**Do not:**

- Enable continuous high-rate frame logging “for Anvil” without a retention
  cap (count, MB, or TTL) owned by robotics
- Leave Anvil `run_dir` / media CAS / model weights on the j30
- Commit house frames or LAN credentials

Default capture style for packs: **low res** (e.g. 320×240), **few frames per
episode**, text captions/detections over raw video.

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
| Offline robot SFT + action tokens (smol edge) | **Phase 4.A** done |
| House pack + edge export + agent dogfood | **Phase 4.A/C** + agentic path |
| Vision on-policy RL | **Phase 4.B** open |

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
- [x] `run_vlm_sft` on ≥1k placed lab rows with observe (forge 2026-07-29: `expert_v0_1k.jsonl` + VLM-3B; train microbatch via `--max-train`)  
- [~] Held-out qualitative sample after export (probes recorded; text quality still operator-judged)  
- [x] Multi-hour resume path + lab smoke — `checkpoint_every`/`resume` + `fake_multi_hour_resume` / `multi_hour` profile  
- [x] Scale ladder demo smoke (`fake_scale_ladder`); forge: `scale_ladder.py --no-demo`


## Out of scope (for now)

- Full OXE download in CI  
- Hosting third-party weights/datasets in this repo  
- Claiming OpenVLA wire compatibility  
