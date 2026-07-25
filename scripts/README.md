# Scripts

| Script | Purpose |
|--------|---------|
| `pull_base_model.py` | SSH to **forge/hammer** and pull HF bases onto lab NVMe (`/mnt/data/models`) |
| `jlens_spike.py` | J-Lens forge CLI — fit/apply Jacobian lens; protocol `solve` + digit/rank scoring; emits `jlens.jsonl`. **Spike parked** 2026-07-19 (see [docs/spikes/jlens-math.md](../docs/spikes/jlens-math.md)). |
| `run_jlens_7b_mixed.sh` | Overnight recipe: fit 7B lens with **mixed WikiText+math** then `solve` apply (resume helper; needs `datasets` + `jlens` on the lab venv) |
| `vlm_smoke.py` | **P3.3** VLM SFT smoke: CAS frame + `run_vlm_sft` (`fake://` or `local://`) |
| `robot_vlm_sft_demo.py` | Short LoRA SFT on Qwen2.5-VL-3B with synthetic tabletop or **LeRobot** frames (ffmpeg) |

Weights stay on lab hosts and out of git. Default vision pull: Qwen2.5-VL-3B. See [docs/models.md](../docs/models.md).

## Vision SFT on forge (smoke)

```bash
# on forge, anvil-venv active
cd /mnt/data/anvil
python scripts/vlm_smoke.py \
  --endpoint local:// \
  --model /mnt/data/models/Qwen2.5-VL-3B-Instruct \
  --media-root /mnt/data/anvil-media \
  --steps 5 \
  --export /mnt/data/anvil-runs/vlm-smoke-out

# real robot frames from LeRobot (downloads mp4, extracts PNGs)
python scripts/robot_vlm_sft_demo.py \
  --source lerobot \
  --lerobot-repo lerobot/aloha_sim_transfer_cube_human \
  --n-examples 8 --steps 20 \
  --export /mnt/data/anvil-runs/robot-vlm-lerobot
```

Requires lab deps: `torch`, `transformers`, `peft`, `pillow`, `torchvision`, `accelerate` (and `ffmpeg` for LeRobot video).

## J-Lens lab deps (optional / parked)

On forge (or any CUDA host with the anvil venv):

```bash
source /mnt/data/anvil-venv/bin/activate
pip install 'git+https://github.com/anthropics/jacobian-lens.git'
pip install 'datasets>=2.14'   # required for --fit-corpus mixed (WikiText-103)
```

Without `datasets`, `jlens_spike.py --fit-corpus mixed` **silently falls back to math-only** prompts (no WikiText). The overnight runner fails fast if `datasets` is missing.

Resume overnight 7B mixed fit+solve:

```bash
nohup bash scripts/run_jlens_7b_mixed.sh &
tail -f /mnt/data/anvil/results/jlens-solve-7b-mixed/run.log
```

Agent control: `anvil mcp` / `anvil agent` — [docs/agentic-control.md](../docs/agentic-control.md).
