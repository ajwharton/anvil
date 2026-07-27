# Scripts

| Script | Purpose |
|--------|---------|
| `pull_base_model.py` | SSH to **forge/hammer** and pull HF bases onto lab NVMe (`/mnt/data/models`) |
| `jlens_spike.py` | J-Lens forge CLI — fit/apply Jacobian lens; protocol `solve` + digit/rank scoring; emits `jlens.jsonl`. **Spike parked** 2026-07-19 (see [docs/spikes/jlens-math.md](../docs/spikes/jlens-math.md)). |
| `run_jlens_7b_mixed.sh` | Overnight recipe: fit 7B lens with **mixed WikiText+math** then `solve` apply (resume helper; needs `datasets` + `jlens` on the lab venv) |
| `vlm_smoke.py` | **P3.3/P3.6** VLM SFT smoke: CAS frame + `run_vlm_sft`; `--run-id` → live `/observe` |
| `convert_robotics_corpus.py` | **P3.5 / 3.B** episode_pack or path JSONL → CAS + Anvil JSONL (resume, subsample) |
| `expert_v0_smoke.py` | **Expert-v0** place → VLM SFT + observe + probes + export |
| `robot_vlm_sft_demo.py` | Short LoRA SFT on Qwen2.5-VL-3B; synthetic/LeRobot; optional `--run-id` observe |
| `grpo_observe_demo.py` | **Productized GRPO** → `ANVIL_OBSERVE_ROOT/<run_id>/metrics.jsonl`; live charts at `/observe/<run_id>` |
| `grpo_recipe_queue.py` | Multi-stage RL curriculum; advances on early-stop |
| `lab_smokes.py` | **Lab live smoke suite** (not GitHub CI) — quick / nightly / full |
| `run_lab_smokes.sh` | Forge/cron entrypoint for `lab_smokes.py` |

Weights stay on lab hosts and out of git. Default vision pull: Qwen2.5-VL-3B. See [docs/models.md](../docs/models.md).

## Lab live smokes (periodic, not CI)

```bash
# laptop — seconds, no GPU
python scripts/lab_smokes.py --profile quick

# forge — real GRPO early-stop + 2-stage queue
./scripts/run_lab_smokes.sh nightly

# weekly / release
./scripts/run_lab_smokes.sh full
```

Reports: `/mnt/data/anvil-runs/lab-smokes/latest.json`. Full notes: [`docs/lab-smokes.md`](../docs/lab-smokes.md).

Cron (forge example)::

```cron
15 3 * * * /mnt/data/anvil/scripts/run_lab_smokes.sh nightly >>/mnt/data/anvil-runs/lab-smokes/cron.log 2>&1
```


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

## Productized GRPO + live web observe

```bash
# terminal A — forge trainer (writes metrics under observe root)
export ANVIL_OBSERVE_ROOT=/mnt/data/anvil-observe
cd /mnt/data/anvil
python scripts/grpo_observe_demo.py \
  --endpoint local:// \
  --model /mnt/data/models/qwen2.5-1.5b-instruct \
  --problem hard \
  --run-id grpo-hard-demo \
  --steps 40 --group-size 8 \
  --attach-wait 8
# hard default = 15*8+7=127 (partial hit rate on 1.5B; not already saturated)
# Early stop (default): abandon after 8 consecutive dead-signal steps
# (reward ceiling/floor + group_std≈0) so overnight runs don't burn power flatlined.

# terminal B — same observe root, control plane + SSE
ANVIL_OBSERVE_ROOT=/mnt/data/anvil-observe anvil-web --host 0.0.0.0 --port 7600
# open http://<host>:7600/observe          (index)
# open http://<host>:7600/observe/grpo-hard-demo  (live reward / probes)
```

Control-plane home also lists disk observe runs and links into the SSE debugger.
Observe UI shows an **EARLY STOP** banner when `metrics.jsonl` gets an `early_stop` event.

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
