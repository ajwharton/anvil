# Lab live smokes (not GitHub CI)

Periodic **host/GPU** checks for Anvil on forge/hammer. CI stays thin (`[dev,web]`,
no torch). This suite is opt-in on lab machines.

## Profiles

| Profile | Contents | Typical use |
|---------|----------|-------------|
| `quick` | host + **fake:** GRPO/SFT early-stop, DPO observe, meta-exec, southward, expert_v0 path, RL queue | **run often** (laptop / pre-push) |
| `multi_hour` | quick + **scale ladder demo** + throughput defaults + multi-hour **resume contract** | Expert-v2 ops (laptop-safe) |
| `nightly` | quick + real GRPO early-stop + 2-stage RL queue on 1.5B | daily cron on forge |
| `full` | multi_hour + real GRPO/queue + VLM SFT (3B) | weekly / release gate |

## Run

```bash
# on forge
cd /mnt/data/anvil
source /mnt/data/anvil-venv/bin/activate
./scripts/run_lab_smokes.sh nightly

# or explicit
python scripts/lab_smokes.py --profile nightly --endpoint local:// --list
```

Reports land in `/mnt/data/anvil-runs/lab-smokes/run-<ts>/report.json` and
`latest.json`.

## Cron (example)

```cron
# forge crontab — 03:15 daily, nightly profile
15 3 * * * /mnt/data/anvil/scripts/run_lab_smokes.sh nightly >>/mnt/data/anvil-runs/lab-smokes/cron.log 2>&1
```

## Individual smokes

| Name | Asserts |
|------|---------|
| `fake_early_stop` | GRPO constant reward → stop before budget |
| `fake_sft_early_stop` | SFT production plateau early-stop |
| `fake_dpo_observe` | DPO metrics.jsonl (job=dpo, length_bias) |
| `fake_meta_exec` | meta-recipe stage advance on early_stop signal |
| `fake_southward` | advantage_collapse detector + southward events |
| `fake_expert_v0` | full expert_v0_smoke (convert→train→southward→meta→promote) |
| `fake_meta_live_runners` | meta-exec default SFT→GRPO live runners |
| `fake_scale_ladder` | demo convert+train rungs 1k/5k/50k (tiny demo_rows) |
| `fake_throughput_defaults` | shape×pattern batch/ckpt defaults coherent |
| `fake_multi_hour_resume` | checkpoint mid-budget → resume → finish |
| `fake_rl_queue` | ceiling stage → next stage, same adapter |
| `grpo_early_stop_local` | real 1.5B GRPO; early-stop if saturated |
| `rl_queue_local` | 2 hard stages; advance on ceiling |
| `vlm_sft_local` | VLM CE few steps with pixels |

**Habit:** `python scripts/lab_smokes.py --profile quick` before PRs and after recipe/observe changes.  
**Expert-v2 ops:** `python scripts/lab_smokes.py --profile multi_hour` · forge real ladder: `docs/scale-ladder.md`.

GitHub Actions **must not** call this script (heavy + private models path).
