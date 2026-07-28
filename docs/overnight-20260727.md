# Overnight handoff — 2026-07-27/28

## PR (merge when ready)

**https://github.com/ajwharton/anvil/pull/39** — CI green  
Branch: `feat/overnight-expert-v0-earlystop-recipes`

| Item | Status |
|------|--------|
| SFT/VLM early-stop production vs calibration | done + tests |
| Personal recipe book + promote-from-run | done + tests |
| LeRobot → JSONL builder | done |
| expert_v0_smoke dogfood flags | done |

## Forge lab run (real frames)

| | |
|--|--|
| **Run ID** | `expert-v0-lerobot-20260727` |
| **Data** | LeRobot `pusht` — 64 frames → CAS JSONL (not synthetic solids) |
| **Train** | 28 examples + 4 holdout probes, rank 16, Qwen2.5-VL-3B |
| **Stop** | **production** early-stop patience **40** (cap 5000 steps) |
| **Promote** | `vlm-lerobot-pusht-v1` → `/mnt/data/anvil-recipes/` |
| **Log** | `/mnt/data/anvil-runs/expert-v0-lerobot-20260727.log` |
| **Metrics** | `/mnt/data/anvil-observe/expert-v0-lerobot-20260727/metrics.jsonl` |
| **UI** | `http://forge:7600/observe/expert-v0-lerobot-20260727` |

### Morning checks

```bash
ssh forge 'tail -30 /mnt/data/anvil-runs/expert-v0-lerobot-20260727.log'
ssh forge 'wc -l /mnt/data/anvil-observe/expert-v0-lerobot-20260727/metrics.jsonl'
ssh forge 'ps -p $(cat /mnt/data/anvil-runs/expert-v0-lerobot-20260727.pid) || echo finished'
ssh forge 'ls -la /mnt/data/anvil-recipes/; cat /mnt/data/anvil-recipes/vlm-lerobot-pusht-v1.json 2>/dev/null | head -40'
```

Expect either **early_stop: loss_plateau_patience_40** well before 5000, or full budget if loss keeps improving. Recipe JSON should exist after clean exit.

### Monitor

A 5‑minute poll monitor was left running in the agent session on this forge run.
