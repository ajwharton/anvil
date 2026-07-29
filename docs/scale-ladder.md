# Scale ladder (Expert-v2)

Climb **1k → 5k → 50k+** converted rows on forge without inventing knobs each time.

| Rung | `max_rows` | Train steps (guide) | `checkpoint_every` | Notes |
|------|------------|---------------------|--------------------|-------|
| **1k** | 1 000 | 50 | 25 | First Bridge/Robo2VLM slice; loss + probes must look sane |
| **5k** | 5 000 | 150 | 25 | Tens of minutes on 3B VLM Spark |
| **50k** | 50 000 | 500 | 50 | Multi-hour; require resume; fewer frames/episode |

Throughput defaults (batch/rank/LR/ckpt) live in `anvil.recipes.throughput` and are
shape×pattern scoped (`dense_vlm`/`vlm_sft`, `dense_lm`/`sft_chat`, GRPO, DPO).

## Demo (CI / laptop)

Same code path, tiny `demo_rows` (32 / 64 / 128):

```bash
python scripts/scale_ladder.py --demo --work-dir /tmp/anvil-ladder
# or one rung
python scripts/scale_ladder.py --demo --rung 1k
python scripts/lab_smokes.py --profile multi_hour
```

## Forge (real corpus)

Episode pack must already exist on NVMe (converter does **not** download OXE):

```bash
# 1k
python scripts/scale_ladder.py --no-demo --rung 1k \
  --source /mnt/data/datasets/bridge_v2/episode_pack \
  --media-root /mnt/data/anvil-media \
  --work-dir /mnt/data/anvil-runs/scale-ladder \
  --endpoint local://

# then 5k / 50k when 1k observe looks healthy
python scripts/scale_ladder.py --no-demo --rung 5k --source ... --work-dir ...
python scripts/scale_ladder.py --no-demo --rung 50k --source ... --work-dir ...
```

Convert-only (JSONL + CAS, no train):

```bash
python scripts/scale_ladder.py --no-demo --rung 50k --source ... --convert-only
```

Or the lower-level converter:

```bash
python scripts/convert_robotics_corpus.py \
  --source /mnt/data/datasets/bridge_v2/episode_pack \
  --media-root /mnt/data/anvil-media \
  --output /mnt/data/datasets/anvil_jsonl/bridge_50k.jsonl \
  --max-rows 50000 --frames-per-episode 2
```

## Multi-hour ops

- Always set `run_dir` + `checkpoint_every` (throughput defaults supply values).
- Restart with `resume=True` and the **same total** `steps` budget.
- Lab smoke `fake_multi_hour_resume` exercises that contract on `fake://`.
- Real multi-hour wall time = forge `scale_ladder.py --no-demo --rung 50k` + cron.

See also: `docs/datasets-robotics.md`, `docs/lab-smokes.md`, `anvil.recipes.checkpoint`.
