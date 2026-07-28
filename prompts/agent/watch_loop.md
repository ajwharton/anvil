# Watch loop (operator habit)

For each open training run, cycle:

## 1. Read state

- Run status, step, recipe/loss, sample backend.  
- Latest metrics window (not only the last point).  
- Latest probe completions + scores if present.  
- Any tripwire flags (e.g. advantage collapse).

## 2. Classify (pick one)

| Class | Rough meaning |
|-------|----------------|
| **Healthy** | Reward/probes improving or stable; no collapse; IS ratio near on-policy if RL; SFT/DPO loss still improving past abs/rel eps |
| **Noisy** | High variance; need more steps or more groups before judging |
| **Cliff** | Clear stall or regression with a known signature (see method_switch); DPO length_bias exploding while loss drops |
| **Broken** | Tool errors, empty batches, sampler out of sync—fix infra first |

### Job field (`metrics.jsonl`)

| `job` | Primary live signals |
|-------|----------------------|
| `grpo` | reward_mean, group_reward_std_mean, is_mean_ratio, probes |
| `sft` / `vlm_sft` | loss, n_image_refs, wall, probes, early_stop |
| `dpo` | loss, n_pairs, length_bias, margin proxy, early_stop |

## 3. Act or wait

- **Healthy / noisy** — usually wait; maybe tighten probe_every for eyes.  
- **Cliff** — use `method_switch.md`; prefer one change at a time.  
- **Broken** — stop or pause; report; do not switch recipes to “fix” infra.

## 4. Log

Record: class, evidence (metric names + rough values), action or “no-op.”

## Probe text matters

Scalars lie. If reward rises while probes go off-rails (format hacking, refusal,
garbage), treat as **cliff** even if reward looks good.
