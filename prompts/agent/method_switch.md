# Method switch suggestions (conservative)

These are **default playbook edges**, not laws. Prefer data/quality fixes over
method thrashing. Make **one** primary change per intervention.

## Signatures → ideas

| Signature | Possible next step |
|-----------|-------------------|
| Offline preference (DPO-like) reward proxy flat for many steps; probes worse | Try on-policy RL recipe (GRPO/IS) if verifiable reward exists; else more/better preference data or SFT recovery |
| On-policy RL: **advantage collapse** (within-group reward std ≈ 0) | Pause RL; diversify prompts/rewards; lower LR; short SFT recovery on clean data |
| On-policy: **IS mean_ratio** drifts far from 1 | Push/sync adapter to sample worker more often; reduce steps between sync; check logprob alignment |
| Probe quality down, reward up | Reward hacking—stop optim; redesign reward or add probe gates; do not “train more” |
| Loss ↓ but eval/probes garbage after SFT | Check train/sample renderer consistency; data labels; LR too high |
| LoRA not moving loss | Rank/LR/targets; model too small; wrong modality freeze |

## Prefer

- Recipe/plan switch over inventing knobs.  
- Export adapter before destructive switches.  
- Document why in the transcript.

## Avoid

- Switching methods every few steps.  
- Forcing blocked MoE/edge recipes without human policy.  
- Full-parameter FT as a default escape hatch.
