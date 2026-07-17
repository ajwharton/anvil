# Scripts

| Script | Purpose |
|--------|---------|
| `pull_base_model.py` | SSH to **forge/hammer** and pull HF bases onto lab NVMe (`/mnt/data/models`) |
| `jlens_spike.py` | **J0** forge spike: fit/apply Jacobian lens on small dense math probes; GO/NO-GO for Phase 2.5 panel |

Weights stay on lab hosts and out of git. Default vision pull: Qwen2.5-VL-3B. See [docs/models.md](../docs/models.md).

J-Lens spike writeup template: [docs/spikes/jlens-math.md](../docs/spikes/jlens-math.md).

