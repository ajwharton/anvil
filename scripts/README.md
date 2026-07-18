# Scripts

| Script | Purpose |
|--------|---------|
| `pull_base_model.py` | SSH to **forge/hammer** and pull HF bases onto lab NVMe (`/mnt/data/models`) |
| `jlens_spike.py` | J-Lens forge CLI — fit/apply Jacobian lens, protocol `solve` (J0 gate GO 2026-07-18); emits `jlens.jsonl` |
| `vlm_smoke.py` | **P3.3** VLM SFT smoke: CAS frame + `run_vlm_sft` (`fake://` or `local://`) |

Weights stay on lab hosts and out of git. Default vision pull: Qwen2.5-VL-3B. See [docs/models.md](../docs/models.md).

Agent control: `anvil mcp` / `anvil agent` — [docs/agentic-control.md](../docs/agentic-control.md).

