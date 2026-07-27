# Expert-v0 smoke checklist

**Bar:** place domain data → train under live observe → stop with evidence → export PEFT.  
**Roadmap:** [`roadmap.md`](roadmap.md) §Expert-v0 · **workflow:** [`design.md`](design.md) §1.0

This is the repeatable “ship one specialist” path for humans and agents.

---

## One command (laptop / CI)

```bash
python scripts/expert_v0_smoke.py --endpoint fake:// --max-rows 20 --steps 3 --run-id expert-v0-demo
```

Expect:

| Artifact | Where |
|----------|--------|
| JSONL (cas:// refs) | `~/.anvil/anvil_jsonl/expert-v0-demo.jsonl` (or `--output-jsonl`) |
| Metrics | `$ANVIL_OBSERVE_ROOT/expert-v0-demo/metrics.jsonl` |
| Probes | `…/probes.jsonl` (held-out greedy samples) |
| Adapter | `~/.anvil/runs/expert-v0-demo/adapter` (PEFT dir on real backends; fake writes stub) |

Open `/observe/expert-v0-demo` with `anvil-web`, or MCP:

```text
anvil_observe_list
anvil_observe_metrics(run_id="expert-v0-demo", tail=20)
anvil_observe_probes(run_id="expert-v0-demo", tail=12)
```

---

## Lab (forge) — real domain slice

1. **Extract** Bridge (or other) into an **episode pack** (see [`datasets-robotics.md`](datasets-robotics.md)).  
2. **Convert** (or let the smoke script convert):

```bash
python scripts/convert_robotics_corpus.py \
  --source /mnt/data/datasets/bridge_v2/episode_pack \
  --media-root /mnt/data/anvil-media \
  --output /mnt/data/datasets/anvil_jsonl/bridge_1k.jsonl \
  --dataset bridge_v2 \
  --max-rows 1000 \
  --frames-per-episode 4 \
  --license "BridgeData V2 — check RAIL terms before redistribute"
```

3. **Train + observe + export:**

```bash
export ANVIL_OBSERVE_ROOT=/mnt/data/anvil-observe
python scripts/expert_v0_smoke.py \
  --endpoint local:// \
  --skip-convert \
  --output-jsonl /mnt/data/datasets/anvil_jsonl/bridge_1k.jsonl \
  --media-root /mnt/data/anvil-media \
  --model /mnt/data/models/Qwen2.5-VL-3B-Instruct \
  --steps 50 \
  --run-id expert-v0-bridge-1k \
  --export /mnt/data/anvil-runs/expert-v0-bridge-1k \
  --holdout 4
```

4. **Judge:** loss curve on `/observe/…`; probe text vs targets; keep export if probes acceptable.

---

## Checklist (pass/fail)

- [ ] Data placed: JSONL rows with `cas://` images only (no blobs in git)  
- [ ] `metrics.jsonl` has `job=vlm_sft` steps with `loss`, `wall_time_s`, `n_image_refs`  
- [ ] Held-out `probes.jsonl` records appear every `probe_every` steps  
- [ ] Observe index lists the run; SSE stream works (or MCP tail works)  
- [ ] Adapter export path exists (PEFT files on `local://`)  
- [ ] Human or agent can state why they would stop or continue (transcript)

---

## What Expert-v0 does **not** require

- Full Bridge download in CI  
- DPO / multi-method ladder (Expert-v1)  
- Multi-hour checkpoint/resume (Expert-v2)  
- Jetson export (Path: edge)  

---

## Agent loop (minimal)

1. `anvil_health`  
2. Run smoke (CLI) or create run via control plane  
3. Poll `anvil_observe_metrics` / `anvil_observe_probes`  
4. Classify healthy / noisy / cliff / broken ([`agent-context.md`](agent-context.md))  
5. Export when probes acceptable; stop if broken  
