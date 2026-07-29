# Org self-host notes (Expert-v2)

Anvil is **sovereign**: you own weights, data, recipes, and observe artifacts.
No phone-home. This note is the ops profile for a lab or org forge (e.g. DGX
Spark, multi-GPU box) without putting secrets in git.

## Layout (suggested)

```text
/mnt/data/
  models/              # base snapshots (Qwen2.5-VL-3B, …)
  datasets/            # episode packs, anvil_jsonl/
  anvil-media/         # LocalMediaStore CAS
  anvil-observe/       # metrics/probes per run_id
  anvil-runs/          # exports, scale-ladder, smoke reports
  anvil-recipes/       # personal book  → ANVIL_RECIPE_BOOK
  anvil-org-packs/     # shared org packs → ANVIL_ORG_RECIPE_PACK
  anvil/               # git checkout (code only)
  anvil-venv/          # python env
```

## Environment (no secrets in repo)

| Variable | Purpose |
|----------|---------|
| `ANVIL_RECIPE_BOOK` | Personal write root for promoted recipes |
| `ANVIL_ORG_RECIPE_PACK` | Read-only org pack dir (merged into list/suggest) |
| `ANVIL_OBSERVE_ROOT` | Live metrics/probes SSOT for web/MCP |
| `ANVIL_ENDPOINT` | Default train endpoint (`local://`, `fake://`, …) |
| `ANVIL_TOKEN` | Optional Bearer for `anvil serve` / web on shared hosts |

Do **not** commit host IPs, tokens, or private model paths into the public tree.
Document lab paths in private runbooks if needed.

## Org recipe packs

Pack layout:

```text
my-org-pack/
  manifest.json          # name, version, recipes: [ids]
  recipes/*.json         # BookRecipe JSON (stop_policy.experience optional)
```

```bash
# install into personal book (copies + tag org_pack)
python -c "
from anvil.recipes.book import install_org_pack, RecipeBook
print([r.id for r in install_org_pack('packs/demo-org-qwen-vl')])
"

# or point suggest/list at the pack without copying
export ANVIL_ORG_RECIPE_PACK=/mnt/data/anvil-org-packs/demo-org-qwen-vl
```

**Experience → patience:** production recipes with `stop_policy.patience` (or
patience embedded in `last_early_stop_reason`) are aggregated (median) in
`suggest_for_model(...)[\"experience_priors\"]`. Calibration-mode recipes are
ignored so overshoot nights do not poison production priors.

## Multi-GPU / multi-worker reality (honest)

- Default path: **single-process** LocalBackend train + optional vLLM sample
  worker(s). Right shape for Spark-class boxes.
- **Horizontal sample:** `run_grpo(sample_endpoints=[…])` →
  :class:`~anvil.workers.pool.SampleWorkerPool` (round-robin sample, fan-out
  adapter snapshot). See [`multi-worker.md`](multi-worker.md).
- Full data-parallel **train** shards are still out of scope; prefer vertical
  scale (batch, seq, rank) + resume first.
- Keep observe on one `ANVIL_OBSERVE_ROOT` NFS/local disk so agents share SSOT.

## Governance

- Public Anvil git: **no secrets**, no private dataset dumps.
- Org packs may live outside the public repo; promote only recipes you intend
  to share.
- Gate force (`force=True`) remains audited (`docs/agent-context.md`).

## Related

- Recipes / book: [`recipes.md`](recipes.md)  
- Scale ladder: [`scale-ladder.md`](scale-ladder.md)  
- Lab smokes: [`lab-smokes.md`](lab-smokes.md)  
- Development process: [`development-process.md`](development-process.md)  
