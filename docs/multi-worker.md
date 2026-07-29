# Multi-worker train / sample (Expert-v2)

When **generation** is the wall (GRPO / on-policy RL), scale **sample workers**
horizontally. Training stays a single LoRA session (LocalBackend); sample is
a pool of vLLM (or Fake) workers that share adapter snapshots.

```text
                 ┌──────────────────┐
                 │  train process   │  LocalBackend / anvil serve --backend local
                 │  forward_backward│
                 │  optim_step      │
                 └────────┬─────────┘
                          │ snapshot_for_sample + load_snapshot (every sync_every)
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
     sample-0         sample-1       sample-N     vllm-sample workers
     (round-robin sample / compute_logprobs)
```

## API

```python
from anvil.recipes.grpo import run_grpo

run_grpo(
    endpoint="http://train-host:8740",           # or local:// on same box
    sample_endpoints=[
        "http://sample-a:8741",
        "http://sample-b:8742",
    ],
    sync_every=1,
    steps=100,
    ...
)
```

- **One** `sample_endpoint` — classic Tier-1 single worker (unchanged).  
- **Many** `sample_endpoints` — :class:`~anvil.workers.pool.SampleWorkerPool`
  (round-robin sample, fan-out snapshot).  
- Inject `sample_backend=SampleWorkerPool([...])` in tests.

## Launch sketch

```bash
# train node
anvil serve --backend local --host 0.0.0.0 --port 8740

# sample nodes (one process each; needs vllm + shared FS for snapshot paths)
anvil serve --backend vllm-sample --model /mnt/data/models/qwen2.5-1.5b-instruct \
  --host 0.0.0.0 --port 8741
# ... second worker on :8742
```

Shared filesystem (NFS / same host paths) is required so PEFT snapshot dirs from
the train node are readable by sample workers.

Python helper for ops docs:

```python
from anvil.workers.pool import MultiWorkerLayout
print("\n".join(MultiWorkerLayout(
    "http://train:8740",
    ("http://s0:8741", "http://s1:8742"),
).launch_hints()))
```

## What this is not

- **Not** multi-GPU data-parallel SFT shards (still open if ever needed).  
- **Not** automatic job scheduling / k8s — you bring process supervision.  
- Prefer vertical scale (batch, rank, resume) before multi-worker train.

## Tests

`tests/test_sample_worker_pool.py` — dual FakeBackend pool + GRPO sync.

## Related

- Sample worker: `anvil/workers/sample.py`  
- GRPO adapter sync: `anvil.recipes.grpo.push_adapter_snapshot`  
- Self-host layout: [`org-self-host.md`](org-self-host.md)  
