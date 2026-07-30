"""Train worker entrypoint notes (Expert-v2 multi-worker layout).

Training is still **single-process** LocalBackend — the four verbs + LoRA.
Horizontal scale for RL is on the **sample** side
(:class:`~anvil.workers.pool.SampleWorkerPool` / vLLM workers).

Launch train side::

    # process A — train host
    anvil serve --backend local --host 0.0.0.0 --port 8740

    # processes B..N — sample hosts (vLLM)
    anvil serve --backend vllm-sample --model $MODEL --host 0.0.0.0 --port 8741

    # orchestrator (often co-located with train)
    run_grpo(
        endpoint="http://train:8740",
        sample_endpoints=["http://s0:8741", "http://s1:8742"],
        sync_every=1,
        ...
    )

Shared filesystem (or equivalent) is required so snapshot paths from
``snapshot_for_sample`` are readable by sample workers.
"""

from __future__ import annotations

WORKER_ROLE = "train"

__all__ = ["WORKER_ROLE"]
