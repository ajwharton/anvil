"""Multi-worker sample pool + GRPO sample_endpoints (Expert-v2)."""

from __future__ import annotations

from anvil.backends.fake import FakeBackend
from anvil.client.service import ServiceClient
from anvil.observe.metrics import METRICS_FILENAME, read_jsonl
from anvil.protocol.types import ModelInput, SamplingParams
from anvil.recipes.grpo import push_adapter_snapshot, run_grpo
from anvil.workers.pool import MultiWorkerLayout, SampleWorkerPool


def test_pool_round_robin_and_fanout_snapshot(tmp_path):
    w0 = FakeBackend(root=tmp_path / "w0")
    w1 = FakeBackend(root=tmp_path / "w1")
    pool = SampleWorkerPool([w0, w1], label="test-pool")
    assert pool.n_workers == 2

    svc = ServiceClient(endpoint=f"fake://{tmp_path / 'train'}")
    tc = svc.create_lora_training_client(base_model="toy", rank=4)
    ref = push_adapter_snapshot(tc, pool, name="sync0")
    assert pool.sync_calls == 1
    assert ref.path
    # both workers hot-loaded
    assert tc.adapter_id.value in w0._hot_adapters
    assert tc.adapter_id.value in w1._hot_adapters

    for _ in range(4):
        pool.sample(
            base_model="toy",
            adapter_id=tc.adapter_id,
            prompt=ModelInput.from_ints([1, 2, 3]),
            sampling_params=SamplingParams(max_tokens=4, temperature=0.0, seed=0),
            num_samples=1,
        )
    assert pool.sample_calls == 4
    assert pool.stats()["n_workers"] == 2
    svc.close()


def test_grpo_with_two_sample_backends(tmp_path):
    """Inject pool as sample_backend — dual Fake sample workers."""
    train_root = tmp_path / "train"
    s0 = FakeBackend(root=tmp_path / "s0")
    s1 = FakeBackend(root=tmp_path / "s1")
    pool = SampleWorkerPool([s0, s1])
    run_dir = tmp_path / "grpo-pool"
    res = run_grpo(
        endpoint=f"fake://{train_root}",
        steps=4,
        group_size=2,
        sample_backend=pool,
        sync_every=1,
        run_dir=str(run_dir),
        early_stop=False,
        stop_on_southward=False,
    )
    assert res.steps_run == 4
    assert res.sync_count == 4
    assert pool.sync_calls == 4
    assert pool.sample_calls >= 4  # groups + maybe probes
    steps = [
        s
        for s in read_jsonl(run_dir / METRICS_FILENAME)
        if s.get("type") == "step"
    ]
    assert all(s.get("adapter_synced") for s in steps)
    assert all("pool" in (s.get("sample_endpoint") or "") or s.get("sample_endpoint") for s in steps)


def test_multi_worker_layout_hints():
    layout = MultiWorkerLayout(
        train_endpoint="http://train:8740",
        sample_endpoints=("http://s0:8741", "http://s1:8742"),
    )
    assert layout.to_public()["n_sample_workers"] == 2
    hints = "\n".join(layout.launch_hints())
    assert "vllm-sample" in hints
    assert "8741" in hints
    assert "sample_endpoints" in hints


def test_pool_requires_snapshot_loader(tmp_path):
    class _NoSnap:
        pass

    try:
        SampleWorkerPool([_NoSnap()])  # type: ignore[list-item]
        raise AssertionError("expected TypeError")
    except TypeError as e:
        assert "SnapshotLoader" in str(e)
