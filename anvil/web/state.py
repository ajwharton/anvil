"""In-process run store for the Anvil web UI (Phase 0 control plane)."""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from anvil import __version__
from anvil.backends.fake import FakeBackend
from anvil.client.service import ServiceClient
from anvil.protocol.types import (
    AdamParams,
    Datum,
    ExportFormat,
    ModelInput,
    SamplingParams,
)

DEFAULT_BASE_MODELS = (
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3.5-4B",
    "microsoft/Phi-4-mini-instruct",
)

LOSS_CHOICES = (
    "cross_entropy",
    "importance_sampling",
    "ppo",
    "dpo",
)


@dataclass
class RunKnobs:
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    rank: int = 32
    alpha: int | None = None
    learning_rate: float = 1e-4
    loss_fn: str = "cross_entropy"
    modalities: list[str] = field(default_factory=lambda: ["text", "image"])
    max_steps: int = 100
    batch_size: int = 4
    seq_len: int = 64
    temperature: float = 0.7
    max_tokens: int = 64
    vision_encoder_lora: bool = False
    mm_projector_lora: bool = True
    language_lora: bool = True
    # Phase 2.5 RL debugger / sample-train split
    probe_every: int = 1
    sync_every: int = 1
    sample_endpoint: str = ""
    sample_adapter_id: str = ""
    write_metrics: bool = True


@dataclass
class StepPoint:
    step: int
    loss: float
    t: float


@dataclass
class RunRecord:
    run_id: str
    name: str
    knobs: RunKnobs
    status: str = "created"  # created | running | paused | completed | failed | exported
    adapter_id: str = ""
    step: int = 0
    last_loss: float | None = None
    history: list[StepPoint] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    export_path: str | None = None
    error: str | None = None

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]
        self.updated_at = time.time()

    def to_public(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status,
            "adapter_id": self.adapter_id,
            "step": self.step,
            "last_loss": self.last_loss,
            "knobs": asdict(self.knobs),
            "history": [asdict(h) for h in self.history[-200:]],
            "logs": self.logs[-80:],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "export_path": self.export_path,
            "error": self.error,
        }


class RunStore:
    """Thread-safe store of training runs backed by FakeBackend (GPU later)."""

    def __init__(self, *, models_root: str | None = None, export_root: str | None = None) -> None:
        self._lock = threading.RLock()
        self.models_root = Path(
            models_root
            or os.environ.get("ANVIL_MODELS_ROOT", "/mnt/data/models")
        )
        self.export_root = Path(
            export_root
            or os.environ.get("ANVIL_EXPORT_ROOT", str(Path.home() / ".anvil" / "exports"))
        )
        self.export_root.mkdir(parents=True, exist_ok=True)
        fake_root = Path(os.environ.get("ANVIL_FAKE_ROOT", str(Path.home() / ".anvil" / "fake")))
        self._backend = FakeBackend(root=fake_root)
        self._svc = ServiceClient(endpoint="fake://", backend=self._backend)
        self._runs: dict[str, RunRecord] = {}
        self._clients: dict[str, Any] = {}  # run_id -> TrainingClient
        self.host = os.environ.get("ANVIL_HOST_LABEL", os.uname().nodename)
        self.spark_dashboard_urls = {
            "forge": os.environ.get("SPARK_DASHBOARD_FORGE", "http://192.168.100.162:3000"),
            "hammer": os.environ.get("SPARK_DASHBOARD_HAMMER", "http://192.168.100.92:3000"),
        }

    # --- inventory ---------------------------------------------------------

    def list_local_models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        root = self.models_root
        if root.is_dir():
            for p in sorted(root.iterdir()):
                if p.name.startswith("."):
                    continue
                if p.is_dir() or p.suffix in {".gguf", ".bin"}:
                    size = _dir_size(p) if p.is_dir() else p.stat().st_size
                    out.append(
                        {
                            "name": p.name,
                            "path": str(p),
                            "size_bytes": size,
                            "source": "models_root",
                        }
                    )
        # Always surface reference HF ids even if not on disk yet
        known = {m["name"] for m in out}
        for mid in DEFAULT_BASE_MODELS:
            short = mid.split("/")[-1]
            if short not in known and mid not in known:
                out.append(
                    {
                        "name": mid,
                        "path": None,
                        "size_bytes": None,
                        "source": "catalog",
                    }
                )
        return out

    def defaults(self) -> dict[str, Any]:
        return {
            "knobs": asdict(RunKnobs()),
            "loss_choices": list(LOSS_CHOICES),
            "base_models": list(DEFAULT_BASE_MODELS),
            "models_root": str(self.models_root),
            "export_root": str(self.export_root),
            "host": self.host,
            "version": __version__,
            "backend": self._backend.name,
            "spark_dashboard": self.spark_dashboard_urls,
            "rl_knobs": {
                "probe_every": {
                    "default": 1,
                    "min": 1,
                    "help": "Greedy live-policy probes every K steps (metrics/probes.jsonl)",
                },
                "sync_every": {
                    "default": 1,
                    "min": 1,
                    "help": "Push train LoRA → sample worker load_snapshot every K steps (Tier 1)",
                },
                "sample_endpoint": {
                    "default": "",
                    "help": "Sample worker URL (e.g. http://host:8741). Empty = Tier 0 in-process",
                },
                "sample_adapter_id": {
                    "default": "",
                    "help": "Adapter id on the sample worker (default: train adapter id)",
                },
                "write_metrics": {
                    "default": True,
                    "help": "Emit metrics.jsonl / probes.jsonl for /observe/{run_id}",
                },
            },
        }

    def overview(self) -> dict[str, Any]:
        with self._lock:
            runs = [r.to_public() for r in sorted(self._runs.values(), key=lambda x: -x.created_at)]
            active = sum(1 for r in self._runs.values() if r.status == "running")
            completed = sum(1 for r in self._runs.values() if r.status in {"completed", "exported"})
        return {
            "host": self.host,
            "version": __version__,
            "backend": self._backend.name,
            "active_runs": active,
            "completed_runs": completed,
            "total_runs": len(runs),
            "models": self.list_local_models(),
            "runs": runs,
            "spark_dashboard": self.spark_dashboard_urls,
            "ts": time.time(),
        }

    # --- runs --------------------------------------------------------------

    def create_run(
        self,
        name: str | None,
        knobs: dict[str, Any] | None = None,
        *,
        pattern: str | None = None,
        shape: str | None = None,
        rationale: list[str] | None = None,
    ) -> RunRecord:
        raw = {**asdict(RunKnobs()), **(knobs or {})}
        # Ignore unknown keys so the UI can grow without 500s
        known = {f.name for f in fields(RunKnobs)}
        k = RunKnobs(**{key: raw[key] for key in known if key in raw})
        run_id = f"run-{uuid.uuid4().hex[:10]}"
        suffix = f"-{pattern}" if pattern else f"-r{k.rank}"
        rec = RunRecord(
            run_id=run_id,
            name=name or f"{k.base_model.split('/')[-1]}{suffix}",
            knobs=k,
        )
        if pattern:
            rec.log(f"pattern={pattern} shape={shape or 'n/a'}")
        if rationale:
            for line in rationale[:8]:
                rec.log(f"recipe: {line}")
        from anvil.protocol.types import LoraTargets

        tc = self._svc.create_lora_training_client(
            base_model=k.base_model,
            rank=k.rank,
            alpha=k.alpha,
            modalities=k.modalities,
            lora_targets=LoraTargets(
                language=k.language_lora,
                vision_encoder=k.vision_encoder_lora,
                mm_projector=k.mm_projector_lora,
            ),
        )
        rec.adapter_id = str(tc.adapter_id)
        rec.status = "created"
        rec.log(f"created adapter {rec.adapter_id} base={k.base_model} rank={k.rank}")
        if k.sample_endpoint:
            rec.log(
                f"rl sample_endpoint={k.sample_endpoint} sync_every={k.sync_every} "
                f"probe_every={k.probe_every}"
            )
        elif k.loss_fn in {"importance_sampling", "ppo"}:
            rec.log(
                f"rl Tier-0 (in-process sample) sync_every={k.sync_every} "
                f"probe_every={k.probe_every}"
            )
        if k.write_metrics:
            rec.log(f"observe → /observe/{run_id} (when run_dir is set by trainer)")
        with self._lock:
            self._runs[run_id] = rec
            self._clients[run_id] = tc
        return rec

    def get_run(self, run_id: str) -> RunRecord:
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as e:
                raise KeyError(f"unknown run: {run_id}") from e

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: -r.created_at)

    def pause_run(self, run_id: str) -> RunRecord:
        """Mark run paused (agent live-control)."""
        rec = self.get_run(run_id)
        if rec.status in {"completed", "failed", "exported"}:
            raise RuntimeError(f"run is {rec.status}; cannot pause")
        rec.status = "paused"
        rec.log("paused (agent/API)")
        return rec

    def resume_run(self, run_id: str) -> RunRecord:
        rec = self.get_run(run_id)
        if rec.status not in {"paused", "created"}:
            raise RuntimeError(f"run is {rec.status}; cannot resume")
        rec.status = "running"
        rec.log("resumed (agent/API)")
        return rec

    def patch_knobs(self, run_id: str, updates: dict[str, Any]) -> RunRecord:
        """Patch run knobs in place (live control). Audited via log line."""
        from dataclasses import fields as dc_fields

        rec = self.get_run(run_id)
        if rec.status in {"completed", "failed", "exported"}:
            raise RuntimeError(f"run is {rec.status}; knobs frozen")
        known = {f.name for f in dc_fields(RunKnobs)}
        applied: dict[str, Any] = {}
        for key, val in updates.items():
            if key not in known:
                continue
            setattr(rec.knobs, key, val)
            applied[key] = val
        if not applied:
            raise ValueError("no valid knobs to patch")
        rec.log(f"knobs patched: {applied}")
        rec.updated_at = time.time()
        return rec

    def train_steps(self, run_id: str, n_steps: int | None = None) -> RunRecord:
        """Run n toy SFT steps on the fake backend (real PEFT in Phase 1)."""
        rec = self.get_run(run_id)
        with self._lock:
            tc = self._clients[run_id]
        if rec.status in {"completed", "failed"}:
            raise RuntimeError(f"run is {rec.status}")
        if rec.status == "paused":
            raise RuntimeError("run is paused; resume before training")

        steps = n_steps if n_steps is not None else 1
        remaining = max(0, rec.knobs.max_steps - rec.step)
        steps = min(steps, remaining) if remaining else steps
        if steps <= 0:
            rec.status = "completed"
            rec.log("max_steps reached")
            return rec

        rec.status = "running"
        rec.log(f"training {steps} step(s) loss={rec.knobs.loss_fn}")
        k = rec.knobs
        for _ in range(steps):
            data = [_synthetic_batch(k.batch_size, k.seq_len)]
            # flatten: one Datum per synthetic example
            flat = data[0]
            fb = tc.forward_backward(flat, loss_fn=k.loss_fn).result()
            opt = tc.optim_step(AdamParams(learning_rate=k.learning_rate)).result()
            rec.step = opt.step
            rec.last_loss = fb.loss
            rec.history.append(StepPoint(step=rec.step, loss=fb.loss, t=time.time()))
            if len(rec.history) > 500:
                rec.history = rec.history[-500:]
            rec.updated_at = time.time()
            if rec.step >= k.max_steps:
                rec.status = "completed"
                rec.log(f"completed at step {rec.step} loss={fb.loss:.4f}")
                break
        else:
            rec.status = "paused"
            rec.log(f"paused at step {rec.step} loss={rec.last_loss:.4f}")
        return rec

    def sample(self, run_id: str) -> dict[str, Any]:
        rec = self.get_run(run_id)
        with self._lock:
            tc = self._clients[run_id]
        sc = tc.save_weights_and_get_sampling_client(name=f"ui-{rec.step}")
        prompt = ModelInput.from_ints(list(range(10, 10 + 16)))
        result = sc.sample(
            prompt,
            SamplingParams(
                max_tokens=rec.knobs.max_tokens,
                temperature=rec.knobs.temperature,
                seed=rec.step,
            ),
            num_samples=1,
        ).result()
        tokens = list(result.sequences[0].tokens)
        rec.log(f"sample step={rec.step} n_tokens={len(tokens)}")
        return {
            "run_id": run_id,
            "step": rec.step,
            "tokens": tokens,
            "n_tokens": len(tokens),
            "stop_reason": result.sequences[0].stop_reason,
        }

    def export_run(self, run_id: str, fmt: str = "peft") -> dict[str, Any]:
        rec = self.get_run(run_id)
        with self._lock:
            tc = self._clients[run_id]
        out = self.export_root / run_id / fmt
        try:
            export_fmt = ExportFormat(fmt)
        except ValueError:
            export_fmt = ExportFormat.PEFT
        result = tc.export_adapter(str(out), format=export_fmt)
        rec.export_path = result.path
        rec.status = "exported"
        rec.log(f"exported {fmt} → {result.path}")
        return {"run_id": run_id, "path": result.path, "format": result.format.value}

    def save_checkpoint(self, run_id: str, name: str | None = None) -> dict[str, Any]:
        rec = self.get_run(run_id)
        with self._lock:
            tc = self._clients[run_id]
        ckpt = name or f"step-{rec.step}"
        ref = tc.save_state(ckpt)
        rec.log(f"checkpoint {ref.name} → {ref.path}")
        return {"name": ref.name, "path": ref.path, "kind": ref.kind}


def _synthetic_batch(batch_size: int, seq_len: int) -> list[Datum]:
    out: list[Datum] = []
    for b in range(batch_size):
        tokens = [1000 + (b * 17 + i * 3) % 500 for i in range(seq_len + 1)]
        weights = [0.0] * (seq_len // 3) + [1.0] * (seq_len - seq_len // 3)
        out.append(
            Datum(
                model_input=ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tokens[1:],
                    "weights": weights,
                    "logprobs": [-1.2] * seq_len,
                    "advantages": [0.5] * seq_len,
                },
            )
        )
    return out


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except OSError:
                    pass
    except OSError:
        return 0
    return total


# Singleton for the web process
_STORE: RunStore | None = None


def get_store() -> RunStore:
    global _STORE
    if _STORE is None:
        _STORE = RunStore()
    return _STORE
