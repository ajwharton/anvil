"""Default live stage runners for meta-recipe execution.

Wires :func:`~anvil.recipes.meta_exec.run_meta_recipe` to real SFT / VLM SFT /
GRPO / DPO / export stages (not injectables). Shared LoRA across stages when
patterns stay on the same train client (SFT family + DPO); GRPO opens its own
session unless ``reuse_grpo_client`` is set later.

Smoke/CLI default: ``make_default_runner(endpoint=\"fake://\", ...)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from anvil.client.service import ServiceClient
from anvil.protocol.messages import Example, Message, TextPart
from anvil.protocol.types import LoraTargets
from anvil.recipes.dpo import PreferencePair, run_dpo
from anvil.recipes.grpo import run_grpo
from anvil.recipes.meta import MetaStage
from anvil.recipes.meta_exec import StageRunResult
from anvil.recipes.robot_offline import run_robot_offline, toy_robot_trajectories
from anvil.recipes.sft import run_sft
from anvil.recipes.vlm_sft import run_vlm_sft, toy_vlm_examples


def _toy_sft_examples() -> list[Example]:
    return [
        Example(
            messages=(
                Message(role="user", content=(TextPart(text="2+2?"),)),
                Message(role="assistant", content=(TextPart(text="4"),)),
            )
        ),
        Example(
            messages=(
                Message(role="user", content=(TextPart(text="3+3?"),)),
                Message(role="assistant", content=(TextPart(text="6"),)),
            )
        ),
    ]


def _toy_dpo_pairs() -> list[PreferencePair]:
    return [
        PreferencePair(
            prompt="2+2?",
            preferred="4",
            rejected="five because five is bigger than four",
        ),
    ]


def normalize_pattern(stage: MetaStage) -> str:
    """Resolve job pattern string from stage.pattern or recipe_id heuristics."""
    if stage.pattern:
        return str(stage.pattern).lower().strip()
    rid = (stage.recipe_id or "").lower()
    if "grpo" in rid or "rl" in rid or "verifiable" in rid:
        return "rl_verifiable"
    if "dpo" in rid or "pref" in rid:
        return "preference_dpo"
    if "vlm" in rid or "vision" in rid or "edge" in rid:
        return "vlm_sft"
    if "export" in rid or stage.id == "export":
        return "export"
    return "sft_chat"


def signal_from_early_stop(reason: str | None, *, default: str = "complete") -> str:
    """Map recipe early_stop_reason → meta edge signal."""
    if not reason:
        return default
    r = str(reason)
    if r.startswith("early_stop:"):
        return r
    if r.startswith("southward:"):
        return f"early_stop:{r}"
    if r.startswith("dpo_"):
        return f"early_stop:{r}"
    return f"early_stop:{r}"


@dataclass
class DefaultRunnerConfig:
    """Knobs for :func:`make_default_runner`."""

    endpoint: str = "fake://"
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    run_dir: str | Path | None = None
    # Per-pattern step budgets (fake-friendly defaults)
    sft_steps: int = 20
    vlm_steps: int = 20
    grpo_steps: int = 12
    dpo_steps: int = 15
    early_stop_patience: int = 12
    grpo_patience: int = 6
    group_size: int = 4
    # Optional data injection (else toy defaults)
    sft_examples: Sequence[Example] | None = None
    vlm_examples: Sequence[Example] | None = None
    dpo_pairs: Sequence[PreferencePair] | None = None
    grpo_prompts: Sequence[Sequence[int]] | None = None
    # Export directory root (stage export uses run_dir/export or this)
    export_root: str | Path | None = None
    fetch_remote: bool = False
    stop_on_southward: bool = False
    share_sft_client: bool = True
    extra_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class DefaultMetaRunner:
    """Callable stage runner that executes live recipes."""

    config: DefaultRunnerConfig
    _svc: ServiceClient | None = field(default=None, init=False, repr=False)
    _tc: Any = field(default=None, init=False, repr=False)
    _adapter_id: str | None = field(default=None, init=False)
    last_export_path: str | None = field(default=None, init=False)
    history: list[dict[str, Any]] = field(default_factory=list, init=False)

    def close(self) -> None:
        if self._svc is not None:
            self._svc.close()
            self._svc = None
            self._tc = None

    def __enter__(self) -> DefaultMetaRunner:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _stage_dir(self, stage: MetaStage) -> str | None:
        if self.config.run_dir is None:
            return None
        return str(Path(self.config.run_dir) / stage.id)

    def _ensure_sft_client(self, base_model: str) -> tuple[ServiceClient, Any]:
        if self._svc is not None and self._tc is not None and self.config.share_sft_client:
            return self._svc, self._tc
        from anvil.recipes.profiles import JobPattern, plan_recipe

        plan = plan_recipe(
            base_model=base_model,
            pattern=JobPattern.SFT_CHAT,
            overrides=self.config.extra_overrides or None,
        )
        k = plan.as_knobs()
        svc = ServiceClient(endpoint=self.config.endpoint)
        tc = svc.create_lora_training_client(
            base_model=plan.base_model,
            rank=k["rank"],
            alpha=k.get("alpha"),
            modalities=k["modalities"],
            lora_targets=LoraTargets(
                language=k["language_lora"],
                vision_encoder=k["vision_encoder_lora"],
                mm_projector=k["mm_projector_lora"],
            ),
        )
        self._svc = svc
        self._tc = tc
        self._adapter_id = str(tc.adapter_id)
        return svc, tc

    def __call__(self, stage: MetaStage, *, step_index: int) -> StageRunResult:
        pattern = normalize_pattern(stage)
        stage_dir = self._stage_dir(stage)
        cfg = self.config
        base = cfg.base_model

        try:
            if pattern in {"export", "save", "done"}:
                return self._run_export(stage, stage_dir)

            if pattern in {"vlm_sft", "vlm_classifier"}:
                return self._run_vlm(stage, stage_dir, base)

            if pattern in {"robot_offline"}:
                return self._run_robot_offline(stage, stage_dir, base)

            if pattern in {"rl_verifiable", "grpo", "rl"}:
                return self._run_grpo(stage, stage_dir, base)

            if pattern in {"preference_dpo", "dpo"}:
                return self._run_dpo(stage, stage_dir, base)

            # default: text SFT
            return self._run_sft(stage, stage_dir, base)
        except Exception as e:  # noqa: BLE001 — stage failure becomes meta halt
            self.history.append(
                {"stage": stage.id, "pattern": pattern, "error": str(e)}
            )
            return StageRunResult(
                signal=f"error:{type(e).__name__}",
                halted=True,
                metrics={"error": str(e), "pattern": pattern},
            )

    def _run_sft(
        self, stage: MetaStage, stage_dir: str | None, base: str
    ) -> StageRunResult:
        cfg = self.config
        examples = list(cfg.sft_examples) if cfg.sft_examples is not None else _toy_sft_examples()
        svc, tc = self._ensure_sft_client(base)
        res = run_sft(
            base_model=base,
            examples=examples,
            steps=cfg.sft_steps,
            endpoint=cfg.endpoint,
            run_dir=stage_dir,
            job="sft",
            early_stop=True,
            early_stop_mode="production",
            early_stop_patience=cfg.early_stop_patience,
            stop_on_southward=cfg.stop_on_southward,
            service_client=svc,
            training_client=tc,
            close_clients=False,
            overrides=cfg.extra_overrides or None,
        )
        self._adapter_id = res.adapter_id
        if res.export_path:
            self.last_export_path = res.export_path
        sig = signal_from_early_stop(res.early_stop_reason, default="sft_complete")
        metrics = {
            "pattern": "sft_chat",
            "steps_run": res.steps_run,
            "adapter_id": res.adapter_id,
            "early_stop_reason": res.early_stop_reason,
        }
        self.history.append({"stage": stage.id, **metrics})
        return StageRunResult(signal=sig, metrics=metrics)

    def _run_vlm(
        self, stage: MetaStage, stage_dir: str | None, base: str
    ) -> StageRunResult:
        cfg = self.config
        examples = (
            list(cfg.vlm_examples) if cfg.vlm_examples is not None else toy_vlm_examples()
        )
        # Prefer shared client when base looks text-like on fake; still works for VLM labels
        svc, tc = self._ensure_sft_client(base)
        res = run_vlm_sft(
            base_model=base,
            examples=examples,
            steps=cfg.vlm_steps,
            endpoint=cfg.endpoint,
            fetch_remote=cfg.fetch_remote,
            run_dir=stage_dir,
            early_stop=True,
            early_stop_mode="production",
            early_stop_patience=cfg.early_stop_patience,
            stop_on_southward=cfg.stop_on_southward,
            service_client=svc,
            training_client=tc,
            close_clients=False,
            overrides=cfg.extra_overrides or None,
        )
        self._adapter_id = res.adapter_id
        if res.export_path:
            self.last_export_path = res.export_path
        sig = signal_from_early_stop(res.early_stop_reason, default="vlm_sft_complete")
        metrics = {
            "pattern": "vlm_sft",
            "steps_run": res.steps_run,
            "adapter_id": res.adapter_id,
            "early_stop_reason": res.early_stop_reason,
        }
        self.history.append({"stage": stage.id, **metrics})
        return StageRunResult(signal=sig, metrics=metrics)

    def _run_robot_offline(
        self, stage: MetaStage, stage_dir: str | None, base: str
    ) -> StageRunResult:
        cfg = self.config
        svc, tc = self._ensure_sft_client(base)
        # Prefer explicit VLM examples if provided; else synthetic trajectories.
        if cfg.vlm_examples is not None:
            res = run_robot_offline(
                base_model=base,
                examples=list(cfg.vlm_examples),
                steps=cfg.vlm_steps,
                endpoint=cfg.endpoint,
                fetch_remote=cfg.fetch_remote,
                run_dir=stage_dir,
                early_stop=True,
                early_stop_mode="production",
                early_stop_patience=cfg.early_stop_patience,
                stop_on_southward=cfg.stop_on_southward,
                service_client=svc,
                training_client=tc,
                close_clients=False,
                overrides=cfg.extra_overrides or None,
            )
        else:
            res = run_robot_offline(
                base_model=base,
                trajectories=toy_robot_trajectories(),
                steps=cfg.vlm_steps,
                endpoint=cfg.endpoint,
                fetch_remote=cfg.fetch_remote,
                run_dir=stage_dir,
                early_stop=True,
                early_stop_mode="production",
                early_stop_patience=cfg.early_stop_patience,
                stop_on_southward=cfg.stop_on_southward,
                service_client=svc,
                training_client=tc,
                close_clients=False,
                overrides=cfg.extra_overrides or None,
            )
        self._adapter_id = res.adapter_id
        if res.export_path:
            self.last_export_path = res.export_path
        sig = signal_from_early_stop(res.early_stop_reason, default="robot_offline_complete")
        metrics = {
            "pattern": "robot_offline",
            "steps_run": res.steps_run,
            "adapter_id": res.adapter_id,
            "early_stop_reason": res.early_stop_reason,
            "n_train_examples": res.n_train_examples,
            "n_heldout_episodes": res.n_heldout_episodes,
        }
        self.history.append({"stage": stage.id, **metrics})
        return StageRunResult(signal=sig, metrics=metrics)

    def _run_grpo(
        self, stage: MetaStage, stage_dir: str | None, base: str
    ) -> StageRunResult:
        cfg = self.config
        res = run_grpo(
            base_model=base,
            endpoint=cfg.endpoint,
            steps=cfg.grpo_steps,
            group_size=cfg.group_size,
            run_dir=stage_dir,
            prompts=list(cfg.grpo_prompts) if cfg.grpo_prompts is not None else None,
            early_stop=True,
            early_stop_patience=cfg.grpo_patience,
            stop_on_southward=cfg.stop_on_southward,
            overrides=cfg.extra_overrides or None,
        )
        self._adapter_id = res.adapter_id
        sig = signal_from_early_stop(res.early_stop_reason, default="grpo_complete")
        # Floor-style reasons should halt the meta ladder
        halted = bool(
            res.early_stop_reason
            and any(
                x in res.early_stop_reason
                for x in ("floor", "reward_floor")
            )
        )
        metrics = {
            "pattern": "rl_verifiable",
            "steps_run": res.steps_run,
            "adapter_id": res.adapter_id,
            "early_stop_reason": res.early_stop_reason,
            "sync_count": res.sync_count,
        }
        self.history.append({"stage": stage.id, **metrics})
        return StageRunResult(signal=sig, halted=halted, metrics=metrics)

    def _run_dpo(
        self, stage: MetaStage, stage_dir: str | None, base: str
    ) -> StageRunResult:
        cfg = self.config
        pairs = list(cfg.dpo_pairs) if cfg.dpo_pairs is not None else _toy_dpo_pairs()
        # Reuse the shared SFT client so DPO continues the same LoRA adapter
        # (SFT → preference on one ladder), matching the other stage runners.
        svc, tc = self._ensure_sft_client(base)
        res = run_dpo(
            base_model=base,
            pairs=pairs,
            steps=cfg.dpo_steps,
            endpoint=cfg.endpoint,
            run_dir=stage_dir,
            early_stop=True,
            early_stop_mode="production",
            early_stop_patience=cfg.early_stop_patience,
            stop_on_southward=cfg.stop_on_southward,
            overrides=cfg.extra_overrides or None,
            service_client=svc,
            training_client=tc,
            close_clients=False,
        )
        self._adapter_id = res.adapter_id
        if res.export_path:
            self.last_export_path = res.export_path
        sig = signal_from_early_stop(res.early_stop_reason, default="dpo_complete")
        metrics = {
            "pattern": "preference_dpo",
            "steps_run": res.steps_run,
            "adapter_id": res.adapter_id,
            "early_stop_reason": res.early_stop_reason,
            "mean_length_bias": res.mean_length_bias,
        }
        self.history.append({"stage": stage.id, **metrics})
        return StageRunResult(signal=sig, metrics=metrics)

    def _run_export(self, stage: MetaStage, stage_dir: str | None) -> StageRunResult:
        cfg = self.config
        export_dir = cfg.export_root or (
            Path(cfg.run_dir) / "export" if cfg.run_dir else None
        )
        path: str | None = self.last_export_path
        if export_dir is not None and self._tc is not None:
            try:
                ref = self._tc.export_adapter(str(export_dir), format="peft")
                path = ref.path
                self.last_export_path = path
            except Exception as e:  # noqa: BLE001
                return StageRunResult(
                    signal="export_failed",
                    halted=True,
                    metrics={"error": str(e)},
                )
        metrics = {
            "pattern": "export",
            "export_path": path,
            "adapter_id": self._adapter_id,
        }
        self.history.append({"stage": stage.id, **metrics})
        return StageRunResult(signal="export_done", metrics=metrics)


def make_default_runner(config: DefaultRunnerConfig | None = None) -> DefaultMetaRunner:
    """Factory used by CLI and lab smokes."""
    return DefaultMetaRunner(config=config or DefaultRunnerConfig())


def run_meta_with_defaults(
    meta: Any,
    *,
    config: DefaultRunnerConfig | None = None,
    max_stages: int | None = None,
    start_stage_id: str | None = None,
) -> Any:
    """Execute a meta-recipe with the default live runners (context-managed)."""
    from anvil.recipes.meta_exec import run_meta_recipe

    cfg = config or DefaultRunnerConfig()
    # Prefer meta run_dir under config
    runner = make_default_runner(cfg)
    try:
        return run_meta_recipe(
            meta,
            runner,
            run_dir=cfg.run_dir,
            max_stages=max_stages,
            start_stage_id=start_stage_id,
        )
    finally:
        runner.close()


__all__ = [
    "DefaultMetaRunner",
    "DefaultRunnerConfig",
    "make_default_runner",
    "normalize_pattern",
    "run_meta_with_defaults",
    "signal_from_early_stop",
]
