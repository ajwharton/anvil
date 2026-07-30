"""Robot offline recipe — trajectories → text-tokenized actions → VLM/SFT.

Phase 4.A: productize offline policy learning under the four verbs without a
new optimizer. Continuous actions are spelled as text bins (OpenVLA-style) or
decimal continuous strings; training is CE LoRA via :func:`run_vlm_sft` /
:func:`run_sft`.

**Memory-constrained robot default:** SmolVLM ~256M
(``HuggingFaceTB/SmolVLM-256M-Instruct``). Freeze vision; small rank/seq.
Never actuate from raw samples without a supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from anvil.protocol.action_tokens import ActionTokenizer, default_edge_tokenizer
from anvil.protocol.messages import Example, Message
from anvil.protocol.trajectory import Trajectory, TrajectoryStep, trajectories_to_examples
from anvil.recipes.model_card import ModelCardFacts, inspect_base_model
from anvil.recipes.profiles import JobPattern, RecipePlan, plan_recipe
from anvil.recipes.sft import SFTResult, examples_to_data, run_sft
from anvil.render.text import ToyTextRenderer

# Severe on-robot memory: 256M-class VLM, not lab 3B.
DEFAULT_ROBOT_BASE = "HuggingFaceTB/SmolVLM-256M-Instruct"


def build_plan(
    base_model: str = DEFAULT_ROBOT_BASE,
    *,
    card: ModelCardFacts | None = None,
    fetch_remote: bool = True,
    **overrides: Any,
) -> RecipePlan:
    card = card or inspect_base_model(base_model, fetch_remote=fetch_remote)
    load_id = card.local_path or base_model or card.repo_id
    return plan_recipe(
        base_model=load_id,
        pattern=JobPattern.ROBOT_OFFLINE,
        shape=card.shape,
        overrides=overrides or None,
        card=card,
    )


def toy_robot_trajectories() -> list[Trajectory]:
    """Synthetic tabletop-style episodes (no real pixels) for CI / fake://."""
    dig = "c" * 64
    ref = f"cas://sha256/{dig}.png"
    return [
        Trajectory(
            episode_id="toy-ep0",
            meta={"instruction": "pick up the blue cube", "source": "toy"},
            steps=(
                TrajectoryStep(
                    observation_refs=(ref,),
                    instruction="pick up the blue cube",
                    action=[0.1, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0],
                    reward=0.0,
                ),
                TrajectoryStep(
                    observation_refs=(ref,),
                    instruction="pick up the blue cube",
                    action=[0.1, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0],
                    reward=1.0,
                    done=True,
                ),
            ),
        ),
        Trajectory(
            episode_id="toy-ep1",
            meta={"instruction": "place the cube on the plate", "source": "toy"},
            steps=(
                TrajectoryStep(
                    observation_refs=(ref,),
                    instruction="place the cube on the plate",
                    action=[-0.05, 0.1, 0.08, 0.0, 0.0, 0.0, 0.0],
                    reward=0.0,
                ),
                TrajectoryStep(
                    observation_refs=(ref,),
                    instruction="place the cube on the plate",
                    action=[-0.05, 0.1, 0.02, 0.0, 0.0, 0.0, 1.0],
                    reward=1.0,
                    done=True,
                ),
            ),
        ),
    ]


def split_heldout_episodes(
    trajectories: Sequence[Trajectory],
    *,
    heldout_fraction: float = 0.25,
    min_heldout: int = 1,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Split by whole episode (not step) for held-out success-proxy probes."""
    trs = list(trajectories)
    if len(trs) < 2:
        return trs, []
    n_hold = max(min_heldout, int(round(len(trs) * heldout_fraction)))
    n_hold = min(n_hold, len(trs) - 1)
    train = trs[:-n_hold]
    hold = trs[-n_hold:]
    return train, hold


def trajectories_to_robot_examples(
    trajectories: Sequence[Trajectory],
    *,
    tokenizer: ActionTokenizer | None = None,
    include_all_frames: bool = False,
    require_images: bool = True,
) -> list[Example]:
    """Trajectories → multimodal Examples with text-tokenized action targets."""
    tok = tokenizer if tokenizer is not None else default_edge_tokenizer()
    return trajectories_to_examples(
        trajectories,
        action_tokenizer=tok,
        include_all_frames=include_all_frames,
        require_images=require_images,
    )


@dataclass
class RobotOfflineResult:
    """Outcome of :func:`run_robot_offline` (wraps underlying SFT result)."""

    sft: SFTResult
    n_train_examples: int
    n_probe_examples: int
    n_train_episodes: int
    n_heldout_episodes: int
    action_tokenizer: dict[str, Any] = field(default_factory=dict)
    base_model: str = DEFAULT_ROBOT_BASE

    @property
    def adapter_id(self) -> str:
        return self.sft.adapter_id

    @property
    def steps_run(self) -> int:
        return self.sft.steps_run

    @property
    def losses(self) -> list[float]:
        return self.sft.losses

    @property
    def run_dir(self) -> str | None:
        return self.sft.run_dir

    @property
    def early_stop_reason(self) -> str | None:
        return self.sft.early_stop_reason

    @property
    def export_path(self) -> str | None:
        return self.sft.export_path


def run_robot_offline(
    *,
    base_model: str = DEFAULT_ROBOT_BASE,
    trajectories: Sequence[Trajectory] | None = None,
    examples: Sequence[Example] | None = None,
    steps: int = 3,
    endpoint: str = "fake://",
    export_dir: str | None = None,
    fetch_remote: bool = True,
    overrides: dict[str, Any] | None = None,
    media_store: Any | None = None,
    renderer: Any | None = None,
    run_dir: str | None = None,
    action_tokenizer: ActionTokenizer | None = None,
    heldout_fraction: float = 0.25,
    probe_every: int = 1,
    early_stop: bool | None = None,
    early_stop_mode: str = "production",
    early_stop_patience: int | None = None,
    early_stop_rel_eps: float | None = None,
    stop_on_southward: bool | None = None,
    southward_min_steps: int = 8,
    service_client: Any | None = None,
    training_client: Any | None = None,
    close_clients: bool = True,
    checkpoint_every: int | None = None,
    resume: bool = False,
    text_only: bool = False,
) -> RobotOfflineResult:
    """Offline robot policy learning (trajectory SFT with action tokens).

    - Default base is **SmolVLM-256M** for memory-constrained robots.
    - ``trajectories``: preferred input; actions tokenized via
      :class:`~anvil.protocol.action_tokens.ActionTokenizer`.
    - Held-out **episodes** become probes (success-proxy: model samples action
      text under live adapter).
    - ``examples``: optional pre-built Examples (skips trajectory convert).
    - ``text_only``: use text SFT path (no vision) for SmolLM-class bases.
    - Metrics / resume / southward: same SSOT as VLM SFT.
    """
    tok = action_tokenizer if action_tokenizer is not None else default_edge_tokenizer()
    n_train_eps = 0
    n_hold_eps = 0
    probes: list[Example] = []

    if examples is not None:
        train_exs = list(examples)
        n_train_eps = len({ex.meta.get("episode_id") for ex in train_exs if ex.meta.get("episode_id")})
    else:
        trs = list(trajectories) if trajectories is not None else toy_robot_trajectories()
        train_trs, hold_trs = split_heldout_episodes(
            trs, heldout_fraction=heldout_fraction
        )
        n_train_eps = len(train_trs)
        n_hold_eps = len(hold_trs)
        train_exs = trajectories_to_robot_examples(
            train_trs,
            tokenizer=tok,
            require_images=not text_only,
        )
        if hold_trs:
            probes = trajectories_to_robot_examples(
                hold_trs,
                tokenizer=tok,
                require_images=not text_only,
            )
        if not train_exs:
            raise ValueError(
                "no train examples from trajectories "
                "(need instruction + action + observation refs)"
            )

    # Mild edge-friendly overrides when caller did not set them.
    ov = dict(overrides or {})
    ov.setdefault("rank", 8)
    ov.setdefault("learning_rate", 2e-4)

    card = inspect_base_model(base_model, fetch_remote=fetch_remote)
    pattern = JobPattern.SFT_CHAT if text_only else JobPattern.ROBOT_OFFLINE
    plan = plan_recipe(
        base_model=card.local_path or base_model or card.repo_id,
        pattern=pattern,
        shape=card.shape,
        overrides=ov or None,
        card=card,
    )

    if renderer is None:
        if (
            not text_only
            and media_store is not None
            and (endpoint.startswith("local://") or endpoint.startswith("http"))
        ):
            from anvil.render.vlm import HFVLMRenderer

            renderer = HFVLMRenderer(plan.base_model, media_store)
        else:
            renderer = ToyTextRenderer()

    _ = examples_to_data(train_exs[:1], renderer=renderer)

    sft_kwargs: dict[str, Any] = dict(
        base_model=plan.base_model,
        examples=train_exs,
        steps=steps,
        endpoint=endpoint,
        export_dir=export_dir,
        plan=plan,
        renderer=renderer,
        run_dir=run_dir,
        job="robot_offline",
        probes=probes or None,
        probe_every=probe_every,
        early_stop=early_stop,
        early_stop_mode=early_stop_mode,
        early_stop_patience=early_stop_patience,
        stop_on_southward=stop_on_southward,
        southward_min_steps=southward_min_steps,
        service_client=service_client,
        training_client=training_client,
        close_clients=close_clients,
        checkpoint_every=checkpoint_every,
        resume=resume,
    )
    if early_stop_rel_eps is not None:
        sft_kwargs["early_stop_rel_eps"] = early_stop_rel_eps
    sft = run_sft(**sft_kwargs)

    return RobotOfflineResult(
        sft=sft,
        n_train_examples=len(train_exs),
        n_probe_examples=len(probes),
        n_train_episodes=n_train_eps,
        n_heldout_episodes=n_hold_eps,
        action_tokenizer=tok.to_public(),
        base_model=base_model,
    )


def toy_text_action_example() -> Example:
    """Single non-vision example (action bins as assistant text)."""
    tok = default_edge_tokenizer()
    target = tok.encode([0.0, 0.5, -0.25, 0.0, 0.0, 0.0, 1.0])
    return Example(
        messages=(
            Message(role="user", content="Move to grasp pose."),
            Message(role="assistant", content=target),
        ),
        meta={"source": "toy_text_action"},
    )
