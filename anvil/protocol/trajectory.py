"""Offline robot / vision trajectory format (Phase 3–4 bridge).

Episodes are sequences of steps with **media refs** for observations (and
optional action/reward fields for RL). They convert to multimodal
:class:`~anvil.protocol.messages.Example` rows for VLM SFT, or keep rewards
for on-policy/offline RL recipes later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from anvil.protocol.messages import Example, ImagePart, Message, TextPart


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One timestep: observation (+ optional language, action, reward)."""

    observation_refs: tuple[str, ...] = ()
    """Content-addressed image (or other media) refs for this step's obs."""

    instruction: str | None = None
    """Language goal / instruction active at this step (often constant per episode)."""

    action: Any | None = None
    """Policy action — free-form (list[float], dict, token string) until action schema locks."""

    reward: float | None = None
    done: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> dict[str, Any]:
        return {
            "observation_refs": list(self.observation_refs),
            "instruction": self.instruction,
            "action": self.action,
            "reward": self.reward,
            "done": self.done,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_public(cls, d: Mapping[str, Any]) -> TrajectoryStep:
        refs = d.get("observation_refs") or d.get("images") or ()
        return cls(
            observation_refs=tuple(str(r) for r in refs),
            instruction=None if d.get("instruction") is None else str(d["instruction"]),
            action=d.get("action"),
            reward=None if d.get("reward") is None else float(d["reward"]),
            done=bool(d.get("done", False)),
            meta=dict(d.get("meta") or {}),
        )


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One episode / trajectory (robot demo, sim rollout, human teleop, …)."""

    steps: tuple[TrajectoryStep, ...]
    episode_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            object.__setattr__(self, "steps", tuple(self.steps))

    def to_public(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "steps": [s.to_public() for s in self.steps],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_public(cls, d: Mapping[str, Any]) -> Trajectory:
        steps = tuple(TrajectoryStep.from_public(s) for s in d.get("steps", []))
        return cls(
            steps=steps,
            episode_id=None if d.get("episode_id") is None else str(d["episode_id"]),
            meta=dict(d.get("meta") or {}),
        )

    def total_reward(self) -> float:
        return float(sum(s.reward for s in self.steps if s.reward is not None))

    def all_image_refs(self) -> list[str]:
        out: list[str] = []
        for s in self.steps:
            out.extend(s.observation_refs)
        return out

    def to_vlm_sft_examples(
        self,
        *,
        response_key: str = "action",
        include_all_frames: bool = False,
        action_tokenizer: Any | None = None,
        require_images: bool = True,
    ) -> list[Example]:
        """Map trajectory steps → multimodal SFT Examples (image + instruction → text).

        Default: one example per step that has an instruction and a stringifiable
        action (or meta[response_key]). Frame set is that step's obs refs (or all
        episode frames if ``include_all_frames``).

        When ``action_tokenizer`` is set (see
        :class:`~anvil.protocol.action_tokens.ActionTokenizer`), vector actions
        become text-tokenized targets (bins or continuous).
        """
        examples: list[Example] = []
        episode_refs = self.all_image_refs() if include_all_frames else ()
        for i, step in enumerate(self.steps):
            instr = step.instruction or self.meta.get("instruction")
            if not instr:
                continue
            resp = step.meta.get(response_key, step.action)
            if resp is None:
                continue
            if action_tokenizer is not None:
                resp = action_tokenizer.encode(resp)
            elif not isinstance(resp, str):
                resp = str(resp)
            refs = episode_refs if include_all_frames else step.observation_refs
            if require_images and not refs:
                continue
            content: list[Any] = [TextPart(text=str(instr))]
            for ref in refs:
                content.append(ImagePart(ref=ref, detail="auto"))
            examples.append(
                Example(
                    messages=(
                        Message(role="user", content=tuple(content)),
                        Message(role="assistant", content=resp),
                    ),
                    meta={
                        "episode_id": self.episode_id,
                        "step": i,
                        "reward": step.reward,
                        "source": "trajectory",
                        **{k: v for k, v in self.meta.items() if k != "instruction"},
                    },
                )
            )
        return examples


def trajectories_to_examples(
    trajectories: Sequence[Trajectory],
    **kwargs: Any,
) -> list[Example]:
    out: list[Example] = []
    for tr in trajectories:
        out.extend(tr.to_vlm_sft_examples(**kwargs))
    return out
