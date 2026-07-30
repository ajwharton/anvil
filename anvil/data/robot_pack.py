"""In-house robot episode packs (j30 / edge vision loop → Anvil).

The Orin-class house robot logs **frames + optional detections/captions**
and, when teleop or scripted motion exists, **action vectors**. This module
turns a small on-disk pack into :class:`~anvil.protocol.trajectory.Trajectory`
objects and Anvil JSONL for ``run_robot_offline`` / VLM SFT.

Pack layout (episode pack, Bridge-like intermediate)::

    pack/
      ep_0001/
        meta.json     # instruction, actions[], captions[], detections[]
        frames/0000.jpg
        frames/0001.jpg
      ep_0002/
        ...

``meta.json`` keys (all optional except a language field)::

    language_instruction | instruction | task | caption
    actions | action_sequence   # list of vectors or strings
    captions                    # per-frame VLM text (j30 see-loop style)
    detections                  # per-frame [{label, conf}, ...]
    source, license, robot_id

No secrets, host IPs, or multi-GB blobs — frames only under a local media root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from anvil.data.convert import format_action_text
from anvil.media.store import LocalMediaStore
from anvil.protocol.action_tokens import ActionTokenizer, default_edge_tokenizer
from anvil.protocol.trajectory import Trajectory, TrajectoryStep

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class HousePackConfig:
    """Options for reading / writing an in-house robot pack."""

    source: Path
    """Root directory of the episode pack."""

    media_root: Path | None = None
    """If set, copy frames into LocalMediaStore and use cas:// refs."""

    action_scheme: str = "bins"
    """``bins`` (robot_offline default) or ``continuous``."""

    action_n_bins: int = 256
    action_decimals: int = 4
    dataset: str = "house_robot"
    robot_id: str = "edge"
    max_episodes: int | None = None
    frames_per_episode: int | None = None
    """Even subsample of frames (None = all)."""

    prefer_caption_as_instruction: bool = False
    """If no instruction, use first caption / detection summary."""


@dataclass
class HousePackResult:
    trajectories: list[Trajectory]
    n_episodes: int
    n_steps: int
    n_images: int
    dataset: str
    action_tokenizer: dict[str, Any] = field(default_factory=dict)


def _list_frame_paths(ep_dir: Path) -> list[Path]:
    for sub in ("frames", "images", "image", "rgb", "obs"):
        d = ep_dir / sub
        if d.is_dir():
            frames = sorted(
                p
                for p in d.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
            )
            if frames:
                return frames
    return sorted(
        p
        for p in ep_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )


def _subsample(frames: Sequence[Path], n: int | None) -> list[Path]:
    if n is None or n <= 0 or len(frames) <= n:
        return list(frames)
    if n == 1:
        return [frames[len(frames) // 2]]
    # evenly spaced including ends
    idxs = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in idxs]


def _load_meta(ep_dir: Path) -> dict[str, Any]:
    for name in ("meta.json", "episode.json", "info.json"):
        p = ep_dir / name
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _instruction(meta: Mapping[str, Any], *, prefer_caption: bool) -> str | None:
    for key in (
        "language_instruction",
        "instruction",
        "task",
        "goal",
        "prompt",
    ):
        if meta.get(key):
            return str(meta[key]).strip()
    if prefer_caption:
        caps = meta.get("captions")
        if isinstance(caps, list) and caps:
            return str(caps[0]).strip()
        if meta.get("caption"):
            return str(meta["caption"]).strip()
    return None


def _actions(meta: Mapping[str, Any]) -> list[Any]:
    for key in ("actions", "action", "action_sequence", "teleop"):
        v = meta.get(key)
        if isinstance(v, list):
            return list(v)
        if v is not None and not isinstance(v, list):
            return [v]
    return []


def _captions(meta: Mapping[str, Any], n: int) -> list[str | None]:
    caps = meta.get("captions")
    if isinstance(caps, list):
        out: list[str | None] = [str(c) if c is not None else None for c in caps]
        while len(out) < n:
            out.append(None)
        return out[:n]
    if meta.get("caption") is not None:
        return [str(meta["caption"])] * n
    return [None] * n


def _detection_summary(dets: Any) -> str | None:
    if not dets:
        return None
    if isinstance(dets, dict):
        label = dets.get("label") or dets.get("class")
        return str(label) if label else None
    if isinstance(dets, list):
        labels = []
        for d in dets:
            if isinstance(d, dict):
                lab = d.get("label") or d.get("class")
                if lab:
                    labels.append(str(lab))
            elif isinstance(d, str):
                labels.append(d)
        if labels:
            return "detect: " + ", ".join(labels[:8])
    return None


def iter_house_episodes(
    root: Path,
) -> Iterator[tuple[str, Path, dict[str, Any], list[Path]]]:
    """Yield ``(episode_id, ep_dir, meta, frame_paths)`` sorted by name."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"house pack root not found: {root}")
    eps = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    for ep_dir in eps:
        frames = _list_frame_paths(ep_dir)
        if not frames:
            continue
        yield ep_dir.name, ep_dir, _load_meta(ep_dir), frames


def house_pack_to_trajectories(
    cfg: HousePackConfig,
    *,
    tokenizer: ActionTokenizer | None = None,
) -> HousePackResult:
    """Load pack → list of Trajectory (media refs as paths or cas://)."""
    tok = tokenizer
    if tok is None and cfg.action_scheme == "bins":
        tok = ActionTokenizer(scheme="bins", n_bins=cfg.action_n_bins)
    elif tok is None:
        tok = ActionTokenizer(scheme="continuous", decimals=cfg.action_decimals)

    store: LocalMediaStore | None = None
    if cfg.media_root is not None:
        store = LocalMediaStore(cfg.media_root)

    trajectories: list[Trajectory] = []
    n_images = 0
    n_steps = 0
    n_ep = 0

    for ep_id, _ep_dir, meta, frames in iter_house_episodes(cfg.source):
        if cfg.max_episodes is not None and n_ep >= cfg.max_episodes:
            break
        frames = _subsample(frames, cfg.frames_per_episode)
        instr = _instruction(meta, prefer_caption=cfg.prefer_caption_as_instruction)
        actions = _actions(meta)
        captions = _captions(meta, len(frames))
        dets_all = meta.get("detections") if isinstance(meta.get("detections"), list) else None

        steps: list[TrajectoryStep] = []
        for i, frame in enumerate(frames):
            if store is not None:
                ref = store.put_path(frame)
            else:
                ref = str(frame.resolve())
            n_images += 1

            action: Any | None = None
            if i < len(actions):
                action = actions[i]
            elif actions:
                action = actions[min(i, len(actions) - 1)]

            # Language target: caption / detection when no continuous action
            step_instr = instr
            if not step_instr and captions[i]:
                step_instr = captions[i]
            if not step_instr and dets_all is not None and i < len(dets_all):
                step_instr = _detection_summary(dets_all[i]) or "observe scene"

            if action is None and captions[i]:
                # Vision caption as discrete "action" text for VLM SFT style
                action = captions[i]
            if action is None and dets_all is not None and i < len(dets_all):
                action = _detection_summary(dets_all[i])

            if step_instr is None:
                step_instr = meta.get("task") or "observe and act"
            if action is None:
                continue

            # Keep raw action on step; tokenization happens at example convert time
            # unless already a string (caption/detection).
            steps.append(
                TrajectoryStep(
                    observation_refs=(ref,),
                    instruction=str(step_instr),
                    action=action,
                    reward=float(meta.get("reward", 0.0)) if i == len(frames) - 1 else 0.0,
                    done=i == len(frames) - 1,
                    meta={
                        "frame": i,
                        "frame_path": str(frame),
                        "caption": captions[i],
                    },
                )
            )
            n_steps += 1

        if not steps:
            continue
        trajectories.append(
            Trajectory(
                steps=tuple(steps),
                episode_id=ep_id,
                meta={
                    "dataset": cfg.dataset,
                    "robot_id": meta.get("robot_id") or cfg.robot_id,
                    "source": meta.get("source") or "house_pack",
                    "instruction": instr,
                    "license": meta.get("license"),
                },
            )
        )
        n_ep += 1

    return HousePackResult(
        trajectories=trajectories,
        n_episodes=n_ep,
        n_steps=n_steps,
        n_images=n_images,
        dataset=cfg.dataset,
        action_tokenizer=tok.to_public(),
    )


def house_pack_to_jsonl(
    cfg: HousePackConfig,
    output_jsonl: Path,
    *,
    tokenizer: ActionTokenizer | None = None,
) -> HousePackResult:
    """Write Anvil VLM/robot JSONL rows (instruction, images, response)."""
    result = house_pack_to_trajectories(cfg, tokenizer=tokenizer)
    tok = tokenizer or (
        ActionTokenizer(scheme="bins", n_bins=cfg.action_n_bins)
        if cfg.action_scheme == "bins"
        else ActionTokenizer(scheme="continuous", decimals=cfg.action_decimals)
    )
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for tr in result.trajectories:
            for i, step in enumerate(tr.steps):
                if step.action is None:
                    continue
                try:
                    if isinstance(step.action, str):
                        resp = step.action
                    else:
                        resp = tok.encode(step.action)
                except Exception:
                    resp = format_action_text(step.action, decimals=cfg.action_decimals)
                row = {
                    "instruction": step.instruction or tr.meta.get("instruction") or "",
                    "images": list(step.observation_refs),
                    "response": resp,
                    "dataset": result.dataset,
                    "episode_id": tr.episode_id,
                    "step": i,
                    "source": tr.meta.get("source"),
                    "robot_id": tr.meta.get("robot_id"),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    result.n_steps = n
    return result


def write_demo_house_pack(
    root: Path,
    *,
    n_episodes: int = 4,
    frames_per: int = 3,
    with_actions: bool = True,
) -> Path:
    """Synthetic j30-like pack for CI / laptop (tiny PNG frames)."""
    from PIL import Image

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    colors = [
        (30, 90, 200),
        (200, 40, 40),
        (40, 160, 50),
        (220, 180, 40),
    ]
    scenes = ["kitchen", "hallway", "living room", "doorway"]
    for i in range(n_episodes):
        ep = root / f"ep_{i:04d}"
        frames_dir = ep / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        rgb = colors[i % len(colors)]
        captions = []
        actions = []
        detections = []
        for j in range(frames_per):
            path = frames_dir / f"{j:04d}.png"
            Image.new("RGB", (64, 48), rgb).save(path, format="PNG")
            captions.append(f"{scenes[i % len(scenes)]} view {j}")
            if with_actions:
                # 7-DoF-ish vector in [-1, 1]
                actions.append(
                    [
                        0.1 * (j + 1) * (1 if i % 2 == 0 else -1),
                        0.05 * i,
                        0.02 * j,
                        0.0,
                        0.0,
                        0.0,
                        1.0 if j == frames_per - 1 else 0.0,
                    ]
                )
            detections.append(
                [{"label": "person" if j == 0 else "chair", "conf": 0.8 - 0.05 * j}]
            )
        meta = {
            "language_instruction": f"navigate through the {scenes[i % len(scenes)]}",
            "captions": captions,
            "detections": detections,
            "source": "demo_house_pack",
            "robot_id": "j30-sim",
            "license": "synthetic-demo",
        }
        if with_actions:
            meta["actions"] = actions
        (ep / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return root


def default_edge_pack_tokenizer() -> ActionTokenizer:
    return default_edge_tokenizer()
