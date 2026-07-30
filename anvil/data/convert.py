"""Robotics corpus → Anvil CAS + JSONL (Phase 3.B).

External corpora (Bridge / OXE / LeRobot exports) are **not** vendored.
This module converts a lab-side intermediate layout into:

- content-addressed frames in a :class:`~anvil.media.store.LocalMediaStore`
- Anvil VLM rows (``instruction`` + ``images`` as ``cas://`` + ``response``)

Supported **sources** (no multi-GB deps required for CI):

1. **episode_pack** — directory of episodes::

       source/<episode_id>/meta.json
       source/<episode_id>/frames/*.jpg   # or images/, or loose *.jpg

   ``meta.json`` keys: ``language_instruction`` / ``instruction``, optional
   ``actions`` (list of vectors or strings), ``response``, ``license``.

2. **path_jsonl** — JSONL rows with local image paths (or already ``cas://``)::

       {"instruction": "...", "images": ["rel/frame.jpg"], "response": "..."}

Resume: state file next to the output JSONL tracks completed ``episode_id`` /
row keys so re-runs skip work and **append** new rows (subsample-safe).

Licenses are recorded per row (``--license`` or meta); redistribution is the
operator's responsibility — see ``docs/datasets-robotics.md``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from anvil.media.store import LocalMediaStore, MediaStore

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")


@dataclass
class ConvertConfig:
    """Knobs for a convert run."""

    source: Path
    """Root directory (episode_pack) or path to input JSONL (path_jsonl)."""

    media_root: Path
    """LocalMediaStore root (CAS blobs)."""

    output_jsonl: Path
    """Destination Anvil JSONL (refs only)."""

    source_kind: str = "episode_pack"
    """``episode_pack`` | ``path_jsonl``."""

    max_rows: int | None = None
    """Stop after this many emitted rows (scale ladder: 1k → 5k → …)."""

    max_episodes: int | None = None
    """Cap episodes processed (episode_pack only)."""

    frames_per_episode: int | None = None
    """Subsample frames within an episode (evenly spaced; None = all)."""

    row_mode: str = "per_frame"
    """``per_frame`` (one row per kept frame) or ``keyframe`` (one row / episode)."""

    dataset: str = "robotics"
    """Dataset tag written into each row."""

    license_note: str | None = None
    """Default license/attribution string when meta omits it."""

    action_decimals: int = 4
    """Rounding for vector → text actions (continuous scheme)."""

    action_scheme: str = "continuous"
    """``continuous`` (decimal text) or ``bins`` (OpenVLA-style discrete tokens)."""

    action_n_bins: int = 256
    """Bin count when ``action_scheme='bins'``."""

    resume: bool = True
    """Skip episode_ids / keys already recorded in the state file."""

    state_path: Path | None = None
    """Override state file path (default: ``<output_jsonl>.state.json``)."""


@dataclass
class ConvertResult:
    n_rows: int
    n_episodes: int
    n_skipped: int
    n_images: int
    output_jsonl: Path
    media_root: Path
    state_path: Path
    dataset: str


@dataclass
class _ConvertState:
    done_keys: set[str] = field(default_factory=set)
    n_rows: int = 0
    n_images: int = 0

    def to_public(self) -> dict[str, Any]:
        return {
            "done_keys": sorted(self.done_keys),
            "n_rows": self.n_rows,
            "n_images": self.n_images,
        }

    @classmethod
    def load(cls, path: Path) -> _ConvertState:
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            done_keys=set(str(x) for x in data.get("done_keys") or []),
            n_rows=int(data.get("n_rows") or 0),
            n_images=int(data.get("n_images") or 0),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_public(), indent=2) + "\n", encoding="utf-8")


def format_action_text(
    action: Any,
    *,
    decimals: int = 4,
    scheme: str = "continuous",
    n_bins: int = 256,
) -> str:
    """Turn an action (vector, dict, or string) into a stable text response.

    ``scheme='bins'`` delegates to
    :class:`~anvil.protocol.action_tokens.ActionTokenizer` (Phase 4.A).
    """
    if scheme == "bins":
        from anvil.protocol.action_tokens import ActionTokenizer

        return ActionTokenizer(scheme="bins", n_bins=n_bins, decimals=decimals).encode(
            action
        )
    if action is None:
        raise ValueError("action is None")
    if isinstance(action, str):
        return action.strip()
    if isinstance(action, Mapping):
        parts = []
        for k in sorted(action.keys(), key=str):
            v = action[k]
            if isinstance(v, (int, float)):
                parts.append(f"{k}={float(v):.{decimals}f}")
            else:
                parts.append(f"{k}={v}")
        return " ".join(parts)
    if isinstance(action, (list, tuple)):
        nums: list[str] = []
        for x in action:
            if isinstance(x, (int, float)):
                nums.append(f"{float(x):.{decimals}f}")
            else:
                nums.append(str(x))
        return " ".join(nums)
    if isinstance(action, (int, float)):
        return f"{float(action):.{decimals}f}"
    return str(action)


def _list_frame_paths(ep_dir: Path) -> list[Path]:
    for sub in ("frames", "images", "image", "rgb"):
        d = ep_dir / sub
        if d.is_dir():
            frames = sorted(
                p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
            )
            if frames:
                return frames
    frames = sorted(
        p for p in ep_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    return frames


def _subsample_paths(paths: Sequence[Path], n: int | None) -> list[Path]:
    if n is None or n <= 0 or len(paths) <= n:
        return list(paths)
    if n == 1:
        return [paths[len(paths) // 2]]
    # Evenly spaced indices including endpoints
    idxs = [round(i * (len(paths) - 1) / (n - 1)) for i in range(n)]
    seen: set[int] = set()
    out: list[Path] = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(paths[i])
    return out


def _load_episode_meta(ep_dir: Path) -> dict[str, Any]:
    for name in ("meta.json", "metadata.json", "info.json"):
        p = ep_dir / name
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _instruction_from_meta(meta: Mapping[str, Any]) -> str:
    for key in (
        "language_instruction",
        "instruction",
        "prompt",
        "task",
        "natural_language_instruction",
    ):
        v = meta.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _actions_from_meta(meta: Mapping[str, Any]) -> list[Any]:
    for key in ("actions", "action", "action_sequence"):
        if key not in meta:
            continue
        raw = meta[key]
        if isinstance(raw, list):
            # list of vectors OR a single vector of numbers
            if raw and all(isinstance(x, (int, float)) for x in raw):
                return [raw]
            return list(raw)
        return [raw]
    return []


def iter_episode_pack(source: Path) -> Iterator[tuple[str, Path, dict[str, Any], list[Path]]]:
    """Yield ``(episode_id, ep_dir, meta, frame_paths)`` for each episode dir."""
    if not source.is_dir():
        raise FileNotFoundError(f"episode_pack source is not a directory: {source}")
    children = sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not children:
        # single-episode layout: source itself is the episode
        frames = _list_frame_paths(source)
        if frames:
            yield source.name or "episode", source, _load_episode_meta(source), frames
            return
        raise FileNotFoundError(f"no episode subdirs or frames under {source}")
    for ep_dir in children:
        frames = _list_frame_paths(ep_dir)
        if not frames:
            continue
        yield ep_dir.name, ep_dir, _load_episode_meta(ep_dir), frames


def _put_image(store: MediaStore, path: Path) -> str:
    if hasattr(store, "put_path"):
        return store.put_path(path)  # type: ignore[attr-defined]
    return store.put(path.read_bytes(), suffix=path.suffix)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _row_response(
    *,
    meta: Mapping[str, Any],
    actions: Sequence[Any],
    frame_index: int,
    decimals: int,
    scheme: str = "continuous",
    n_bins: int = 256,
) -> str:
    for key in ("response", "answer", "action_text", "output"):
        if key in meta and meta[key] is not None:
            return str(meta[key])
    if actions:
        # Align action index when possible; else last / only
        if frame_index < len(actions):
            act = actions[frame_index]
        else:
            act = actions[min(frame_index, len(actions) - 1)]
        return format_action_text(
            act, decimals=decimals, scheme=scheme, n_bins=n_bins
        )
    instr = _instruction_from_meta(meta)
    if instr:
        # Last-resort: language-only target (caption-style); prefer real actions
        return instr
    raise ValueError("no response/action available for row")


def convert_episode_pack(
    cfg: ConvertConfig,
    store: MediaStore,
    state: _ConvertState,
) -> ConvertResult:
    n_ep = 0
    n_skipped = 0
    emitted_this_run = 0

    for ep_id, _ep_dir, meta, frames in iter_episode_pack(cfg.source):
        if cfg.max_episodes is not None and n_ep >= cfg.max_episodes:
            break
        if cfg.max_rows is not None and emitted_this_run >= cfg.max_rows:
            break

        key = f"ep:{ep_id}"
        if cfg.resume and key in state.done_keys:
            n_skipped += 1
            continue

        instr = _instruction_from_meta(meta)
        if not instr:
            n_skipped += 1
            continue

        kept = _subsample_paths(frames, cfg.frames_per_episode)
        actions = _actions_from_meta(meta)
        license_note = meta.get("license") or cfg.license_note
        dataset = str(meta.get("dataset") or cfg.dataset)

        if cfg.row_mode == "keyframe":
            kept = [kept[len(kept) // 2]] if kept else []

        try:
            stopped_early = False
            for fi, frame_path in enumerate(kept):
                if cfg.max_rows is not None and emitted_this_run >= cfg.max_rows:
                    stopped_early = True
                    break
                row_key = f"{key}:f{fi}:{frame_path.name}"
                if cfg.resume and row_key in state.done_keys:
                    continue
                ref = _put_image(store, frame_path)
                state.n_images += 1
                response = _row_response(
                    meta=meta,
                    actions=actions,
                    frame_index=fi if cfg.row_mode == "per_frame" else 0,
                    decimals=cfg.action_decimals,
                    scheme=cfg.action_scheme,
                    n_bins=cfg.action_n_bins,
                )
                row = {
                    "instruction": instr,
                    "images": [ref],
                    "response": response,
                    "dataset": dataset,
                    "episode_id": ep_id,
                    "frame_index": fi,
                    "source": "episode_pack",
                }
                if license_note:
                    row["license"] = str(license_note)
                _append_jsonl(cfg.output_jsonl, row)
                state.done_keys.add(row_key)
                state.n_rows += 1
                emitted_this_run += 1
            # Only mark episode complete when all kept frames were considered
            # (otherwise max_rows mid-episode would skip remaining frames forever).
            if not stopped_early:
                state.done_keys.add(key)
                n_ep += 1
        finally:
            state.save(cfg.state_path or _default_state_path(cfg.output_jsonl))

    sp = cfg.state_path or _default_state_path(cfg.output_jsonl)
    state.save(sp)
    return ConvertResult(
        n_rows=emitted_this_run,
        n_episodes=n_ep,
        n_skipped=n_skipped,
        n_images=state.n_images,
        output_jsonl=cfg.output_jsonl,
        media_root=cfg.media_root,
        state_path=sp,
        dataset=cfg.dataset,
    )


def convert_path_jsonl(
    cfg: ConvertConfig,
    store: MediaStore,
    state: _ConvertState,
) -> ConvertResult:
    """Materialize path-based VLM JSONL into cas:// refs JSONL."""
    src = cfg.source
    if not src.is_file():
        raise FileNotFoundError(src)
    base = src.parent
    n_skipped = 0
    emitted = 0
    line_no = 0

    with src.open(encoding="utf-8") as f:
        for line in f:
            line_no += 1
            line = line.strip()
            if not line:
                continue
            if cfg.max_rows is not None and emitted >= cfg.max_rows:
                break
            row_in = json.loads(line)
            key = str(row_in.get("episode_id") or row_in.get("id") or f"line:{line_no}")
            row_key = f"jsonl:{key}:{line_no}"
            if cfg.resume and row_key in state.done_keys:
                n_skipped += 1
                continue

            raw_images = row_in.get("images") or row_in.get("image") or []
            if isinstance(raw_images, str):
                raw_images = [raw_images]
            refs: list[str] = []
            for item in raw_images:
                s = str(item)
                if s.startswith("cas://"):
                    refs.append(s)
                    continue
                path = Path(s)
                if not path.is_file():
                    path = base / s
                if not path.is_file():
                    raise FileNotFoundError(f"image not found for line {line_no}: {s}")
                refs.append(_put_image(store, path))
                state.n_images += 1

            instr = (
                row_in.get("instruction")
                or row_in.get("language_instruction")
                or row_in.get("prompt")
                or ""
            )
            resp = (
                row_in.get("response")
                or row_in.get("answer")
                or row_in.get("action_text")
                or row_in.get("output")
            )
            if resp is None and row_in.get("action") is not None:
                resp = format_action_text(
                    row_in["action"],
                    decimals=cfg.action_decimals,
                    scheme=cfg.action_scheme,
                    n_bins=cfg.action_n_bins,
                )
            if not str(instr).strip() or resp is None:
                n_skipped += 1
                continue

            out = {
                "instruction": str(instr),
                "images": refs,
                "response": str(resp) if not isinstance(resp, str) else resp,
                "dataset": str(row_in.get("dataset") or cfg.dataset),
                "episode_id": key,
                "source": "path_jsonl",
            }
            lic = row_in.get("license") or cfg.license_note
            if lic:
                out["license"] = str(lic)
            for k in ("reward", "step"):
                if k in row_in:
                    out[k] = row_in[k]

            _append_jsonl(cfg.output_jsonl, out)
            state.done_keys.add(row_key)
            state.n_rows += 1
            emitted += 1
            if emitted % 50 == 0:
                state.save(cfg.state_path or _default_state_path(cfg.output_jsonl))

    sp = cfg.state_path or _default_state_path(cfg.output_jsonl)
    state.save(sp)
    return ConvertResult(
        n_rows=emitted,
        n_episodes=0,
        n_skipped=n_skipped,
        n_images=state.n_images,
        output_jsonl=cfg.output_jsonl,
        media_root=cfg.media_root,
        state_path=sp,
        dataset=cfg.dataset,
    )


def _default_state_path(output_jsonl: Path) -> Path:
    return Path(str(output_jsonl) + ".state.json")


def convert_corpus(cfg: ConvertConfig) -> ConvertResult:
    """Run a convert according to ``cfg.source_kind``."""
    kind = cfg.source_kind.replace("-", "_")
    if kind not in {"episode_pack", "path_jsonl"}:
        raise ValueError(
            f"unknown source_kind {cfg.source_kind!r}; "
            "use episode_pack or path_jsonl"
        )
    if cfg.row_mode not in {"per_frame", "keyframe"}:
        raise ValueError(f"row_mode must be per_frame|keyframe, got {cfg.row_mode!r}")

    cfg.source = Path(cfg.source)
    cfg.media_root = Path(cfg.media_root)
    cfg.output_jsonl = Path(cfg.output_jsonl)
    state_path = Path(cfg.state_path) if cfg.state_path else _default_state_path(cfg.output_jsonl)
    cfg.state_path = state_path

    store: MediaStore = LocalMediaStore(cfg.media_root)
    state = _ConvertState.load(state_path) if cfg.resume else _ConvertState()
    if not cfg.resume and cfg.output_jsonl.exists():
        # Fresh run: truncate output so counts match state
        cfg.output_jsonl.write_text("", encoding="utf-8")

    if kind == "episode_pack":
        return convert_episode_pack(cfg, store, state)
    return convert_path_jsonl(cfg, store, state)


def write_solid_png(path: Path, rgb: tuple[int, int, int] = (40, 120, 200), size: int = 8) -> None:
    """Write a solid RGB PNG with no third-party deps (stdlib zlib only).

    CI does not install Pillow; convert tests and ``--demo`` must stay dep-light.
    """
    import struct
    import zlib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = h = int(size)
    r, g, b = (int(rgb[0]) & 255, int(rgb[1]) & 255, int(rgb[2]) & 255)
    # raw image: each row is filter_byte(0) + RGB pixels
    raw = b"".join(b"\x00" + bytes([r, g, b]) * w for _ in range(h))
    compressed = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def write_demo_episode_pack(
    root: Path,
    *,
    n_episodes: int = 3,
    frames_per: int = 2,
) -> Path:
    """Create a tiny synthetic episode_pack for tests / lab smoke (stdlib only)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    for i in range(n_episodes):
        ep = root / f"ep_{i:04d}"
        frames = ep / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        rgb = (40 + i * 40, 80, 160)
        for j in range(frames_per):
            write_solid_png(frames / f"{j:04d}.png", rgb=rgb, size=8)
        meta = {
            "language_instruction": f"pick up object {i}",
            "actions": [[0.1 * i, 0.0, 0.05 * j, 0.0, 0.0, 0.0, 1.0] for j in range(frames_per)],
            "dataset": "demo_bridge_like",
            "license": "synthetic-demo-not-bridge",
        }
        (ep / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return root


__all__ = [
    "ConvertConfig",
    "ConvertResult",
    "convert_corpus",
    "format_action_text",
    "iter_episode_pack",
    "write_demo_episode_pack",
    "write_solid_png",
]
