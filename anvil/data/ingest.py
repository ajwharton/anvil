"""Ingest helpers: paths / simple JSONL → media-store refs + Examples.

Robotics corpora (OXE, Bridge, LeRobot exports) differ wildly. Anvil's
contract is multimodal :class:`~anvil.protocol.messages.Example` rows with
``cas://`` image refs. Convertors for specific formats can live here or in
recipes; this module covers the common path:

- put local image files into :class:`~anvil.media.store.LocalMediaStore`
- load JSONL rows ``{instruction, images: [path|ref], response, …}``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from anvil.media.store import MediaStore
from anvil.protocol.messages import Example, ImagePart, ImageUrlPart, Message, TextPart


def put_images_from_paths(
    store: MediaStore,
    paths: Sequence[str | Path],
) -> list[str]:
    """Store each path; return ordered content-addressed refs."""
    refs: list[str] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(path)
        if hasattr(store, "put_path"):
            refs.append(store.put_path(path))  # type: ignore[attr-defined]
        else:
            refs.append(store.put(path.read_bytes(), suffix=path.suffix))
    return refs


def materialize_image_urls(example: Example, store: MediaStore) -> Example:
    """Replace ``ImageUrlPart`` with ``ImagePart`` after fetching URL bytes.

    Local ``file://`` URLs and plain paths are supported without network.
    ``http(s)://`` requires the caller to supply bytes via a custom store or
    pre-download — this helper only handles ``file://`` and absolute paths.
    """
    new_messages: list[Message] = []
    for m in example.messages:
        parts: list[Any] = []
        for p in m.parts():
            if isinstance(p, ImageUrlPart):
                url = p.url
                if url.startswith("file://"):
                    path = Path(url[7:])
                elif url.startswith("/") or (len(url) > 2 and url[1] == ":"):
                    path = Path(url)
                else:
                    raise ValueError(
                        f"materialize_image_urls only handles file:// or local paths; "
                        f"got {url!r} — download first and put() into the media store"
                    )
                ref = (
                    store.put_path(path)  # type: ignore[attr-defined]
                    if hasattr(store, "put_path")
                    else store.put(path.read_bytes(), suffix=path.suffix)
                )
                parts.append(ImagePart(ref=ref, detail="auto"))
            else:
                parts.append(p)
        new_messages.append(Message(role=m.role, content=tuple(parts)))
    return Example(messages=tuple(new_messages), meta=dict(example.meta))


def example_from_vlm_row(
    row: Mapping[str, Any],
    store: MediaStore | None = None,
    *,
    base_dir: str | Path | None = None,
) -> Example:
    """Build one Example from a dict row.

    Accepted keys (aliases in parentheses):

    - ``instruction`` (``prompt``, ``question``, ``language_instruction``)
    - ``response`` (``answer``, ``output``, ``action_text``)
    - ``images`` (``image``, ``observation_refs``) — list of refs or local paths
    - ``meta`` — optional dict merged into example meta
    """
    instr = (
        row.get("instruction")
        or row.get("prompt")
        or row.get("question")
        or row.get("language_instruction")
        or ""
    )
    resp = row.get("response") or row.get("answer") or row.get("output") or row.get("action_text")
    if resp is None:
        raise ValueError("row missing response/answer/output/action_text")
    if not isinstance(resp, str):
        resp = str(resp)

    raw_images = row.get("images") or row.get("image") or row.get("observation_refs") or []
    if isinstance(raw_images, str):
        raw_images = [raw_images]

    refs: list[str] = []
    base = Path(base_dir) if base_dir else None
    for item in raw_images:
        s = str(item)
        if s.startswith("cas://"):
            refs.append(s)
            continue
        if store is None:
            raise ValueError("local image paths require a MediaStore")
        path = Path(s)
        if not path.is_file() and base is not None:
            path = base / s
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {s}")
        refs.append(
            store.put_path(path)  # type: ignore[attr-defined]
            if hasattr(store, "put_path")
            else store.put(path.read_bytes(), suffix=path.suffix)
        )

    user_parts: list[Any] = [TextPart(text=str(instr))]
    for ref in refs:
        user_parts.append(ImagePart(ref=ref, detail=str(row.get("detail", "auto"))))

    meta = dict(row.get("meta") or {})
    for k in ("episode_id", "dataset", "source", "reward"):
        if k in row and k not in meta:
            meta[k] = row[k]

    return Example(
        messages=(
            Message(role="user", content=tuple(user_parts)),
            Message(role="assistant", content=str(resp)),
        ),
        meta=meta,
    )


def examples_from_vlm_jsonl(
    path: str | Path,
    store: MediaStore | None = None,
    *,
    base_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[Example]:
    """Load a JSONL file of VLM/robot instruction rows into Examples."""
    p = Path(path)
    base = base_dir if base_dir is not None else p.parent
    out: list[Example] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(example_from_vlm_row(row, store, base_dir=base))
            if limit is not None and len(out) >= limit:
                break
    return out


def write_examples_jsonl(path: str | Path, examples: Iterable[Example]) -> int:
    """Write Examples as public JSONL (refs only — no image bytes)."""
    n = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_public(), ensure_ascii=False) + "\n")
            n += 1
    return n
