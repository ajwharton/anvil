"""Content-addressed media store (local dir for Phase 0)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class MediaStore(Protocol):
    def put(self, data: bytes, *, suffix: str = "") -> str:
        """Store bytes; return content-addressed ref."""
        ...

    def get(self, ref: str) -> bytes:
        ...

    def exists(self, ref: str) -> bool:
        ...


class LocalMediaStore:
    """Filesystem CAS under ``root``. Refs look like ``cas://sha256/<hex>[.ext]``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, *, suffix: str = "") -> str:
        digest = hashlib.sha256(data).hexdigest()
        ext = suffix if suffix.startswith(".") or not suffix else f".{suffix}"
        rel = f"{digest[:2]}/{digest}{ext}"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return f"cas://sha256/{digest}{ext}"

    def _path_for(self, ref: str) -> Path:
        if not ref.startswith("cas://sha256/"):
            raise ValueError(f"unsupported ref scheme: {ref!r}")
        rest = ref[len("cas://sha256/") :]
        # rest = hex or hex.ext
        hex_part = rest.split(".", 1)[0]
        if len(hex_part) < 2:
            raise ValueError(f"bad ref: {ref!r}")
        return self.root / hex_part[:2] / rest

    def get(self, ref: str) -> bytes:
        path = self._path_for(ref)
        if not path.exists():
            raise FileNotFoundError(ref)
        return path.read_bytes()

    def exists(self, ref: str) -> bool:
        try:
            return self._path_for(ref).exists()
        except ValueError:
            return False
