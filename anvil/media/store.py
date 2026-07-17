"""Content-addressed media store (local dir; S3/MinIO later).

Refs look like ``cas://sha256/<hex>[.ext]``. Batches and trajectories carry
refs, not multi-MB blobs, so train/sample workers resolve once via the store.
"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable


@runtime_checkable
class MediaStore(Protocol):
    def put(self, data: bytes, *, suffix: str = "") -> str:
        """Store bytes; return content-addressed ref."""
        ...

    def get(self, ref: str) -> bytes:
        ...

    def exists(self, ref: str) -> bool:
        ...


def _normalize_suffix(suffix: str) -> str:
    if not suffix:
        return ""
    return suffix if suffix.startswith(".") else f".{suffix}"


class LocalMediaStore:
    """Filesystem CAS under ``root``. Refs: ``cas://sha256/<hex>[.ext]``."""

    scheme = "cas://sha256/"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, *, suffix: str = "") -> str:
        digest = hashlib.sha256(data).hexdigest()
        ext = _normalize_suffix(suffix)
        rel = f"{digest[:2]}/{digest}{ext}"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return f"{self.scheme}{digest}{ext}"

    def put_path(self, path: str | Path, *, suffix: str | None = None) -> str:
        """Read a local file into the CAS; suffix defaults from the filename."""
        p = Path(path)
        data = p.read_bytes()
        if suffix is None:
            suffix = p.suffix or ""
        return self.put(data, suffix=suffix)

    def put_file(self, fp: BinaryIO, *, suffix: str = "") -> str:
        return self.put(fp.read(), suffix=suffix)

    def path_for(self, ref: str) -> Path:
        """Absolute filesystem path for a ref (must exist under this store)."""
        return self._path_for(ref)

    def open(self, ref: str) -> BinaryIO:
        return self.path_for(ref).open("rb")

    def mime_type(self, ref: str) -> str | None:
        path = self._path_for(ref)
        mime, _ = mimetypes.guess_type(str(path))
        return mime

    def _path_for(self, ref: str) -> Path:
        if not ref.startswith(self.scheme):
            raise ValueError(f"unsupported ref scheme: {ref!r}")
        rest = ref[len(self.scheme) :]
        hex_part = rest.split(".", 1)[0]
        if len(hex_part) != 64 or any(c not in "0123456789abcdef" for c in hex_part.lower()):
            raise ValueError(f"bad ref digest: {ref!r}")
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
