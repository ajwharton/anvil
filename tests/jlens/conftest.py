"""Shelved J-lens measurement kit — optional; excluded from default pytest.

Run: ``pytest -m jlens`` or ``pytest tests/jlens/ -o addopts=``

Product path does not depend on J-lens; see docs/spikes/jlens-math.md.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    mark = pytest.mark.jlens
    for item in items:
        # Only mark tests living under tests/jlens/
        path = str(getattr(item, "path", item.fspath))
        if "/jlens/" in path.replace("\\", "/") or path.replace("\\", "/").endswith(
            "/jlens"
        ):
            item.add_marker(mark)
