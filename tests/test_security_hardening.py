"""Regression tests for security, red-line, and path-traversal fixes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil.backends.local import _safe_checkpoint_name
from anvil.export.edge import try_run_converter
from anvil.media.store import LocalMediaStore
from anvil.recipes.book import RecipeBook


def test_no_private_lan_ip_defaults_in_run_store(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SPARK_DASHBOARD_FORGE", raising=False)
    monkeypatch.delenv("SPARK_DASHBOARD_HAMMER", raising=False)
    monkeypatch.setenv("ANVIL_FAKE_ROOT", str(tmp_path / "fake"))
    from anvil.web.state import RunStore

    store = RunStore(models_root=str(tmp_path / "models"), export_root=str(tmp_path / "exports"))
    assert store.spark_dashboard_urls["forge"] == ""
    assert store.spark_dashboard_urls["hammer"] == ""
    # Lazy mkdir: constructing RunStore does not touch the filesystem
    assert not (tmp_path / "exports").exists()


def test_converter_blocks_shell_injection(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pwned = tmp_path / "pwned"
    # Even if destination path contains shell metacharacters, shell=False prevents execution
    bad_dst = tmp_path / "out; touch pwned"
    monkeypatch.setenv("ANVIL_GGUF_CONVERTER", "touch {dst}")
    ok, _msg = try_run_converter(
        env_key="ANVIL_GGUF_CONVERTER",
        source_dir=tmp_path,
        dest=bad_dst,
    )
    assert ok
    assert bad_dst.is_file()
    assert not pwned.exists()


def test_media_store_rejects_suffix_path_traversal(tmp_path: Path):
    store = LocalMediaStore(tmp_path / "media")
    digest = "a" * 64
    with pytest.raises(ValueError):
        store.path_for(f"cas://sha256/{digest}./../../../etc/passwd")
    with pytest.raises(ValueError):
        store.path_for(f"cas://sha256/{digest}.png/../../escape")


def test_recipe_book_get_rejects_path_traversal(tmp_path: Path):
    book = RecipeBook(root=tmp_path / "book")
    with pytest.raises(ValueError):
        book.get("../outside")
    with pytest.raises(ValueError):
        book.get("a..b")


def test_checkpoint_name_rejects_traversal():
    assert _safe_checkpoint_name("step-10") == "step-10"
    for bad in ("", "../escape", "a/b", "a\\b", "foo..bar"):
        with pytest.raises(ValueError):
            _safe_checkpoint_name(bad)


def test_web_export_rejects_invalid_format_traversal(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from anvil.web import state as state_mod
    from anvil.web.app import create_app

    monkeypatch.setenv("ANVIL_FAKE_ROOT", str(tmp_path / "fake"))
    monkeypatch.setenv("ANVIL_EXPORT_ROOT", str(tmp_path / "exports"))
    monkeypatch.setenv("ANVIL_MODELS_ROOT", str(tmp_path / "models"))
    state_mod._STORE = None
    try:
        with TestClient(create_app()) as c:
            run = c.post("/api/runs", json={"name": "sec"}).json()
            r = c.post(
                f"/api/runs/{run['run_id']}/export",
                json={"format": "../../escaped"},
            )
            assert r.status_code == 400
            assert not (tmp_path / "escaped").exists()
    finally:
        state_mod._STORE = None


def test_observe_index_escapes_html(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from anvil.web import state as state_mod
    from anvil.web.app import create_app

    obs = tmp_path / "obs"
    run_dir = obs / "run-xss"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"type": "step", "step": 1, "loss": 0.5, "job": "<script>alert(1)</script>"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANVIL_OBSERVE_ROOT", str(obs))
    monkeypatch.setenv("ANVIL_FAKE_ROOT", str(tmp_path / "fake"))
    monkeypatch.setenv("ANVIL_EXPORT_ROOT", str(tmp_path / "exports"))
    state_mod._STORE = None
    try:
        with TestClient(create_app()) as c:
            r = c.get("/observe")
            assert r.status_code == 200
            assert "<script>alert(1)</script>" not in r.text
            assert "&lt;script&gt;alert(1)&lt;/script&gt;" in r.text
    finally:
        state_mod._STORE = None
