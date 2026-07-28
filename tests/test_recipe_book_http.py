"""HTTP + control-client surfaces for personal recipe book and meta-recipes."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from anvil.agent.client import AnvilControlClient
from anvil.recipes.book import BookRecipe, RecipeBook
from anvil.recipes.meta import MetaEdge, MetaRecipe, MetaStage, save_meta_recipe
from anvil.web import state as state_mod
from anvil.web.app import create_app


@pytest.fixture()
def book_root(tmp_path, monkeypatch):
    root = tmp_path / "recipe-book"
    monkeypatch.setenv("ANVIL_RECIPE_BOOK", str(root))
    return root


@pytest.fixture()
def client(tmp_path, book_root, monkeypatch):
    monkeypatch.setenv("ANVIL_FAKE_ROOT", str(tmp_path / "fake"))
    monkeypatch.setenv("ANVIL_EXPORT_ROOT", str(tmp_path / "exports"))
    monkeypatch.setenv("ANVIL_MODELS_ROOT", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    state_mod._STORE = None
    app = create_app()
    with TestClient(app) as c:
        yield c
    state_mod._STORE = None


def _seed_book(book_root):
    book = RecipeBook(root=book_root)
    book.save(
        BookRecipe(
            id="lab-edge-vlm",
            title="Lab edge VLM",
            pattern="vlm_sft",
            family="qwen2.5-vl",
            knobs={"rank": 16, "learning_rate": 1e-4},
            stop_policy={"early_stop_patience": 20},
            notes="from dogfood",
        )
    )
    save_meta_recipe(
        MetaRecipe(
            id="sft-then-pref",
            title="SFT then preference",
            stages=[
                MetaStage(id="s0", recipe_id="vlm_sft_edge", pattern="vlm_sft"),
                MetaStage(id="s1", recipe_id="lab-edge-vlm", source="personal_book"),
            ],
            edges=[
                MetaEdge(on="early_stop:loss_plateau*", from_stage="s0", to_stage="s1"),
            ],
            family="qwen2.5-vl",
        ),
        root=book_root,
    )


def test_http_recipe_book_list_and_get(client, book_root):
    _seed_book(book_root)
    r = client.get("/api/recipe-book")
    assert r.status_code == 200
    body = r.json()
    assert body["root"] == str(book_root)
    ids = {x["id"] for x in body["recipes"]}
    assert "lab-edge-vlm" in ids

    r = client.get("/api/recipe-book", params={"family": "qwen2.5-vl"})
    assert r.status_code == 200
    assert any(x["id"] == "lab-edge-vlm" for x in r.json()["recipes"])

    r = client.get("/api/recipe-book/lab-edge-vlm")
    assert r.status_code == 200
    assert r.json()["knobs"]["rank"] == 16

    assert client.get("/api/recipe-book/missing-id").status_code == 404


def test_http_meta_recipes_list_and_get(client, book_root):
    _seed_book(book_root)
    r = client.get("/api/meta-recipes")
    assert r.status_code == 200
    body = r.json()
    assert any(m["id"] == "sft-then-pref" for m in body["meta_recipes"])

    r = client.get("/api/meta-recipes/sft-then-pref")
    assert r.status_code == 200
    m = r.json()
    assert m["kind"] == "meta_recipe"
    assert len(m["stages"]) == 2
    assert m["edges"][0]["on"].startswith("early_stop")

    assert client.get("/api/meta-recipes/nope").status_code == 404


def test_control_client_book_and_meta(client, book_root, monkeypatch):
    """AnvilControlClient list/get book + meta against the same HTTP SSOT."""
    _seed_book(book_root)

    # Bridge urllib → TestClient so we exercise the real client methods.
    def _urlopen(req, timeout=None):
        method = req.get_method()
        path = req.full_url.replace("http://testserver", "")
        if "?" in path:
            path_only, qs = path.split("?", 1)
        else:
            path_only, qs = path, ""
        headers = dict(req.headers)
        data = req.data
        if method == "GET":
            resp = client.request("GET", path_only + (("?" + qs) if qs else ""), headers=headers)
        else:
            resp = client.request(
                method,
                path_only + (("?" + qs) if qs else ""),
                content=data,
                headers=headers,
            )

        class _R:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def read(self_inner):
                return resp.content

        if resp.status_code >= 400:
            import urllib.error

            raise urllib.error.HTTPError(
                req.full_url, resp.status_code, "err", hdrs=None, fp=type(
                    "F", (), {"read": lambda self: resp.content}
                )()
            )
        return _R()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    c = AnvilControlClient(base_url="http://testserver")
    book = c.list_recipe_book(family="qwen2.5-vl")
    assert any(r["id"] == "lab-edge-vlm" for r in book["recipes"])
    one = c.get_recipe_book("lab-edge-vlm")
    assert one["title"] == "Lab edge VLM"
    meta = c.list_meta_recipes()
    assert any(m["id"] == "sft-then-pref" for m in meta["meta_recipes"])
    got = c.get_meta_recipe("sft-then-pref")
    assert got["stages"][1]["source"] == "personal_book"


def test_mcp_and_harness_expose_book_tools():
    from anvil.agent.harness import tool_specs

    names = {t["function"]["name"] for t in tool_specs()}
    for required in (
        "anvil_list_recipe_book",
        "anvil_get_recipe_book",
        "anvil_list_meta_recipes",
        "anvil_get_meta_recipe",
    ):
        assert required in names

    pytest.importorskip("mcp")
    from anvil.agent.mcp_server import build_mcp_server

    srv = build_mcp_server("http://127.0.0.1:7600")
    assert srv is not None
