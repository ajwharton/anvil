"""HTTP client for the Anvil control / observe SSOT (same as anvil-web).

Used by MCP tools and the optional agent harness. Default base URL:
``http://127.0.0.1:7600`` (anvil-web).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping


class AnvilControlClient:
    """Thin urllib client — no extra deps (works in the core package)."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("ANVIL_CONTROL_URL") or "http://127.0.0.1:7600").rstrip(
            "/"
        )
        self.token = token if token is not None else os.environ.get("ANVIL_TOKEN")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if query:
            q = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
            url = f"{url}?{q}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                if not raw:
                    return None
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8"))
            except Exception:
                detail = {"detail": str(e)}
            raise RuntimeError(f"HTTP {e.code} {path}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"unreachable {self.base_url}: {e}") from e

    # --- discover ------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/health")

    def defaults(self) -> dict[str, Any]:
        return self._request("GET", "/api/defaults")

    def overview(self) -> dict[str, Any]:
        return self._request("GET", "/api/overview")

    def list_recipes(self, group: str | None = None) -> list[dict[str, Any]]:
        return self._request("GET", "/api/recipes", query={"group": group})

    def suggest(self, base_model: str, *, fetch_remote: bool = False) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/suggest",
            query={"base_model": base_model, "fetch_remote": str(fetch_remote).lower()},
        )

    def gate(
        self,
        recipe_id: str,
        shape: str,
        *,
        param_count: int | None = None,
        has_vision: bool | None = None,
        rank: int | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            "/api/gate",
            query={
                "recipe_id": recipe_id,
                "shape": shape,
                "param_count": param_count,
                "has_vision": None if has_vision is None else str(has_vision).lower(),
                "rank": rank,
                "learning_rate": learning_rate,
            },
        )

    def plan(
        self,
        base_model: str,
        *,
        pattern: str | None = None,
        recipe_id: str | None = None,
        shape: str | None = None,
        overrides: dict[str, Any] | None = None,
        force: bool = False,
        fetch_remote: bool = False,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/plan",
            {
                "base_model": base_model,
                "pattern": pattern,
                "recipe_id": recipe_id,
                "shape": shape,
                "overrides": overrides or {},
                "force": force,
                "fetch_remote": fetch_remote,
            },
        )

    def list_audit(self, kind: str | None = None) -> list[dict[str, Any]]:
        return self._request("GET", "/api/audit", query={"kind": kind})

    # --- runs ----------------------------------------------------------------

    def list_runs(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/runs")

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/runs/{run_id}")

    def create_run(
        self,
        *,
        name: str | None = None,
        knobs: dict[str, Any] | None = None,
        recipe_id: str | None = None,
        pattern: str | None = None,
        shape: str | None = None,
        rationale: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/runs",
            {
                "name": name,
                "knobs": knobs or {},
                "recipe_id": recipe_id,
                "pattern": pattern,
                "shape": shape,
                "rationale": rationale or [],
                "force": force,
            },
        )

    def train(self, run_id: str, steps: int = 1) -> dict[str, Any]:
        return self._request("POST", f"/api/runs/{run_id}/train", {"steps": steps})

    def pause(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/runs/{run_id}/pause", {})

    def resume(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/runs/{run_id}/resume", {})

    def patch_knobs(self, run_id: str, knobs: dict[str, Any]) -> dict[str, Any]:
        return self._request("PATCH", f"/api/runs/{run_id}/knobs", {"knobs": knobs})

    def sample(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/runs/{run_id}/sample", {})

    def export(self, run_id: str, fmt: str = "peft") -> dict[str, Any]:
        return self._request("POST", f"/api/runs/{run_id}/export", {"format": fmt})

    def checkpoint(self, run_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/runs/{run_id}/checkpoint", {})

    # --- observe -------------------------------------------------------------

    def list_observe_runs(self) -> dict[str, Any]:
        return self._request("GET", "/api/observe")

    def observe_metrics(self, run_id: str, tail: int = 50) -> dict[str, Any]:
        return self._request(
            "GET", f"/api/observe/{run_id}/metrics", query={"tail": tail}
        )

    def observe_probes(self, run_id: str, tail: int = 24) -> dict[str, Any]:
        return self._request(
            "GET", f"/api/observe/{run_id}/probes", query={"tail": tail}
        )

    def observe_jlens(self, run_id: str, tail: int = 20) -> dict[str, Any]:
        return self._request(
            "GET", f"/api/observe/{run_id}/jlens", query={"tail": tail}
        )
