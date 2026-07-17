"""Anvil web control plane — FastAPI + spark-dashboard-inspired UI.

Run::

    pip install -e ".[web]"
    anvil-web --host 0.0.0.0 --port 7600
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from anvil.recipes import (
    gate_recipe,
    inspect_base_model,
    list_patterns,
    list_recipes,
    plan_recipe,
    recipes_for_shape,
    suggest_for_model,
)
from anvil.web.state import get_store

STATIC = Path(__file__).resolve().parent / "static"


class CreateRunIn(BaseModel):
    name: str | None = None
    knobs: dict[str, Any] = Field(default_factory=dict)
    pattern: str | None = None
    recipe_id: str | None = None
    shape: str | None = None
    rationale: list[str] = Field(default_factory=list)
    force: bool = False


class PlanIn(BaseModel):
    base_model: str
    pattern: str | None = None
    recipe_id: str | None = None
    shape: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    fetch_remote: bool = False
    force: bool = False


class TrainIn(BaseModel):
    steps: int = 1


class ExportIn(BaseModel):
    format: str = "peft"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Anvil",
        description="Tinker-shaped post-training control plane",
        version="0.0.1",
    )
    store = get_store()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "anvil-web"}

    @app.get("/api/defaults")
    def defaults() -> dict[str, Any]:
        return store.defaults()

    @app.get("/api/patterns")
    def patterns() -> list[dict[str, Any]]:
        """Low-level job patterns (loss families). Prefer /api/recipes."""
        return list_patterns()

    @app.get("/api/recipes")
    def recipes(group: str | None = None) -> list[dict[str, Any]]:
        """Bounded catalog (~14) with architecture boundaries."""
        return list_recipes(group=group)

    @app.get("/api/recipes/for-shape")
    def recipes_shape(
        shape: str,
        param_count: int | None = None,
        has_vision: bool | None = None,
        include_blocked: bool = False,
    ) -> list[dict[str, Any]]:
        return recipes_for_shape(
            shape,
            param_count=param_count,
            has_vision=has_vision,
            include_blocked=include_blocked,
        )

    @app.get("/api/gate")
    def gate(
        recipe_id: str,
        shape: str,
        param_count: int | None = None,
        has_vision: bool | None = None,
        rank: int | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        try:
            return gate_recipe(
                recipe_id,
                shape=shape,
                param_count=param_count,
                has_vision=has_vision,
                rank=rank,
                learning_rate=learning_rate,
            ).to_public()
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/suggest")
    def suggest(
        base_model: str,
        fetch_remote: bool = False,
        include_blocked: bool = False,
    ) -> dict[str, Any]:
        """Shape + gated catalog recipes for a base model."""
        return suggest_for_model(
            base_model,
            fetch_remote=fetch_remote,
            include_blocked=include_blocked,
        )

    @app.get("/api/model-card")
    def model_card(base_model: str, fetch_remote: bool = True) -> dict[str, Any]:
        """Inspect HF card / local config.json → architecture facts."""
        try:
            return inspect_base_model(base_model, fetch_remote=fetch_remote).to_public()
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/plan")
    def plan(payload: PlanIn) -> dict[str, Any]:
        try:
            return plan_recipe(
                base_model=payload.base_model,
                pattern=payload.pattern,
                recipe_id=payload.recipe_id,
                shape=payload.shape,
                overrides=payload.overrides or None,
                fetch_remote=payload.fetch_remote,
                force=payload.force,
            ).to_public()
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return store.overview()

    @app.get("/api/audit")
    def audit(kind: str | None = None) -> list[dict[str, Any]]:
        """Control-plane audit trail (gate overrides now; multi-user in Phase 5)."""
        from anvil.control.audit import default_log

        return [e.to_public() for e in default_log().events(kind=kind)]

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        return store.list_local_models()

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [r.to_public() for r in store.list_runs()]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return store.get_run(run_id).to_public()
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/runs")
    def create_run(payload: CreateRunIn) -> dict[str, Any]:
        knobs = dict(payload.knobs)
        rationale = list(payload.rationale)
        shape = payload.shape
        pattern = payload.pattern
        # Recipe/pattern path: re-derive plan (knobs = overrides; gates enforced)
        if payload.recipe_id or payload.pattern:
            try:
                plan = plan_recipe(
                    base_model=knobs.get("base_model")
                    or "Qwen/Qwen2.5-VL-3B-Instruct",
                    pattern=payload.pattern,
                    recipe_id=payload.recipe_id,
                    shape=payload.shape,
                    overrides=knobs or None,
                    force=payload.force,
                )
                knobs = plan.as_knobs()
                if not rationale:
                    rationale = list(plan.rationale)
                if plan.gate and plan.gate.get("level") == "stretch":
                    rationale = rationale + list(plan.gate.get("stretch_reasons") or [])
                shape = plan.shape.value
                pattern = plan.pattern.value
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            except KeyError as e:
                raise HTTPException(404, str(e)) from e
        try:
            rec = store.create_run(
                payload.name,
                knobs,
                pattern=pattern or payload.recipe_id,
                shape=shape,
                rationale=rationale,
            )
        except TypeError as e:
            raise HTTPException(400, f"invalid knobs: {e}") from e
        return rec.to_public()

    @app.post("/api/runs/{run_id}/train")
    def train(run_id: str, payload: TrainIn) -> dict[str, Any]:
        try:
            rec = store.train_steps(run_id, n_steps=max(1, min(payload.steps, 50)))
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return rec.to_public()

    @app.post("/api/runs/{run_id}/sample")
    def sample(run_id: str) -> dict[str, Any]:
        try:
            return store.sample(run_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/runs/{run_id}/export")
    def export(run_id: str, payload: ExportIn) -> dict[str, Any]:
        try:
            return store.export_run(run_id, fmt=payload.format)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except Exception as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/runs/{run_id}/checkpoint")
    def checkpoint(run_id: str) -> dict[str, Any]:
        try:
            return store.save_checkpoint(run_id)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    if STATIC.is_dir():
        assets = STATIC / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC / "index.html")

    return app


app = create_app()
