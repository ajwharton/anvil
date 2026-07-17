"""Anvil web control plane — FastAPI + spark-dashboard-inspired UI.

Run::

    pip install -e ".[web]"
    anvil-web --host 0.0.0.0 --port 7600
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from anvil.observe.metrics import METRICS_FILENAME, PROBES_FILENAME, read_jsonl

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


# --- P2.5 run observability (metrics.jsonl / probes.jsonl tailing) ---------

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _observe_root() -> Path:
    return Path(
        os.environ.get("ANVIL_OBSERVE_ROOT", str(Path.home() / ".anvil" / "observe"))
    )


def _observe_run_dir(run_id: str) -> Path:
    if not _SAFE_RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise HTTPException(400, f"bad run id: {run_id!r}")
    d = _observe_run_dir_root(run_id)
    if not d.is_dir():
        raise HTTPException(404, f"no observe dir for run {run_id!r}")
    return d


def _observe_run_dir_root(run_id: str) -> Path:
    return _observe_root() / run_id


def _observe_html(run_id: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>anvil observe — {run_id}</title>
<style>
 body{{background:#0d1117;color:#c9d1d9;font:14px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif;margin:0;padding:24px;max-width:980px}}
 h1{{font-size:18px;color:#58a6ff;margin:0 0 16px}}
 h2{{font-size:13px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;margin:0 0 10px}}
 .card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:0 0 16px}}
 #trip{{display:none;background:#3d1d1d;border:1px solid #f85149;color:#f85149;padding:8px 12px;border-radius:6px;margin-bottom:16px;font-weight:600}}
 .probe{{border-top:1px solid #30363d;padding:8px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre-wrap;word-break:break-word}}
 .meta{{color:#8b949e;font-size:12px;margin-top:8px}}
 canvas{{width:100%;background:#0d1117;border:1px solid #21262d;border-radius:4px}}
</style></head><body>
<h1>anvil observe — {run_id}</h1>
<div id="trip">&#9888; ADVANTAGE COLLAPSED — group reward std &asymp; 0, gradient signal dead</div>
<div class="card"><h2>reward_mean/step (blue) &middot; group reward std (orange)</h2>
<canvas id="chart" width="900" height="220"></canvas>
<div class="meta" id="laststep">waiting for first step&hellip;</div></div>
<div class="card"><h2>probes — live policy</h2><div id="probes" class="meta">no probes yet</div></div>
<script>
const rid = {json.dumps(run_id)};
const EPS = 1e-8;
let recs = [];
function draw() {{
  const c = document.getElementById('chart'), ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  if (!recs.length) return;
  const pad = 30, W = c.width - 2*pad, H = c.height - 2*pad;
  const vals = recs.flatMap(r => [r.reward_mean || 0, r.group_reward_std_mean || 0]);
  const maxV = Math.max(1e-9, ...vals), minV = Math.min(0, ...vals);
  const n = Math.max(recs.length - 1, 1);
  const x = i => pad + W*i/n, y = v => pad + H*(1 - (v - minV)/(maxV - minV));
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, y(0)); ctx.lineTo(pad + W, y(0)); ctx.stroke();
  for (const s of [{{key:'reward_mean', color:'#58a6ff'}}, {{key:'group_reward_std_mean', color:'#f0883e'}}]) {{
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    recs.forEach((r, i) => {{
      const v = r[s.key]; if (v == null) return;
      if (started) ctx.lineTo(x(i), y(v)); else {{ ctx.moveTo(x(i), y(v)); started = true; }}
    }});
    ctx.stroke();
  }}
  const last = recs[recs.length - 1];
  document.getElementById('laststep').textContent =
    'step ' + last.step + ' · reward_mean ' + last.reward_mean.toFixed(4)
    + ' · group_std ' + last.group_reward_std_mean.toFixed(6)
    + ' · loss ' + (last.loss == null ? '—' : last.loss.toFixed(5))
    + (last.is_mean_ratio != null ? ' · IS ratio ' + last.is_mean_ratio.toFixed(4) : '');
  document.getElementById('trip').style.display = (last.group_reward_std_mean < EPS) ? '' : 'none';
}}
function loadProbes() {{
  fetch('/api/observe/' + rid + '/probes?tail=24').then(r => r.ok ? r.json() : null).then(d => {{
    if (!d || !d.probes.length) return;
    const el = document.getElementById('probes');
    el.classList.remove('meta'); el.innerHTML = '';
    d.probes.slice().reverse().forEach(p => {{
      const div = document.createElement('div'); div.className = 'probe';
      const txt = p.text != null ? p.text : '[' + p.tokens.join(', ') + ']';
      div.textContent = 'step ' + p.step + ' · probe ' + p.probe_idx
        + (p.reward != null ? ' · r=' + p.reward : '') + '\\n' + txt;
      el.appendChild(div);
    }});
  }});
}}
const es = new EventSource('/api/observe/' + rid + '/metrics/stream');
es.onmessage = ev => {{ recs.push(JSON.parse(ev.data)); draw(); }};
loadProbes(); setInterval(loadProbes, 4000);
</script></body></html>"""


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

    @app.get("/api/observe")
    def observe_runs() -> dict[str, Any]:
        root = _observe_root()
        runs = (
            sorted(
                p.name
                for p in root.iterdir()
                if p.is_dir() and (p / METRICS_FILENAME).exists()
            )
            if root.is_dir()
            else []
        )
        return {"root": str(root), "runs": runs}

    @app.get("/api/observe/{run_id}/metrics")
    def observe_metrics(run_id: str, tail: int = 500) -> dict[str, Any]:
        d = _observe_run_dir(run_id)
        return {"run_id": run_id, "metrics": read_jsonl(d / METRICS_FILENAME, tail=tail)}

    @app.get("/api/observe/{run_id}/probes")
    def observe_probes(run_id: str, tail: int = 200) -> dict[str, Any]:
        d = _observe_run_dir(run_id)
        return {"run_id": run_id, "probes": read_jsonl(d / PROBES_FILENAME, tail=tail)}

    @app.get("/api/observe/{run_id}/metrics/stream")
    async def observe_metrics_stream(run_id: str, request: Request) -> StreamingResponse:
        path = _observe_run_dir(run_id) / METRICS_FILENAME

        async def gen():
            sent = 0
            while True:
                if await request.is_disconnected():
                    return
                records = read_jsonl(path)
                for rec in records[sent:]:
                    yield f"data: {json.dumps(rec)}\n\n"
                sent = len(records)
                yield ": hb\n\n"
                await asyncio.sleep(1.5)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/observe/{run_id}", response_class=HTMLResponse)
    def observe_page(run_id: str) -> str:
        _observe_run_dir(run_id)  # 400/404 on bad or unknown ids
        return _observe_html(run_id)

    if STATIC.is_dir():
        assets = STATIC / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC / "index.html")

    return app


app = create_app()
