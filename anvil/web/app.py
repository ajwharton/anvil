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

from anvil.observe.metrics import (
    JLENS_FILENAME,
    METRICS_FILENAME,
    PROBES_FILENAME,
    read_jsonl,
)
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


class PatchKnobsIn(BaseModel):
    knobs: dict[str, Any] = Field(default_factory=dict)


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
 #stop{{display:none;background:#1d2a3d;border:1px solid #58a6ff;color:#58a6ff;padding:8px 12px;border-radius:6px;margin-bottom:16px;font-weight:600}}
 .probe{{border-top:1px solid #30363d;padding:8px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:pre-wrap;word-break:break-word}}
 .meta{{color:#8b949e;font-size:12px;margin-top:8px}}
 canvas{{width:100%;background:#0d1117;border:1px solid #21262d;border-radius:4px}}
</style></head><body>
<h1>anvil observe — {run_id}</h1>
<div id="trip">&#9888; ADVANTAGE COLLAPSED — group reward std &asymp; 0, gradient signal dead</div>
<div id="stop">&#9632; EARLY STOP — run abandoned (dead signal streak); no more power burn</div>
<div class="card"><h2 id="chart-title">metrics / step</h2>
<canvas id="chart" width="900" height="220"></canvas>
<div class="meta" id="laststep">waiting for first step&hellip;</div></div>
<div class="card"><h2>probes — live policy</h2><div id="probes" class="meta">no probes yet</div></div>
<script>
const rid = {json.dumps(run_id)};
const EPS = 1e-8;
let recs = [];
let stopEvt = null;
function stepRecs() {{
  return recs.filter(r => r.type === 'step'
    || (r.type !== 'event' && (r.reward_mean != null || r.loss != null)));
}}
function isSft(steps) {{
  if (!steps.length) return false;
  const j = steps[steps.length - 1].job;
  if (j === 'sft' || j === 'vlm_sft') return true;
  // legacy: SFT records have loss but no reward_mean
  return steps[steps.length - 1].reward_mean == null
    && steps[steps.length - 1].loss != null;
}}
function draw() {{
  const c = document.getElementById('chart'), ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const steps = stepRecs();
  if (!steps.length) return;
  const sft = isSft(steps);
  document.getElementById('chart-title').textContent = sft
    ? 'loss / step (blue)' + (steps.some(r => r.n_image_refs) ? ' · n_image_refs on last step' : '')
    : 'reward_mean/step (blue) · group reward std (orange)';
  const pad = 30, W = c.width - 2*pad, H = c.height - 2*pad;
  const series = sft
    ? [{{key:'loss', color:'#58a6ff'}}]
    : [{{key:'reward_mean', color:'#58a6ff'}}, {{key:'group_reward_std_mean', color:'#f0883e'}}];
  const vals = steps.flatMap(r => series.map(s => r[s.key]).filter(v => v != null));
  if (!vals.length) return;
  const maxV = Math.max(...vals), minV = Math.min(...vals);
  const span = Math.max(1e-9, maxV - minV);
  const n = Math.max(steps.length - 1, 1);
  const x = i => pad + W*i/n, y = v => pad + H*(1 - (v - minV)/span);
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  const zeroInRange = minV <= 0 && maxV >= 0;
  if (zeroInRange) {{
    ctx.beginPath(); ctx.moveTo(pad, y(0)); ctx.lineTo(pad + W, y(0)); ctx.stroke();
  }}
  for (const s of series) {{
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    steps.forEach((r, i) => {{
      const v = r[s.key]; if (v == null) return;
      if (started) ctx.lineTo(x(i), y(v)); else {{ ctx.moveTo(x(i), y(v)); started = true; }}
    }});
    ctx.stroke();
  }}
  const last = steps[steps.length - 1];
  let sync = '';
  if (last.adapter_synced === true) sync = ' · adapter SYNC';
  else if (last.adapter_synced === false) sync = ' · adapter held';
  let meta = 'step ' + last.step
    + (last.job ? ' · ' + last.job : '')
    + ' · loss ' + (last.loss == null ? '—' : Number(last.loss).toFixed(5));
  if (!sft) {{
    meta += ' · reward_mean ' + (last.reward_mean == null ? '—' : Number(last.reward_mean).toFixed(4))
      + ' · group_std ' + (last.group_reward_std_mean == null ? '—' : Number(last.group_reward_std_mean).toFixed(6));
  }} else {{
    if (last.n_image_refs != null) meta += ' · n_image_refs ' + last.n_image_refs;
    if (last.n_tokens != null) meta += ' · n_tokens ' + last.n_tokens;
    if (last.wall_time_s != null) meta += ' · wall ' + Number(last.wall_time_s).toFixed(3) + 's';
  }}
  if (last.is_mean_ratio != null) meta += ' · IS ratio ' + Number(last.is_mean_ratio).toFixed(4);
  meta += sync
    + (last.sample_endpoint ? ' · sample ' + last.sample_endpoint : '')
    + (stopEvt ? ' · EARLY STOP ' + (stopEvt.reason || '') : '');
  document.getElementById('laststep').textContent = meta;
  const collapsed = !sft && last.group_reward_std_mean != null
    && last.group_reward_std_mean < EPS;
  document.getElementById('trip').style.display = collapsed ? '' : 'none';
  document.getElementById('stop').style.display = stopEvt ? '' : 'none';
  if (stopEvt) document.getElementById('stop').textContent =
    '■ EARLY STOP — ' + (stopEvt.reason || 'dead signal') + ' at step ' + stopEvt.step
    + ' (abandoned; no further train steps)';
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
es.onmessage = ev => {{
  const rec = JSON.parse(ev.data);
  recs.push(rec);
  if (rec.type === 'event' && rec.event === 'early_stop') stopEvt = rec;
  draw();
}};
loadProbes(); setInterval(loadProbes, 4000);
</script></body></html>"""


def create_app() -> FastAPI:
    app = FastAPI(
        title="Anvil",
        description="Anvil post-training control plane",
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
        include_personal_book: bool = True,
    ) -> dict[str, Any]:
        """Shape + gated catalog + personal book matches for a base model."""
        return suggest_for_model(
            base_model,
            fetch_remote=fetch_remote,
            include_blocked=include_blocked,
            include_personal_book=include_personal_book,
        )

    @app.get("/api/recipe-book")
    def recipe_book_list(family: str | None = None, pattern: str | None = None) -> dict[str, Any]:
        """List personal recipe book entries (sovereign local store)."""
        from anvil.recipes.book import RecipeBook

        book = RecipeBook()
        rows = book.prefer(family=family, pattern=pattern)
        return {
            "root": str(book.root),
            "recipes": [r.to_public() for r in rows],
        }

    @app.get("/api/recipe-book/{recipe_id}")
    def recipe_book_get(recipe_id: str) -> dict[str, Any]:
        from anvil.recipes.book import RecipeBook

        rec = RecipeBook().get(recipe_id)
        if rec is None:
            raise HTTPException(404, f"no personal recipe {recipe_id!r}")
        return rec.to_public()

    @app.get("/api/meta-recipes")
    def meta_recipes_list() -> dict[str, Any]:
        """List meta-recipes (stage graphs) in the personal book meta/ dir."""
        from anvil.recipes.meta import list_meta_recipes, meta_book_dir

        rows = list_meta_recipes()
        return {
            "root": str(meta_book_dir()),
            "meta_recipes": [m.to_public() for m in rows],
        }

    @app.get("/api/meta-recipes/{meta_id}")
    def meta_recipes_get(meta_id: str) -> dict[str, Any]:
        from anvil.recipes.meta import get_meta_recipe

        try:
            m = get_meta_recipe(meta_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if m is None:
            raise HTTPException(404, f"no meta-recipe {meta_id!r}")
        return m.to_public()

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
        # Recipe/pattern path: re-derive plan (knobs = overrides; gates enforced).
        # RL debugger fields are UI/run-level — not part of RecipePlan — so keep them.
        _RL_KNOB_KEYS = (
            "probe_every",
            "sync_every",
            "sample_endpoint",
            "sample_adapter_id",
            "write_metrics",
        )
        rl_keep = {k: knobs[k] for k in _RL_KNOB_KEYS if k in knobs}
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
                knobs = {**plan.as_knobs(), **rl_keep}
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
        else:
            knobs = {**knobs, **rl_keep}
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

    @app.post("/api/runs/{run_id}/pause")
    def pause(run_id: str) -> dict[str, Any]:
        """Live control: pause a run (agent/MCP)."""
        try:
            return store.pause_run(run_id).to_public()
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str) -> dict[str, Any]:
        try:
            return store.resume_run(run_id).to_public()
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    @app.patch("/api/runs/{run_id}/knobs")
    def patch_knobs(run_id: str, payload: PatchKnobsIn) -> dict[str, Any]:
        """Live control: patch knobs mid-run (logged on the run)."""
        try:
            return store.patch_knobs(run_id, payload.knobs).to_public()
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except (RuntimeError, ValueError) as e:
            raise HTTPException(409 if isinstance(e, RuntimeError) else 400, str(e)) from e

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
        """List disk observe runs (GRPO + SFT/VLM metrics.jsonl).

        Each entry includes last metrics step when available so the control-plane
        UI can deep-link without polling every metrics file twice.
        """
        root = _observe_root()
        runs: list[dict[str, Any]] = []
        if root.is_dir():
            for p in sorted(root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if not p.is_dir():
                    continue
                metrics_path = p / METRICS_FILENAME
                if not metrics_path.exists() and not (p / PROBES_FILENAME).exists():
                    continue
                recs = read_jsonl(metrics_path) if metrics_path.exists() else []
                last = recs[-1] if recs else None
                runs.append(
                    {
                        "run_id": p.name,
                        "path": str(p),
                        "n_steps": len(recs),
                        "last": last,
                        "mtime": p.stat().st_mtime,
                        "observe_url": f"/observe/{p.name}",
                    }
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

    @app.get("/api/observe/{run_id}/jlens")
    def observe_jlens(run_id: str, tail: int = 100) -> dict[str, Any]:
        """Tail jlens.jsonl residual-readout records (J1 artifact path; panel later)."""
        d = _observe_run_dir(run_id)
        return {"run_id": run_id, "jlens": read_jsonl(d / JLENS_FILENAME, tail=tail)}

    @app.get("/api/observe/{run_id}/metrics/stream")
    async def observe_metrics_stream(
        run_id: str, request: Request, once: bool = False
    ) -> StreamingResponse:
        """SSE tail of metrics.jsonl. ``once=true`` emits current lines and exits
        (for tests / one-shot clients); default is a live poll loop."""
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
                if once:
                    return
                yield ": hb\n\n"
                await asyncio.sleep(1.5)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/observe", response_class=HTMLResponse)
    def observe_index() -> str:
        """Landing page for live post-training runs on disk (GRPO + SFT/VLM)."""
        data = observe_runs()
        rows = []
        for r in data["runs"]:
            last = r.get("last") or {}
            job = last.get("job") or ""
            rew = last.get("reward_mean")
            loss = last.get("loss")
            if isinstance(rew, (int, float)):
                signal_s = f"r={rew:.3f}"
            elif isinstance(loss, (int, float)):
                signal_s = f"loss={loss:.4f}"
            else:
                signal_s = "—"
            if job:
                signal_s = f"{job} · {signal_s}"
            step = last.get("step") if last else None
            step_s = str(step) if step is not None else "—"
            rows.append(
                f'<tr><td><a href="{r["observe_url"]}">{r["run_id"]}</a></td>'
                f'<td class="mono">{step_s}</td><td class="mono">{signal_s}</td>'
                f'<td class="mono">{r["n_steps"]}</td></tr>'
            )
        body = (
            "\n".join(rows)
            if rows
            else '<tr><td colspan="4" class="meta">no runs yet — start '
            "<code>scripts/grpo_observe_demo.py</code> or "
            "<code>scripts/vlm_smoke.py --run-id …</code> with this "
            f"ANVIL_OBSERVE_ROOT ({data['root']})</td></tr>"
        )
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>anvil observe</title>
<style>
 body{{background:#0d1117;color:#c9d1d9;font:14px/1.5 -apple-system,sans-serif;margin:0;padding:24px;max-width:900px}}
 h1{{font-size:18px;color:#58a6ff}} a{{color:#58a6ff}}
 table{{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:8px}}
 th,td{{padding:10px 12px;border-bottom:1px solid #30363d;text-align:left}}
 th{{color:#8b949e;font-size:12px;text-transform:uppercase}}
 .mono{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
 .meta{{color:#8b949e}} code{{background:#21262d;padding:2px 6px;border-radius:4px}}
</style></head><body>
<h1>anvil observe — live runs</h1>
<p class="meta">root <code>{data["root"]}</code> · auto-refresh 5s ·
<a href="/">control plane</a></p>
<table><thead><tr><th>run</th><th>last step</th><th>signal</th><th>n_steps</th></tr></thead>
<tbody>{body}</tbody></table>
<script>setTimeout(() => location.reload(), 5000);</script>
</body></html>"""

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
