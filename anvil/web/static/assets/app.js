/* Anvil dashboard client */

const state = {
  selectedRunId: null,
  overview: null,
  defaults: null,
  suggest: null,
  selectedPattern: null,
  selectedRecipeId: null,
  selectedPlan: null,
};

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || JSON.stringify(j);
    } catch (_) {}
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  return res.json();
}

function $(id) {
  return document.getElementById(id);
}

function fmtBytes(n) {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  const u = ["KB", "MB", "GB", "TB"];
  let i = -1;
  do {
    n /= 1024;
    i++;
  } while (n >= 1024 && i < u.length - 1);
  return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${u[i]}`;
}

function setArc(el, frac) {
  const max = 157; // approx semicircle path length
  const f = Math.max(0, Math.min(1, frac));
  el.style.strokeDasharray = String(max);
  el.style.strokeDashoffset = String(max * (1 - f));
}

function drawLossChart(history) {
  const canvas = $("loss-chart");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // grid
  ctx.strokeStyle = "#1a1a1e";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h * i) / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  if (!history || history.length < 1) {
    ctx.fillStyle = "#52525b";
    ctx.font = "12px JetBrains Mono, monospace";
    ctx.fillText("no steps yet", 12, h / 2);
    return;
  }

  const losses = history.map((p) => p.loss);
  let min = Math.min(...losses);
  let max = Math.max(...losses);
  if (min === max) {
    min -= 0.1;
    max += 0.1;
  }
  const pad = 8;
  const pts = history.map((p, i) => {
    const x = pad + ((w - 2 * pad) * i) / Math.max(history.length - 1, 1);
    const y = pad + (h - 2 * pad) * (1 - (p.loss - min) / (max - min));
    return [x, y];
  });

  // fill
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(118,185,0,0.25)");
  grad.addColorStop(1, "rgba(118,185,0,0)");
  ctx.beginPath();
  ctx.moveTo(pts[0][0], h);
  pts.forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.lineTo(pts[pts.length - 1][0], h);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // line
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.strokeStyle = "#76b900";
  ctx.lineWidth = 2;
  ctx.stroke();

  // last point
  const [lx, ly] = pts[pts.length - 1];
  ctx.beginPath();
  ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
  ctx.fillStyle = "#a4d930";
  ctx.fill();
}

function formKnobs(form) {
  const fd = new FormData(form);
  const modalities = [];
  if (fd.get("mod_text")) modalities.push("text");
  if (fd.get("mod_image")) modalities.push("image");
  return {
    base_model: String(fd.get("base_model") || ""),
    rank: Number(fd.get("rank")),
    learning_rate: Number(fd.get("learning_rate")),
    loss_fn: String(fd.get("loss_fn")),
    max_steps: Number(fd.get("max_steps")),
    batch_size: Number(fd.get("batch_size")),
    seq_len: Number(fd.get("seq_len")),
    max_tokens: Number(fd.get("max_tokens")),
    temperature: Number(fd.get("temperature")),
    language_lora: Boolean(fd.get("language_lora")),
    mm_projector_lora: Boolean(fd.get("mm_projector_lora")),
    vision_encoder_lora: Boolean(fd.get("vision_encoder_lora")),
    modalities,
  };
}

function applyPlanToForm(plan) {
  const form = $("run-form");
  const k = plan.knobs || plan;
  const set = (name, val) => {
    const el = form.elements.namedItem(name);
    if (!el) return;
    if (el.type === "checkbox") el.checked = Boolean(val);
    else el.value = val;
  };
  set("base_model", plan.base_model || k.base_model);
  set("rank", k.rank);
  set("learning_rate", k.learning_rate);
  set("loss_fn", k.loss_fn);
  set("max_steps", k.max_steps);
  set("batch_size", k.batch_size);
  set("seq_len", k.seq_len);
  set("max_tokens", k.max_tokens);
  set("temperature", k.temperature);
  set("language_lora", k.language_lora);
  set("mm_projector_lora", k.mm_projector_lora);
  set("vision_encoder_lora", k.vision_encoder_lora);
  const mods = k.modalities || [];
  form.elements.namedItem("mod_text").checked = mods.includes("text");
  form.elements.namedItem("mod_image").checked = mods.includes("image");
  $("pattern-field").value = plan.pattern || "";
  const lines = [];
  (plan.rationale || []).forEach((r) => lines.push(`• ${r}`));
  (plan.cautions || []).forEach((c) => lines.push(`! ${c}`));
  if (plan.export_hint) lines.push(`export → ${plan.export_hint}`);
  $("rationale-box").textContent = lines.join("\n") || "";
}

function gateClass(level) {
  if (level === "recommended") return "gate-ok";
  if (level === "stretch") return "gate-stretch";
  return "gate-block";
}

function renderRecipeCards(suggest) {
  const root = $("recipe-cards");
  root.innerHTML = "";
  if (!suggest) return;
  $("shape-tag").textContent = suggest.shape;
  $("shape-label").textContent =
    (suggest.shape_label || "") +
    (suggest.default_recipe_id ? ` · default ${suggest.default_recipe_id}` : "");
  for (const card of suggest.recipes || []) {
    const rid = card.recipe_id || card.pattern;
    const gate = card.gate || {};
    const level = gate.level || "recommended";
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className =
      "recipe-card" +
      (state.selectedRecipeId === rid ? " active" : "") +
      (level === "blocked" ? " blocked" : "");
    const p = card.plan || {};
    const kn = p.knobs || p;
    btn.innerHTML = `
      <div class="rc-title">
        ${escapeHtml(card.title)}
        <span class="gate-pill ${gateClass(level)}">${escapeHtml(level)}</span>
      </div>
      <div class="rc-sum">${escapeHtml(card.summary)}</div>
      <div class="rc-meta">${escapeHtml(rid)} · loss ${escapeHtml(
        p.loss_fn || kn.loss_fn || ""
      )} · rank ${p.lora ? p.lora.rank : kn.rank || "?"} · lr ${
        p.learning_rate ?? kn.learning_rate ?? "?"
      } · export ${escapeHtml(p.export_hint || "")}</div>`;
    btn.addEventListener("click", () => {
      if (level === "blocked" && !card.plan) {
        alert(
          "Blocked for this architecture:\n" +
            (gate.blocked_reasons || []).join("\n") +
            "\n\nStretch past the gate only if you know why (API force=true)."
        );
        return;
      }
      state.selectedRecipeId = rid;
      state.selectedPattern = card.pattern;
      state.selectedPlan = card.plan;
      if (card.plan) applyPlanToForm(card.plan);
      $("pattern-field").value = rid;
      renderRecipeCards(suggest);
    });
    root.appendChild(btn);
  }
}

async function refreshSuggest() {
  const base = $("base-model").value;
  if (!base) return;
  try {
    const s = await api(`/api/suggest?base_model=${encodeURIComponent(base)}`);
    state.suggest = s;
    const preferred =
      s.default_recipe_id &&
      (s.recipes || []).find(
        (r) => r.recipe_id === s.default_recipe_id && r.plan && r.gate?.level === "recommended"
      );
    if (!state.selectedRecipeId && (preferred || s.recipes?.length)) {
      const pick =
        preferred ||
        (s.recipes || []).find((r) => r.plan && r.gate?.level === "recommended") ||
        (s.recipes || []).find((r) => r.plan);
      if (pick) {
        state.selectedRecipeId = pick.recipe_id;
        state.selectedPattern = pick.pattern;
        state.selectedPlan = pick.plan;
        applyPlanToForm(pick.plan);
        $("pattern-field").value = pick.recipe_id || pick.pattern;
      }
    } else if (state.selectedRecipeId) {
      const match = (s.recipes || []).find((r) => r.recipe_id === state.selectedRecipeId);
      if (match?.plan) {
        state.selectedPlan = match.plan;
        applyPlanToForm(match.plan);
      }
    }
    renderRecipeCards(s);
  } catch (e) {
    console.error(e);
  }
}

function enableActions(on) {
  ["btn-train-1", "btn-train-10", "btn-sample", "btn-ckpt", "btn-export"].forEach((id) => {
    $(id).disabled = !on;
  });
}

function renderRunDetail(run) {
  if (!run) {
    $("selected-run-label").textContent = "no run selected";
    $("run-log").textContent = "select or create a run";
    setArc($("arc-step"), 0);
    setArc($("arc-progress"), 0);
    $("gauge-step").textContent = "0";
    $("gauge-pct").textContent = "0%";
    drawLossChart([]);
    enableActions(false);
    return;
  }
  $("selected-run-label").textContent = `${run.name} · ${run.run_id}`;
  $("run-log").textContent = (run.logs || []).join("\n") || "(no logs)";
  const max = run.knobs.max_steps || 1;
  const frac = run.step / max;
  setArc($("arc-step"), frac);
  setArc($("arc-progress"), frac);
  $("gauge-step").textContent = `${run.step}/${max}`;
  $("gauge-pct").textContent = `${Math.round(frac * 100)}%`;
  drawLossChart(run.history || []);
  $("stat-loss").textContent =
    run.last_loss == null ? "—" : Number(run.last_loss).toFixed(4);
  enableActions(true);
}

function renderRuns(runs) {
  const body = $("runs-body");
  body.innerHTML = "";
  for (const r of runs) {
    const tr = document.createElement("tr");
    if (r.run_id === state.selectedRunId) tr.classList.add("selected");
    tr.innerHTML = `
      <td>${escapeHtml(r.name)}</td>
      <td><span class="status ${escapeHtml(r.status)}">${escapeHtml(r.status)}</span></td>
      <td>${r.step}</td>
      <td>${r.last_loss == null ? "—" : Number(r.last_loss).toFixed(4)}</td>
      <td title="${escapeHtml(r.knobs.base_model)}">${escapeHtml(
        shortModel(r.knobs.base_model)
      )}</td>`;
    tr.addEventListener("click", () => {
      state.selectedRunId = r.run_id;
      renderRunDetail(r);
      renderRuns(runs);
    });
    body.appendChild(tr);
  }
}

function renderModels(models) {
  const body = $("models-body");
  body.innerHTML = "";
  for (const m of models) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td title="${escapeHtml(m.path || m.name)}">${escapeHtml(m.name)}</td>
      <td>${escapeHtml(m.source)}</td>
      <td>${fmtBytes(m.size_bytes)}</td>`;
    body.appendChild(tr);
  }
  $("stat-models").textContent = String(models.filter((m) => m.source === "models_root").length);
}

function shortModel(id) {
  if (!id) return "—";
  const parts = id.split("/");
  return parts[parts.length - 1];
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function refresh() {
  try {
    const ov = await api("/api/overview");
    state.overview = ov;
    $("pill-host").textContent = `host ${ov.host}`;
    $("pill-backend").textContent = `backend ${ov.backend}`;
    $("pill-conn").textContent = "live";
    $("pill-conn").className = "pill pill-ok";
    $("stat-active").textContent = ov.active_runs;
    $("stat-done").textContent = ov.completed_runs;
    $("stat-total").textContent = ov.total_runs;
    $("footer-version").textContent = `anvil ${ov.version}`;
    $("footer-ts").textContent = new Date(ov.ts * 1000).toLocaleTimeString();
    if (ov.spark_dashboard) {
      $("link-spark-forge").href = ov.spark_dashboard.forge || "#";
      $("link-spark-hammer").href = ov.spark_dashboard.hammer || "#";
    }
    renderRuns(ov.runs || []);
    renderModels(ov.models || []);
    if (state.selectedRunId) {
      const run = (ov.runs || []).find((r) => r.run_id === state.selectedRunId);
      renderRunDetail(run || null);
    }
  } catch (e) {
    $("pill-conn").textContent = "offline";
    $("pill-conn").className = "pill pill-warn";
    console.error(e);
  }
}

async function initDefaults() {
  const d = await api("/api/defaults");
  state.defaults = d;
  $("models-root").textContent = d.models_root;
  const bm = $("base-model");
  bm.innerHTML = "";
  for (const m of d.base_models) {
    const o = document.createElement("option");
    o.value = m;
    o.textContent = m;
    bm.appendChild(o);
  }
  const lf = $("loss-fn");
  lf.innerHTML = "";
  for (const l of d.loss_choices) {
    const o = document.createElement("option");
    o.value = l;
    o.textContent = l;
    lf.appendChild(o);
  }
  if (d.spark_dashboard) {
    $("link-spark-forge").href = d.spark_dashboard.forge;
    $("link-spark-hammer").href = d.spark_dashboard.hammer;
  }
  await refreshSuggest();
}

function bind() {
  $("run-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const form = ev.target;
    const name = new FormData(form).get("name") || null;
    const knobs = formKnobs(form);
    const recipeId = $("pattern-field").value || state.selectedRecipeId || null;
    const shape = state.suggest?.shape || null;
    const rationale = state.selectedPlan?.rationale || [];
    const gateLevel = state.selectedPlan?.gate?.level;
    try {
      const run = await api("/api/runs", {
        method: "POST",
        body: JSON.stringify({
          name: name || null,
          knobs,
          recipe_id: recipeId,
          pattern: state.selectedPattern || null,
          shape,
          rationale,
          force: gateLevel === "stretch" || gateLevel === "blocked",
        }),
      });
      state.selectedRunId = run.run_id;
      await refresh();
    } catch (e) {
      alert(`create failed: ${e.message}`);
    }
  });

  $("base-model").addEventListener("change", () => {
    state.selectedPattern = null;
    state.selectedRecipeId = null;
    state.selectedPlan = null;
    refreshSuggest();
  });

  $("btn-train-1").addEventListener("click", () => train(1));
  $("btn-train-10").addEventListener("click", () => train(10));
  $("btn-sample").addEventListener("click", async () => {
    if (!state.selectedRunId) return;
    try {
      const out = await api(`/api/runs/${state.selectedRunId}/sample`, {
        method: "POST",
        body: "{}",
      });
      alert(`sample: ${out.n_tokens} tokens (toy ids) stop=${out.stop_reason}`);
      await refresh();
    } catch (e) {
      alert(e.message);
    }
  });
  $("btn-ckpt").addEventListener("click", async () => {
    if (!state.selectedRunId) return;
    try {
      const out = await api(`/api/runs/${state.selectedRunId}/checkpoint`, {
        method: "POST",
        body: "{}",
      });
      alert(`checkpoint ${out.name}\n${out.path}`);
      await refresh();
    } catch (e) {
      alert(e.message);
    }
  });
  $("btn-export").addEventListener("click", async () => {
    if (!state.selectedRunId) return;
    try {
      const out = await api(`/api/runs/${state.selectedRunId}/export`, {
        method: "POST",
        body: JSON.stringify({ format: "peft" }),
      });
      alert(`exported ${out.format}\n${out.path}`);
      await refresh();
    } catch (e) {
      alert(e.message);
    }
  });

  window.addEventListener("resize", () => {
    if (state.selectedRunId && state.overview) {
      const run = (state.overview.runs || []).find((r) => r.run_id === state.selectedRunId);
      if (run) drawLossChart(run.history || []);
    }
  });
}

async function train(steps) {
  if (!state.selectedRunId) return;
  try {
    await api(`/api/runs/${state.selectedRunId}/train`, {
      method: "POST",
      body: JSON.stringify({ steps }),
    });
    await refresh();
  } catch (e) {
    alert(e.message);
  }
}

(async function main() {
  bind();
  try {
    await initDefaults();
  } catch (e) {
    console.error(e);
  }
  await refresh();
  setInterval(refresh, 2000);
})();
