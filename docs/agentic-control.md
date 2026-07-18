# Agentic control — what Anvil owns vs what you bring

**Related:** [`product.md`](product.md) (thesis) · [`roadmap.md`](roadmap.md)  
**Date:** 2026-07-17  
**Status:** product note — design intent for the agent surface

---

## Position

Anvil should **own the control plane and the agent harness shape**—not the
frontier (or local) **brain** that does the watching and deciding.

| Layer | Who provides it | What it is |
|-------|-----------------|------------|
| **Train substrate** | Anvil + your GPUs | Four verbs, recipes, backends, media, observe artifacts |
| **Control / observe APIs** | Anvil | HTTP (+ SSE), audits, gates, run lifecycle |
| **MCP tool surface** | **Anvil** | Thin, stable tools over those APIs (list / plan / run / tail / act / audit) |
| **Optional harness** | **Anvil** | A small loop: tool schemas, session policy, stop conditions, generic prompts |
| **Agent model (“brain”)** | **You** | Any capable model—frontier API or local—plugged into the harness or into *your* harness |
| **Policy / red lines** | **You** | What the agent may force, spend, export, or switch |

We do **not** ship “Anvil’s private agent model” as the product. We ship
**hooks for good agents** and defaults so people are not stuck inventing
prompt engineering from zero.

---

## Why this split

1. **Brains move fast.** Frontier models improve weekly; pinning Anvil to one
   vendor model is a product trap.  
2. **The hard platform problem is the loop.** Watch cliffs → decide method
   switch → act with audit. That needs tools, streams, and recipes—not
   another chat app.  
3. **Harness + prompts are hard.** Most users will not invent a good
   training-operator agent. Shipping a **generic prompt pack** + optional
   harness lowers the floor; experts can replace either.  
4. **Trust.** Humans keep the keys to the model account and the spend
   limits; Anvil keeps the train surface honest and auditable.

---

## What Anvil wraps (we build this)

### 1. MCP server (primary agent interface)

Tools that mirror the product SSOT (same data as `anvil-web` / observe):

| Tool family | Examples |
|-------------|---------|
| **Discover** | `list_recipes`, `suggest_for_model`, `gate_recipe`, `list_runs` |
| **Plan / start** | `plan_recipe`, `create_run`, `start_sft` / `start_grpo` (or verb-level) |
| **Watch** | `tail_metrics`, `tail_probes`, `get_run`, stream subscribe |
| **Act** | `patch_knobs`, `pause_run`, `sync_adapter`, `switch_recipe`, `export_adapter`, `stop_run` |
| **Account** | `list_audit`, `get_tripwires` |

Implementation principle: **MCP is a façade over HTTP/JSON already used by
the UI**—no second brain for run state.

### 2. Optional Anvil agent harness

A thin, replaceable loop (process or library):

```text
  your model (config: API base, model id, keys in env)
           │
           ▼
  Anvil harness  ──system/policy prompts──►  tool calls
           │                                    │
           │                                    ▼
           │                              Anvil MCP / HTTP
           │                                    │
           └──────────── observations ◄─────────┘
```

Harness responsibilities:

- Load **generic system + operator prompts** from the repo (see below).  
- Expose **MCP tools** (or HTTP) to the model.  
- Enforce **policy**: max steps, max spend hooks, deny-list of force-gates
  unless human pre-approved, always-on audit.  
- Emit a **transcript** of decisions (why we switched DPO → GRPO, etc.).

Harness non-responsibilities:

- Choosing which frontier model is “best.”  
- Hiding the four verbs behind a proprietary agent API.  
- Running without a human-accessible kill switch.

### 3. Generic prompt pack (works *with or without* our harness)

Prompt engineering for “training operator” agents is genuinely hard. Anvil
should ship **versioned, boring, high-quality prompts** that people can:

- use **as-is** inside the Anvil harness, or  
- **copy into** Cursor / Claude Code / custom orchestrators / multi-agent
  graphs that already talk MCP or HTTP.

Prompt pack goals:

| Prompt | Role |
|--------|------|
| `system_operator.md` | Role: post-training operator; prefer recipes; respect gates; never invent metrics |
| `watch_loop.md` | How to read metrics/probes; what a cliff looks like; when to wait vs act |
| `method_switch.md` | Conservative playbook edges (e.g. preference stall → try on-policy; collapse → lower LR / SFT recovery)—**suggestions**, not hard rules |
| `safety_policy.md` | No silent force; log overrides; don’t export secrets; stop on human interrupt |

Prompts must reference **tool names and artifact schemas** that match Anvil
(so they stay accurate). When APIs change, prompts ship in the same PR.

Location (target): `prompts/agent/` in-repo, linked from this doc.

---

## What you bring

| You bring | Notes |
|-----------|--------|
| **Agent model** | Frontier (OpenAI / Anthropic / xAI / …) or local (vLLM, Ollama, lab host) |
| **Credentials** | API keys via env / secret store—not committed to Anvil |
| **Compute** | Lab GPUs for train/sample (forge/hammer pattern) |
| **Data** | Preference pairs, GRPO envs, VLM frames, robot trajectories |
| **Policy** | Max cost, allowed force recipes, export destinations |

Anvil documents **how to point the harness at your model** (env vars,
OpenAI-compatible base URL, etc.). It does not lock you into one vendor.

---

## Three adoption paths

```text
A. Full stack (easiest floor)
   Your model + Anvil harness + Anvil MCP + Anvil train backends

B. Your harness (power users)
   Your orchestrator + Anvil MCP (or raw HTTP) + Anvil prompt pack (optional)

C. Human only
   anvil-web + CLI recipes — same SSOT; no agent required
```

All three must remain valid. **B** is how most serious agent shops will
integrate. **A** is how solo researchers get agentic value without building a
harness. **C** is the trust and debug path.

---

## Method cliffs under agent control (recap)

Different methods fail differently; the agent’s job is to **detect and
switch**, not to pretend one recipe is universal:

| Signal class | Example | Possible act (agent-suggested) |
|--------------|---------|--------------------------------|
| Preference stall | DPO reward proxy flat, probes worse | Switch to GRPO / on-policy; or SFT recovery |
| Advantage collapse | group reward std → 0 | Stop RL; refresh data; lower LR; SFT |
| IS drift | mean_ratio far from 1 | Sync sample adapter; reduce steps between sync |
| Probe regression | fixed probes off-rails | Pause; export; human review |

The **prompt pack** teaches the language of these signals; the **MCP tools**
expose them; the **recipe graph** (product.md) suggests next edges. The
**brain** you bring does the actual judgment.

---

## Implementation order (when we build this)

Does not block Phase 3 vision work; runs as a parallel track when prioritized:

1. Stabilize HTTP SSOT (every UI panel = JSON/SSE)—mostly started.  
2. **MCP server** over that SSOT (no smarts, only tools).  
3. **`prompts/agent/`** pack v0 (generic operator prompts).  
4. **Optional harness** CLI: `anvil agent --model …` (or compose with a
   documented OpenAI-compatible client).  
5. Live **act** tools (pause / switch / sync) with audit.  
6. Recipe-graph doc + tripwire library.

Until (2)–(4) exist, agents can still use raw HTTP + hand-written prompts;
this note is the contract we build toward.

---

## Red lines

- Anvil does not scrape proprietary model UIs or hide vendor lock-in.  
- Agent `force=True` past architecture gates always hits the **audit log**.  
- Prompt pack is **advice**, not an unattended root shell on the lab.  
- No secrets in prompts or example configs.  
- RSI-shaped ambition (product.md) still sits under human governance.

---

## Summary

| Anvil provides | You provide |
|----------------|-------------|
| Train verbs, recipes, data slots, observe | GPUs, data, agent **model** |
| MCP tools + optional harness | API keys / local inference for the brain |
| Generic operator **prompts** (portable) | Policy, judgment overrides, custom harness if desired |

**Own the control plane and the agent *interface*. Leave the frontier brain
to the user—and make good prompting a shipped artifact, not homework.**
