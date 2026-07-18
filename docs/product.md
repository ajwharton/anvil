# Anvil product note

**Audience:** humans building with Anvil, and agents that will *run* it.  
**Date:** 2026-07-17  
**Status:** product thesis (independent of Phase 3 completion)

This note is the short “what is this for?” document. Architecture detail stays in
[`design.md`](design.md); phase gates stay in [`roadmap.md`](roadmap.md).

---

## One sentence

Anvil is a **LoRA-first post-training platform** (SFT + RL) with a tiny stable
API, good default recipes, and **live observability**—usable by a single
operator, but built so it **shines under agent control**.

---

## Two modes of use

| Mode | Who | How Anvil should feel |
|------|-----|------------------------|
| **Individual** | Researcher / roboticist on a laptop + lab GPUs | Small recipes, honest knobs, web UI, “run this loop and see the curves” |
| **Agentic** | An autonomous (or semi-autonomous) training agent | **Everything is machine-readable**: runs, metrics, probes, recipes, gates, audits—via **HTTP API and/or MCP**—so the agent can *watch*, *decide*, and *act* without a human staring at plots |

Individuals must never be second-class. The same surfaces that power the web UI
should power agents. Agents are not a bolt-on chat wrapper; they are a
**first-class client** of the control and observe planes.

---

## What the platform is (and is not)

**Is:**

1. **Mechanisms** — four verbs (`forward_backward`, `optim_step`, `sample`,
   `save_state`), named losses, LoRA adapters, train/sample consistency.  
2. **Recipes** — architecture-aware starting points (SFT, GRPO/IS/PPO, VLM,
   freeze policies) with explicit gates (recommended / stretch / blocked).  
3. **Places for data** — multimodal examples, trajectories, media refs, run
   artifacts (`metrics.jsonl`, `probes.jsonl`, exports)—not black-box “upload
   and hope.”  
4. **Live signal** — reward, advantage collapse, IS drift, probe completions,
   audit events—so *negative marginal returns* are visible *during* the run.

**Is not:**

- A single mega-`train()` that hides the policy loop.  
- A proprietary hosted cloud (you bring GPUs).  
- A claim that any one method (DPO, GRPO, SFT, …) always wins.

---

## Why agent control matters

RL and preference methods do not fail politely. Quality of **data** and
**method** interact:

- DPO can climb a reward-looking proxy and then **cliff** (collapse, mode
  collapse, preference hacking).  
- GRPO / on-policy RL can keep learning where offline preference plateaus—or
  the reverse.  
- LoRA SFT may be the right recovery when policy gradients go unstable.  
- The “right” switch depends on **live** signals, not a fixed schedule written
  before the run.

**Expectation:** an agent (or multi-agent system) **monitors** the run in real
time and **alters** it in real time—recipe, loss family, LR, rank, sample
backend, probe set, stop/export—under human policy constraints.

That is only possible if:

| Requirement | Product implication |
|-------------|---------------------|
| All run state is queryable | HTTP + optional **MCP** tools over the same SSOT (runs, metrics, probes, plans, audits, adapters) |
| Streams, not only dumps | SSE / tail APIs (already sketched for metrics); agents subscribe |
| Controls are explicit | Named verbs + recipe plans + gate overrides (audited)—no hidden Trainer state |
| Cliffs are first-class | Tripwires (e.g. advantage collapse) and probe text, not only scalar reward |
| Method switches are cheap | Same client surface for SFT ↔ GRPO ↔ preference; adapters are small LoRA artifacts |

The web UI is the human viewport. **MCP/API is the agent viewport.** Both must
see the same truth.

---

## Agent-facing surface (target shape)

Agents should be able to, without scraping HTML:

1. **Discover** — list recipes, gates for a base model, open runs.  
2. **Start** — create a run from a plan (or force past a gate with audit).  
3. **Watch** — stream metrics, probes, collapse flags, sample outputs.  
4. **Decide** — e.g. “DPO mean_reward flat + probe quality down for K steps.”  
5. **Act** — pause; change knobs; switch recipe/loss; push adapter to sample
   worker; export; start a follow-on run.  
6. **Account** — read audit log for overrides and method switches.

**Today (partial):** HTTP serve for verbs, web API for plans/runs/observe,
metrics JSONL + SSE, gate-override audit.  
**Gap:** unified agent tool surface (MCP), live **control** during a run
(not only post-hoc config), explicit **method-switch** recipes, richer cliff
detectors per family (DPO vs GRPO vs SFT).

---

## Data as the bottleneck (and the opportunity)

Post-training quality is dominated by **data** and **when you stop or switch**.
Anvil’s job is to make:

- data easy to place (examples, trajectories, media CAS, JSONL ingest);  
- methods easy to start (recipes + gates);  
- failure easy to see early (observe + probes);  
- intervention easy to automate (API/MCP + audited control).

At significant scale—many runs, many bases, continuous evaluation—the only
tractable operator is an **agent loop**. Humans set policy and red lines;
agents execute watch/decide/act.

---

## North-star ambition (explicitly lofty)

Done well and run at scale, a platform of this shape is a candidate substrate
for **recursive self-improvement (RSI)**-style loops: models (or agents using
models) that improve training procedures by observing outcomes and changing
recipes, data mixtures, and stop conditions—always under human governance.

That is **not** a near-term phase gate and **not** a claim Anvil “is RSI.”
It is the direction of the product: **instrumented, controllable, LoRA-first
post-training that agents can operate**.

Nearer-term success remains concrete: a researcher or roboticist (or their
agent) can SFT/RL a small LLM/VLM on own GPUs, see cliffs as they form, and
switch methods without rewriting infrastructure.

---

## Agentic control split (summary)

Anvil **owns** the control plane, **MCP tool surface**, optional harness, and
**portable prompt pack**. You **bring the agent model** (frontier or local).
Harness + prompt engineering is hard—so we ship generic operator prompts that
work inside our harness *or* drop into yours.

Detail: [`agentic-control.md`](agentic-control.md) · prompts: [`prompts/agent/`](../prompts/agent/).

## Implications for the roadmap (guidance, not a new phase list)

Prioritize work that increases **agent operability** even while Phase 3 vision
and edge land:

1. **SSOT APIs** — every web panel has a stable JSON/SSE counterpart.  
2. **MCP server** — thin tools over those APIs (list/run/tail/act/audit); Anvil-owned façade.  
3. **Optional harness** — model-agnostic loop; user plugs brain + keys.  
4. **Prompt pack** — versioned generic operator prompts (portable to foreign harnesses).  
5. **Live control** — pause, patch knobs, hot-swap sample adapter, method
   switch without process death where possible.  
6. **Cliff library** — per-method tripwires + probe policies (DPO, GRPO, SFT).  
7. **Recipe graph** — documented “if cliff X, try recipe Y” edges agents can
   follow (human-editable, not hardcoded magic).  
8. **Vision/robot data paths** — keep going; agents need the same observe loop
   on VLM/robot runs.

Human UI remains essential for trust and debugging. It should be a client of
the same control plane, not a separate parallel system.

---

## Red lines (product)

- Public repo, no secrets or private LAN dumps.  
- No trademark collision with proprietary products.  
- Agent force-overrides of architecture gates stay **audited**.  
- Humans can always stop the loop; RSI-shaped ambition does not remove
  governance.  
- Prefer small, reviewable changes (see [`development-process.md`](development-process.md)).

---

## Related docs

| Doc | Role |
|-----|------|
| [`design.md`](design.md) | Architecture, verbs, backends |
| [`roadmap.md`](roadmap.md) | Phase exit criteria |
| [`agentic-control.md`](agentic-control.md) | MCP harness vs user brain; prompt pack |
| [`phase3-vision.md`](phase3-vision.md) | Vision slice plan |
| [`datasets-robotics.md`](datasets-robotics.md) | Robotics / VLM data sources |
| [`../prompts/agent/`](../prompts/agent/) | Drop-in operator prompts |
| [`start.md`](../start.md) | Session entry for humans and agents |
