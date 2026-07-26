# Anvil product note

**Audience:** humans building with Anvil, and agents that will *run* it.  
**Date:** 2026-07-26 (sovereign domain expert facing)  
**Status:** product thesis (SSOT for “why”)

This note is the short “what is this for?” document. Architecture detail stays in
[`design.md`](design.md); phase gates stay in [`roadmap.md`](roadmap.md).
Operator brief for agents: [`agent-context.md`](agent-context.md).

---

## One sentence

**Anvil forges sovereign domain experts from base models.**

You bring the base, the domain data (and optional rewards), and the GPUs. Anvil
is the forge: place data, run post-training levers (SFT, preference, GRPO, …),
**instrument while data is applied**, decide how much is enough and when to
shift gears, and export an adapter **you own**.

Mechanics today are **LoRA-first** with a tiny four-verb API and live
observability—usable by a single operator, built so it **shines under agent
control**. The substrate may change (post-Transformer included); the product
job does not: **domain expert from base, under sovereign control**.

---

## Primary use case

| You bring | Anvil provides |
|-----------|----------------|
| Base model (small VLM/SLM up through self-hosted large models) | Recipes, gates, train/sample/export contract |
| Domain data / preferences / verifiable rewards | Placement (CAS, JSONL, trajectories), convert paths |
| Compute you control | Local/lab backends, observe artifacts, web + MCP |
| Policy + optional agent brain | Live metrics/probes, early-stop, method switch, audit |

**Profiles** (same forge, different scale):

- **Robotics / edge** — small VLM, real frames, freeze defaults, later Jetson export  
- **Org self-host** — internal corpus → specialist adapters on owned hardware  
- **Solo researcher** — recipes + curves + probes without a platform team  

“Done” is rarely a single step count. It is a **sliding judgment**: probes on
domain tasks, no active cliff, exportable expert, transcript of why you stopped.

---

## The idea most people skip

Default industry habit for post-training:

1. Pick a dataset and a method.  
2. Run the full budget (epochs / steps / mixture).  
3. Evaluate **afterward**.  
4. Discover (too late) that quality peaked early—or that the model got **worse**.

Anvil’s product bet is the opposite:

> **Instrument the run while data is being applied**, so you can decide *how
> much training is enough* and *when to shift gears*—long before the budget
> ends.

That applies to **every** post-training job—text SFT, DPO, GRPO, VLM/robot
LoRA—not only robotics. Robotics is a first-class *path*; live sufficiency is
the *mechanism*; **sovereign domain experts** are the *purpose*.

In a long run there is often a moment where returns go **southward**: loss still
moves, a proxy still climbs, but probes, held-out quality, or group-relative
signal say the policy is getting dumber or more collapsed. **Anvil’s job is to
surface that moment and make smart choices available**—stop, early-stop,
advance the recipe queue, switch method, change LR/rank, swap data mixture—
under human policy, optionally executed by an agent.

Lenses, metrics, and probes applied **during** the pass over data are not
decorative dashboards. They are the control signal for sufficiency.

---

## Two modes of use

| Mode | Who | How Anvil should feel |
|------|-----|------------------------|
| **Individual** | Researcher / roboticist / org ML on own GPUs | Small recipes, honest knobs, web UI, “run this loop and see the curves” |
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
4. **Live signal** — reward, loss, advantage collapse, IS drift, probe
   completions, audit events—so *negative marginal returns* and *southward
   turns* are visible *during* the run.  
5. **Sufficiency & gear-shift** — early-stop, recipe queue advance, method
   switch, and pause/patch so “enough” is a live decision, not a fixed epoch
   count.

**Is not:**

- A single mega-`train()` that hides the policy loop.  
- A fire-and-forget full-dataset pass with evaluation only at the end.  
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
- failure easy to see early (observe + probes + cliffs);  
- “enough” easy to decide mid-run (early-stop, recipe advance, method switch);  
- intervention easy to automate (API/MCP + audited control).

### Live sufficiency (all methods)

| Job type | Live signals (examples) | Smart choices |
|----------|-------------------------|---------------|
| SFT / VLM SFT | loss, held-out probes, n_image_refs | stop, resume, freeze knobs, next data stage |
| GRPO / on-policy | reward, group_std (advantage collapse), IS ratio, probes | early-stop, next recipe stage, temp/LR |
| Preference (DPO, …) | margin proxies, probe quality, collapse | switch to SFT/GRPO, stop, re-mix prefs |
| Robot offline / later RL | trajectory reward, success proxies, probes | stop, next task, export edge |

**Anti-pattern we reject:** train the entire set blindly, then test.  
**Pattern we productize:** lenses/metrics/probes **while** the set is applied;
detect the southward turn; stop or shift gears.

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
It is the direction of a forge that agents can operate at scale under human
governance.

Nearer-term success remains concrete: a researcher, roboticist, or org (or
their agent) can turn a base model into a **domain specialist they own**—on
their GPUs, with live cliffs, method switches, and export—without rewriting
infrastructure.

---

## Agentic control split (summary)

Anvil **owns** the control plane, **MCP tool surface**, optional harness, and
**portable prompt pack**. You **bring the agent model** (frontier or local).
Harness + prompt engineering is hard—so we ship generic operator prompts that
work inside our harness *or* drop into yours.

Detail: [`agentic-control.md`](agentic-control.md) · prompts: [`prompts/agent/`](../prompts/agent/).

## Implications for the roadmap

Prioritize the **Expert ladder** in [`roadmap.md`](roadmap.md) — not open
historical phase checkboxes alone:

| Ladder | Product bar |
|--------|-------------|
| **Expert-v0** | Ship one specialist: place data → train under observe → export |
| **Expert-v1** | Method ladder + cliffs agents can act on |
| **Expert-v2** | Multi-hour / large-corpus / org-scale ops |

Supporting priorities (live sufficiency + agent operability) map into that ladder:

1. **SSOT APIs** — every web panel has a stable JSON/SSE counterpart.  
2. **Observe for every train path** — SFT/VLM done for metrics; probes + DPO → v0/v1.  
3. **Cliff library** — per-method tripwires → **v1**.  
4. **Early-stop + recipe queue** — GRPO done; SFT/VLM + method switch → **v1**.  
5. **Long-job ops** — checkpoint/resume, multi-hour → **v2**.  
6. **MCP + live control** — pause/patch/export; exercise watch→act → **v0/v1**.  
7. **Recipe graph** — “if cliff X, try Y” → **v1**.  
8. **Vision/robot paths** — convert done; lab corpus + probes → **v0**; robot RL/edge → **Path**.

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
