# Anvil

**Anvil forges sovereign domain experts from base models.**

Open-source **LoRA-first post-training** (SFT + RL): tiny four-verb API, honest
recipes, and **live observability**—so you decide *how much is enough* and
*when to shift gears* while data is applied, not only after a failed eval.

Built for **two audiences at once**:

| Mode | Who | How it feels |
|------|-----|----------------|
| **Individual** | Researcher / roboticist / org ML + own GPUs | Recipes, web UI, run a loop, see cliffs on curves and probes |
| **Agentic** | Operator agent (you bring the model) | Same truth via **HTTP + MCP** — watch metrics/probes, pause, patch knobs, switch methods under audit |

You bring **base model, domain data, GPUs, and (optionally) the agent brain**.
Anvil owns the **forge**: control plane, MCP tools, optional harness, portable
prompts, and export. Prompt packs also drop into *your* harness if you do not use ours.

> Originally inspired by the product shape of Thinking Machines’ Tinker API (four low-level train/sample verbs + LoRA). Anvil is independent open-source software; no affiliation, no wire compatibility.

| Doc | Role |
|-----|------|
| **[docs/product.md](docs/product.md)** | Product thesis (domain experts, live sufficiency, dual mode) |
| **[docs/agent-context.md](docs/agent-context.md)** | Operator brief for agents (metrics, tools, classify→act) |
| **[docs/agentic-control.md](docs/agentic-control.md)** | MCP/harness vs user brain; adoption paths A/B/C |
| **[prompts/agent/](prompts/agent/)** | Drop-in operator prompts |
| **[docs/roadmap.md](docs/roadmap.md)** | Phase gates |
| **[start.md](start.md)** | Session entry for humans and coding agents |

## Why

Domain expertise is the product—not a single training method. Different levers
(**SFT**, **DPO**, **GRPO**, …) hit different **cliffs**. Anvil lowers the
systems tax: you keep data, loss, reward, and environment; we keep a stable
contract across **train**, **sample**, and **export**, and make cliffs
**visible and actionable** for a human or an agent under policy.

Core verbs:

```text
forward_backward · optim_step · sample · save_state
```

## Product surface

| Surface | What it is |
|---------|------------|
| **Four-verb client** | Typed train/sample API; `fake://`, `local://` (torch+PEFT), remote HTTP, vLLM sample worker |
| **Web control plane** | `anvil-web` (:7600) — recipes, knobs, runs, gates, observe charts |
| **RL observability** | `metrics.jsonl` (reward, group-std / advantage-collapse, IS ratio, loss) + SSE |
| **Live inference probes** | Fixed prompts from the *current* policy every K steps — eyes before final eval |
| **Sample/train split** | Train local; sample on vLLM with LoRA hot-swap |
| **Agent control** | `AnvilControlClient`, live pause/resume/patch knobs, **`anvil mcp`**, **`anvil agent`** |
| **Vision (Phase 3)** | Media CAS, trajectories, VLM renderer, image modality, `vlm_smoke` |

## Status

**Phase 2 / 2.5 core complete · Phase 3 vision in progress · agent control plane v0 landed.**

Shipped highlights:

- Typed client + `fake://` / `local://` (hand-rolled torch+PEFT verbs, no HF Trainer)
- `HFChatRenderer` / `HFVLMRenderer`; train/sample consistency paths
- HTTP `anvil serve` + `RemoteBackend`; GRPO + IS/PPO; verb queue
- vLLM sample worker + adapter hot-swap
- Metrics, probes, `/observe/{run_id}`; recipe gates + audit
- Vision foundation + image-modality train path; `scripts/vlm_smoke.py`
- Agent: control HTTP client, MCP server (`[mcp]` extra), harness + prompt pack

J-Lens residual readouts (Jacobian lens) are **optional research tooling, not load-bearing**: schema + scoring in `anvil/observe/jlens.py`, `log_jlens` → `jlens.jsonl`, `GET /api/observe/{run_id}/jlens`, forge CLI `scripts/jlens_spike.py` (+ `scripts/run_jlens_7b_mixed.sh` overnight resume). Spike **parked** 2026-07-19 after 1.5B/7B math-order gates failed J2 entry (see `docs/spikes/jlens-math.md`). Mixed-corpus fit needs the `datasets` package on the lab host. The RL debugger ships via metrics, probes, and cliffs without it.

## Quick start

### Human / individual

```bash
pip install -e ".[web,dev]"
anvil-web --host 0.0.0.0 --port 7600
# open http://localhost:7600
# live RL debugger: http://localhost:7600/observe/<run_id>
```

GPU telemetry: [spark-dashboard](https://github.com/niklasfrick/spark-dashboard) on lab hosts (:3000). Anvil owns *training* signals; spark-dashboard owns *hardware* signals.

```python
from anvil import ServiceClient, Datum, ModelInput, AdamParams

svc = ServiceClient()  # fake:// — no GPU
tc = svc.create_lora_training_client(base_model="toy/TinyLM", rank=8)
datum = Datum(
    model_input=ModelInput.from_ints([1, 2, 3, 4]),
    loss_fn_inputs={"target_tokens": [2, 3, 4, 5], "weights": [1, 1, 1, 1]},
)
print(tc.forward_backward([datum], "cross_entropy").result().loss)
tc.optim_step(AdamParams(learning_rate=1e-4)).result()
```

### Agent / MCP

```bash
# terminal A — control plane (same SSOT as the UI)
anvil-web --port 7600

# terminal B — MCP stdio for Cursor / Claude Desktop / custom hosts
pip install -e ".[mcp]"
anvil mcp --url http://127.0.0.1:7600
# or: anvil-mcp --url http://127.0.0.1:7600
```

```bash
# portable prompts only (paste into your harness)
anvil agent --print-prompts

# optional Anvil harness — you bring the brain
export ANVIL_AGENT_API_KEY=…          # or OPENAI_API_KEY
export ANVIL_AGENT_MODEL=gpt-4o-mini  # any OpenAI-compatible model
# optional: ANVIL_AGENT_BASE_URL=https://…/v1
anvil agent "List runs and pause any that are running"
```

Live control HTTP (also used by MCP):

```http
POST /api/runs/{id}/pause
POST /api/runs/{id}/resume
PATCH /api/runs/{id}/knobs   {"knobs": {"learning_rate": 5e-5}}
GET  /api/observe/{id}/metrics?tail=50
GET  /api/audit
```

### RL with metrics + probes

```python
from anvil.recipes.grpo import run_grpo

run_grpo(
    endpoint="local://…",
    sample_endpoint="http://forge:8741",  # empty → Tier 0 in-process
    sync_every=5,
    run_dir="runs/demo",
    probes=[[…token ids…]],
    probe_every=10,
)
```

Watch **advantage collapse** (`group_reward_std_mean → 0`), IS ratio drift, and probe text—the point of negative returns often shows up here first.

### Vision smoke

```bash
PYTHONPATH=. python scripts/vlm_smoke.py --endpoint fake:// --steps 2
# forge: --endpoint local:// --model /mnt/data/models/Qwen2.5-VL-3B-Instruct
```

## Docs

| Doc | When |
|-----|------|
| **[start.md](start.md)** | Session entry (humans + coding agents) |
| **[docs/product.md](docs/product.md)** | Product thesis |
| **[docs/agentic-control.md](docs/agentic-control.md)** | MCP / harness / prompt pack split |
| **[prompts/agent/](prompts/agent/)** | Operator prompts (Anvil or foreign harness) |
| [docs/design.md](docs/design.md) | Architecture |
| [docs/roadmap.md](docs/roadmap.md) | Phases (current: Phase 3 vision + agent v0) |
| [docs/phase3-vision.md](docs/phase3-vision.md) | Vision slices |
| [docs/datasets-robotics.md](docs/datasets-robotics.md) | OXE / Bridge / LeRobot / … |
| [docs/governance.md](docs/governance.md) | Decisions & contributions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | PR hygiene (you merge; agents open PRs) |

## Layout

```text
anvil/
  client/       # ServiceClient / TrainingClient / SamplingClient / futures / remote
  protocol/     # Datum, ModelInput, messages, trajectories, serde
  agent/        # AnvilControlClient, MCP server, optional harness
  control/      # session, adapter registry, gate-override audit
  observe/      # metrics.jsonl + probes.jsonl (+ jlens schema)
  render/       # ToyTextRenderer, HFChatRenderer, HFVLMRenderer
  media/        # content-addressed LocalMediaStore
  data/         # JSONL / path ingest for VLM & robot rows
  backends/     # FakeBackend; LocalBackend (torch+PEFT, image modality)
  workers/      # VLLMSampleBackend
  serve/        # anvil serve — four-verb HTTP
  web/          # anvil-web control plane + /observe
  recipes/      # architecture → pattern → plan; GRPO/SFT/VLM
  losses/       # named loss registry
  export/       # peft / gguf / onnx / trt tags
docs/             # product, design, roadmap, agentic-control, …
prompts/agent/    # portable operator prompt pack
scripts/          # pull_base_model, vlm_smoke, jlens_spike, run_jlens_7b_mixed, …
tests/
```

## Quick principles

1. **Four verbs** before a mega `train()`.
2. **LoRA-first** — small artifacts, hot-swap, export.
3. **Same renderer** for train and sample (critical for RL).
4. **Observe while training** — scalars + live probes catch negative returns and reward hacking.
5. **Dual clients** — human UI and agent MCP/API share one SSOT.
6. **You bring the agent brain** — Anvil wraps tools, harness shape, and prompts.
7. **Vision and edge** are first-class in the data model (not afterthoughts).
8. **Audited force** — architecture gates and method switches leave a trail.

## Keywords

post-training · SFT · RL · GRPO · PPO · DPO · LoRA · PEFT · vLLM · observability · RL debugger · live probes · agent control · MCP · recipe gates · vision-language · robotics · Jetson · edge AI · fine-tuning · LLM training

## License

[Apache-2.0](LICENSE)

## Handoff

Open Grok (or your agent) **in this directory**, say `read start.md`, then one line:

`Outcome: …`
