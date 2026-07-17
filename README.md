# Anvil

**Open-source post-training toolkit** — SFT and RL with a tiny four-verb API, LoRA-first adapters, pluggable backends (lab GPU → dual DGX Spark → edge export), and a **live RL debugger** so you can see *when* training stops helping.

> Originally inspired by the product shape of Thinking Machines’ Tinker API (four low-level train/sample verbs + LoRA). Anvil is independent open-source software; no affiliation, no wire compatibility.

## Why

Democratize RL and fine-tuning by lowering the systems tax: you keep **data, loss, reward, environment**; Anvil handles a stable contract across **train**, **sample**, and **export** — and surfaces the signals that usually only show up after a failed eval.

Core verbs:

```text
forward_backward · optim_step · sample · save_state
```

## Product surface

| Surface | What it is |
|---------|------------|
| **Four-verb client** | Typed train/sample API; `fake://`, `local://` (torch+PEFT), remote HTTP, vLLM sample worker |
| **Web control plane** | `anvil-web` (:7600) — recipe knobs, runs, architecture gates, live observe charts |
| **RL observability** | Per-step `metrics.jsonl` (reward, group-std / advantage-collapse tripwire, IS ratio, loss) + SSE live charts |
| **Live inference probes** | Fixed probe set sampled from the *current* policy every K steps during RL — eyes catch reward hacking and the rollover into **negative marginal returns** before final eval |
| **Sample/train split** | Train on LocalBackend; sample on a dedicated vLLM worker with LoRA hot-swap |
| **Roadmap: J-Lens** | Jacobian-lens / latent-space monitor as a debugger panel (spike-gated) — unverbalized reasoning traces as training signal |

## Status

**Phase 2 complete · Phase 2.5 (RL debugger) in progress.**

Shipped:

- Typed client + `fake://` / `local://` (hand-rolled torch+PEFT verbs, no HF Trainer)
- `HFChatRenderer` train/sample prefix consistency
- HTTP transport (`anvil serve` + `RemoteBackend`)
- GRPO loop + IS/PPO losses; async verb queue
- vLLM sample worker with adapter hot-swap (verified mac → forge)
- Metrics scaffolding + Tier-0 live probes + `/observe/{run_id}` UI

Next gate: J-lens spike (last). See [docs/roadmap.md](docs/roadmap.md).

### Web UI & live observe

Spark-dashboard-inspired dark UI for knobs, runs, loss/reward curves, models, export — and a dedicated **observe** page that tails training metrics and probe completions while RL runs:

```bash
pip install -e ".[web]"
anvil-web --host 0.0.0.0 --port 7600
# open http://localhost:7600
# live run debugger: http://localhost:7600/observe/<run_id>
```

Links out to forge/hammer [spark-dashboard](https://github.com/niklasfrick/spark-dashboard) (:3000) for GPU telemetry. Anvil owns *training* signals (reward collapse, IS drift, probe text); spark-dashboard owns *hardware* signals.

### Minimal client example

```python
import anvil
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

### RL with metrics + probes

```python
from anvil.recipes.grpo import run_grpo

# Emits metrics.jsonl + probes.jsonl under run_dir; anvil-web tails them live.
# Tier 1: push LoRA to a vLLM sample worker every sync_every steps.
run_grpo(
    endpoint="local://…",                 # train
    sample_endpoint="http://forge:8741",  # sample worker (empty → Tier 0 in-process)
    sync_every=5,
    run_dir="runs/demo",
    probes=[[…token ids…]],
    probe_every=10,
    # …reward, groups, steps
)
```

In **anvil-web**, open **RL debugger** for `probe_every`, `sync_every`, sample endpoint, and observe links. Watch for **advantage collapse** (`group_reward_std_mean → 0`), IS ratio drift, adapter sync flags, and probe completions that go off-rails — the point of negative returns often shows up here first.

## Docs (thin start)

| Doc | When |
|-----|------|
| **[start.md](start.md)** | Session entry for agents/humans |
| [docs/design.md](docs/design.md) | Full design |
| [docs/roadmap.md](docs/roadmap.md) | Phases 0–5 (current: 2.5 RL debugger) |
| [docs/governance.md](docs/governance.md) | How decisions & contributions work |
| [CONTRIBUTING.md](CONTRIBUTING.md) | PR hygiene |

## Layout

```text
anvil/
  client/       # ServiceClient / TrainingClient / SamplingClient / futures / remote
  protocol/     # Datum, ModelInput, messages (vision-ready), serde
  control/      # session, adapter registry, gate-override audit
  observe/      # RunMetricsWriter — metrics.jsonl + probes.jsonl
  render/       # ToyTextRenderer + HFChatRenderer (real chat templates)
  media/        # content-addressed LocalMediaStore
  backends/     # FakeBackend; LocalBackend (torch+PEFT)
  workers/      # VLLMSampleBackend (sample-only + LoRA hot-swap)
  serve/        # anvil serve — HTTP four-verb transport
  web/          # anvil-web control plane + /observe live debugger
  recipes/      # architecture → pattern → plan; GRPO/SFT helpers
  losses/       # named loss registry (ce, is, ppo, …)
  export/       # peft / gguf / onnx / trt format tags
docs/
recipes/        # sl_loop, rl-facing scripts
tests/
```

## Quick principles

1. **Four verbs** before a mega `train()`.
2. **LoRA-first** — small artifacts, hot-swap, export.
3. **Same renderer** for train and sample (critical for RL).
4. **Observe while training** — scalars + live probes catch negative returns and reward hacking before final eval.
5. **Vision in the schema** from day one (image refs, not afterthoughts).
6. **Edge export** (Jetson/ONNX/TRT/GGUF) is a product path, not a blog post.
7. **Latent monitors (J-Lens)** are debugger views, not the hot path — spike-gated before product panels.

## Keywords

post-training · SFT · RL · GRPO · PPO · LoRA · PEFT · vLLM · observability · RL debugger · live probes · advantage collapse · Jacobian lens · latent space · vision-language · Jetson · edge AI · fine-tuning · LLM training

## License

[Apache-2.0](LICENSE)

## Handoff

Open Grok (or your agent) **in this directory**, say `read start.md`, then one line:

`Outcome: …`
