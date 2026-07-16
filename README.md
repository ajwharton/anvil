# Anvil

**Open-source, Tinker-shaped post-training** — SFT and RL with a tiny API, LoRA-first adapters, pluggable backends (lab GPU → dual DGX Spark → edge export). Vision is first-class; Jetson/robot export is a first-class path.

> Not affiliated with Thinking Machines Lab. **Anvil** names the open tool; **Tinker** is their hosted API. We copy the *shape* (four verbs, LoRA, train/sample consistency), not their cloud.

## Why

Democratize RL and fine-tuning by lowering the systems tax: you keep **data, loss, reward, environment**; Anvil handles a stable contract across **train**, **sample**, and **export**.

Core verbs (target API):

```text
forward_backward · optim_step · sample · save_state
```

## Status

**Phase 0 stubs.** Typed client contract + in-process `fake://` backend + golden SFT test + web control plane. No real GPU trainer yet. See [docs/roadmap.md](docs/roadmap.md).

### Web UI

Spark-dashboard-inspired dark UI for knobs, runs, loss curves, models, and export:

```bash
pip install -e ".[web]"
anvil-web --host 0.0.0.0 --port 7600
# open http://localhost:7600
```

Links out to forge/hammer [spark-dashboard](https://github.com/niklasfrick/spark-dashboard) (:3000) for GPU telemetry.

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

## Docs (thin start)

| Doc | When |
|-----|------|
| **[start.md](start.md)** | Session entry for agents/humans |
| [docs/design.md](docs/design.md) | Full design (from mia-rl note) |
| [docs/roadmap.md](docs/roadmap.md) | Phases 0–5 |
| [docs/governance.md](docs/governance.md) | How decisions & contributions work |
| [CONTRIBUTING.md](CONTRIBUTING.md) | PR hygiene |

## Layout

```text
anvil/
  client/       # ServiceClient / TrainingClient / SamplingClient / futures
  protocol/     # Datum, ModelInput, messages (vision-ready)
  control/      # session + adapter registry
  render/       # ToyTextRenderer (+ real templates later)
  media/        # content-addressed LocalMediaStore
  backends/     # FakeBackend; local_gpu / dual_spark later
  workers/      # train + sample worker stubs
  losses/       # named loss registry (ce, is, ppo, dpo, …)
  export/       # peft / gguf / onnx / trt format tags
docs/
recipes/        # sl_loop, rl_loop, vlm, robot (later)
tests/          # golden SFT + unit tests
```

## Quick principles

1. **Four verbs** before a mega `train()`.
2. **LoRA-first** — small artifacts, hot-swap, export.
3. **Same renderer** for train and sample (critical for RL).
4. **Vision in the schema** from day one (image refs, not afterthoughts).
5. **Edge export** (Jetson/ONNX/TRT/GGUF) is a product path, not a blog post.

## License

[Apache-2.0](LICENSE)

## Handoff

Open Grok (or your agent) **in this directory**, say `read start.md`, then one line:

`Outcome: …`
