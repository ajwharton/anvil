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

**Scaffold / design.** No production trainer yet. See [docs/roadmap.md](docs/roadmap.md).

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
anvil/          # Python package (stubs)
  client/       # ServiceClient / TrainingClient / SamplingClient
  protocol/     # wire types
  render/       # text + multimodal
  media/        # content-addressed blobs
  workers/      # train + sample backends
  losses/       # named loss plugins
  export/       # LoRA / merge / edge formats
  backends/     # local_gpu, dual_spark, jetson_edge
docs/           # design, roadmap, governance
recipes/        # sl_loop, rl_loop, vlm, robot offline RL (later)
scripts/        # operator helpers
tests/
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
