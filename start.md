# start.md — session entry for Anvil

> **How to start a session:** open Grok (or your agent) in this directory, then say
> `read start.md` (or just `start`). Then state your **Outcome** in one line.
>
> Do **not** open other docs unless this file’s pull-on-miss table says so.
> Thin harness: `~/.grok/docs/thin-harness.md`

**Outcome default:** advance the next roadmap phase gate without expanding scope.  
**Track:** Lab / OSS product (not mia-rl Ship coach). **Current phase:** vision (Phase 3).

State: `Track: anvil | Outcome: <one line>`

## What this is

**Anvil** = open-source post-training toolkit (SFT/RL):

- Four verbs: `forward_backward`, `optim_step`, `sample`, `save_state`
- LoRA-first adapters; train/sample consistency; vision + edge (Jetson) path
- Backends: local GPU → dual DGX Spark → export to edge
- **RL debugger**: web UI (`anvil-web`), live metrics, inference probes during RL (negative-return detection)
- **Agent-first control**: same run/metrics/recipe surface via API (and eventually MCP)—usable solo, shines when an agent watches cliffs and switches methods live

Product thesis: [`docs/product.md`](docs/product.md). Not a full MoE train stack on day one. Inspiration one-liner lives in `README.md` only.

## Red lines

- **Public repo** — no secrets, keys, host IPs of private LAN, or personal infra dumps in commits.
- **No direct pushes to `main`.** Branch → PR → **Andrew merges** (Mia-style). Agents never `gh pr merge` unless explicitly told for that PR. See `docs/development-process.md`.
- Prefer **small PRs**; design changes go through `docs/` before large code.
- Do not brand Anvil with third-party product names or claim proprietary API compatibility.
- Vision and robot/Jetson paths stay **first-class in design**; don’t strip them to ship text-only forever.

## Facts (one screen)

| Item | Value |
|------|--------|
| Repo | public GitHub (this tree) |
| License | Apache-2.0 |
| Design SSOT | `docs/design.md` |
| Roadmap | `docs/roadmap.md` |
| Governance | `docs/governance.md` |
| Package | `anvil/` (Phase 1: typed API + `fake://` + `local://` torch/PEFT backend) |

## Prefer artifacts

```text
docs/product.md              # product thesis (human + agentic control)
docs/design.md               # architecture SSOT
docs/roadmap.md              # phases and exit criteria
docs/governance.md           # maintainers, DCO, decision process
docs/development-process.md  # branch → PR → human merge (Mia-aligned)
README.md                    # human front door
```

## Pull-on-miss only

| Path | When |
|------|------|
| `docs/product.md` | Why Anvil; agent control; MCP/API; method cliffs / RSI-shaped ambition |
| `docs/design.md` | API/backend/vision/Jetson design detail |
| `docs/roadmap.md` | Phase scope, exit criteria, next gate |
| `docs/governance.md` | Contribution / maintainer / license dispute |
| `docs/development-process.md` | PR workflow; who merges |
| `CONTRIBUTING.md` | PR / test expectations |
| `docs/handoff.md` | Spinning a new agent session off this project |
| `docs/models.md` | Reference base models (VLM / Jetson); pull script |
| `anvil/web/` | Control-plane UI (`anvil-web` → :7600); GPU metrics stay on spark-dashboard :3000 |
| `anvil/recipes/` | **Architecture → pattern → plan** (product intelligence; knobs are derived) |
| `docs/phase3-vision.md` | Vision phase slices (media → renderer → forge VLM) |
| `docs/datasets-robotics.md` | OXE / Bridge / LeRobot / Robo2VLM → Anvil Examples |

## Do not

- Preload mia-rl Ship scoreboard or coach ontology  
- Commit large weights, datasets, or media blobs  
- Store multi‑GB bases on the Mac — pull to **forge/hammer** (`docs/models.md`, `scripts/pull_base_model.py`)  
- Expand to full distributed MoE train before Phase 1 SFT smoke works  
- Multi-doc “session startup” lists  

## Agent memory

Optional. Artifacts + live checks = SSOT for Anvil.
