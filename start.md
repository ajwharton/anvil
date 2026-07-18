# start.md — session entry for Anvil

> **How to start a session:** open Grok (or your agent) in this directory, then say
> `read start.md` (or just `start`). Then state your **Outcome** in one line.
>
> Do **not** open other docs unless this file’s pull-on-miss table says so.
> Thin harness: `~/.grok/docs/thin-harness.md`

**Outcome default:** advance the next roadmap phase gate without expanding scope.  
**Track:** Lab / OSS product (not mia-rl Ship coach).  
**Current:** Phase 3 vision + agent control plane v0 (MCP/harness).

State: `Track: anvil | Outcome: <one line>`

## What this is

**Anvil** = open-source **LoRA-first post-training platform** (SFT/RL) with a dual product focus:

| Focus | Meaning |
|-------|---------|
| **Individual** | Recipes, four verbs, web UI, live RL debugger on your GPUs |
| **Agentic** | Same SSOT via HTTP + **MCP**; optional harness; portable prompts; **you bring the brain** |

Mechanics:

- Four verbs: `forward_backward`, `optim_step`, `sample`, `save_state`
- LoRA-first adapters; train/sample consistency; vision + edge (Jetson) path
- Backends: local GPU → dual DGX Spark → export to edge
- **RL debugger**: `anvil-web`, metrics, probes, cliffs (advantage collapse, IS drift)
- **Agent control**: `AnvilControlClient`, `anvil mcp`, `anvil agent`, `prompts/agent/`

Product thesis: [`docs/product.md`](docs/product.md) · agent split: [`docs/agentic-control.md`](docs/agentic-control.md).  
Not a full MoE train stack on day one. Inspiration one-liner lives in `README.md` only.

## Red lines

- **Public repo** — no secrets, keys, host IPs of private LAN, or personal infra dumps in commits.
- **No direct pushes to `main`.** Branch → PR → **Andrew merges** (Mia-style). Agents never `gh pr merge` unless explicitly told for that PR. See `docs/development-process.md`.
- Prefer **small PRs**; design / product thesis changes go through `docs/` before large code.
- Do not brand Anvil with third-party product names or claim proprietary API compatibility.
- Vision and robot/Jetson paths stay **first-class in design**; don’t strip them to ship text-only forever.
- Agent **force** past architecture gates must stay **audited**.

## Facts (one screen)

| Item | Value |
|------|--------|
| Repo | public GitHub (this tree) |
| License | Apache-2.0 |
| Product thesis | `docs/product.md` |
| Agentic control | `docs/agentic-control.md` + `prompts/agent/` |
| Design SSOT | `docs/design.md` |
| Roadmap | `docs/roadmap.md` |
| Governance | `docs/governance.md` |
| Package | `anvil/` — client, backends, web, agent/MCP, observe, recipes, vision |

## Prefer artifacts

```text
docs/product.md              # dual focus: human + agentic
docs/agentic-control.md      # MCP/harness vs user brain
prompts/agent/               # portable operator prompts
docs/design.md               # architecture SSOT
docs/roadmap.md              # phases and exit criteria
docs/governance.md           # maintainers, DCO, decision process
docs/development-process.md  # branch → PR → human merge
README.md                    # public front door
```

## Pull-on-miss only

| Path | When |
|------|------|
| `docs/product.md` | Why Anvil; cliffs; API/MCP; RSI-shaped ambition |
| `docs/agentic-control.md` | Harness/MCP ownership; adoption paths A/B/C |
| `prompts/agent/` | Operator prompts for Anvil or foreign harnesses |
| `anvil/agent/` | Control client, MCP server, harness code |
| `docs/design.md` | API/backend/vision/Jetson design detail |
| `docs/roadmap.md` | Phase scope, exit criteria, next gate |
| `docs/governance.md` | Contribution / maintainer / license dispute |
| `docs/development-process.md` | PR workflow; who merges |
| `CONTRIBUTING.md` | PR / test expectations |
| `docs/handoff.md` | Spinning a new agent session off this project |
| `docs/models.md` | Reference base models (VLM / Jetson); pull script |
| `anvil/web/` | Control-plane UI (`anvil-web` → :7600) |
| `anvil/recipes/` | Architecture → pattern → plan |
| `docs/phase3-vision.md` | Vision phase slices |
| `docs/datasets-robotics.md` | OXE / Bridge / LeRobot / Robo2VLM |

## Do not

- Preload mia-rl Ship scoreboard or coach ontology  
- Commit large weights, datasets, or media blobs  
- Store multi‑GB bases on the Mac — pull to **forge/hammer** (`docs/models.md`, `scripts/pull_base_model.py`)  
- Treat agent control as a second, HTML-scraping path — use HTTP/MCP SSOT  
- Multi-doc “session startup” lists  

## Agent memory

Optional. Artifacts + live checks = SSOT for Anvil.
