# start.md — session entry for Anvil

> **How to start a session:** open Grok (or your agent) in this directory, then say
> `read start.md` (or just `start`). Then state your **Outcome** in one line.
>
> Do **not** open other docs unless this file’s pull-on-miss table says so.
> Thin harness: `~/.grok/docs/thin-harness.md`

**Outcome default:** advance the next roadmap phase gate without expanding scope.  
**Track:** Lab / OSS product (not mia-rl Ship coach).

State: `Track: anvil | Outcome: <one line>`

## What this is

**Anvil** = open-source, Tinker-**shaped** post-training toolkit:

- Four verbs: `forward_backward`, `optim_step`, `sample`, `save_state`
- LoRA-first adapters; train/sample consistency; vision + edge (Jetson) path
- Backends: local GPU → dual DGX Spark → export to edge

Not Thinking Machines’ cloud. Not a full MoE train stack on day one.

## Red lines

- **Public repo** — no secrets, keys, host IPs of private LAN, or personal infra dumps in commits.
- Prefer **small PRs**; design changes go through `docs/` before large code.
- Do not trademark-collide: never call this project “Tinker.”
- Do not claim bit-for-bit compatibility with Tinker’s proprietary API.
- Vision and robot/Jetson paths stay **first-class in design**; don’t strip them to ship text-only forever.

## Facts (one screen)

| Item | Value |
|------|--------|
| Repo | public GitHub (this tree) |
| License | Apache-2.0 |
| Design SSOT | `docs/design.md` |
| Roadmap | `docs/roadmap.md` |
| Governance | `docs/governance.md` |
| Package | `anvil/` (Phase 0 typed API + `fake://` backend; GPU in Phase 1) |

## Prefer artifacts

```text
docs/design.md      # architecture & Tinker analysis
docs/roadmap.md     # phases and exit criteria
docs/governance.md  # maintainers, DCO, decision process
README.md           # human front door
```

## Pull-on-miss only

| Path | When |
|------|------|
| `docs/design.md` | API/backend/vision/Jetson design detail |
| `docs/roadmap.md` | Phase scope, exit criteria, next gate |
| `docs/governance.md` | Contribution / maintainer / license dispute |
| `CONTRIBUTING.md` | PR / test expectations |
| `docs/handoff.md` | Spinning a new agent session off this project |
| `docs/models.md` | Reference base models (VLM / Jetson); pull script |
| `anvil/web/` | Control-plane UI (`anvil-web` → :7600); GPU metrics stay on spark-dashboard :3000 |
| `anvil/recipes/` | **Architecture → pattern → plan** (product intelligence; knobs are derived) |

## Do not

- Preload mia-rl Ship scoreboard or coach ontology  
- Commit large weights, datasets, or media blobs  
- Store multi‑GB bases on the Mac — pull to **forge/hammer** (`docs/models.md`, `scripts/pull_base_model.py`)  
- Expand to full distributed MoE train before Phase 1 SFT smoke works  
- Multi-doc “session startup” lists  

## Agent memory

Optional. Artifacts + live checks = SSOT for Anvil.
