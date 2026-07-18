# Contributing

Thanks for helping democratize post-training—and for keeping it **operable by humans and agents**.

## Before you code

1. Read [start.md](start.md) (thin card).  
2. Skim [docs/product.md](docs/product.md) if the change touches product surface or agent control.  
3. Check [docs/roadmap.md](docs/roadmap.md) for the active phase.  
4. For API, agent/MCP, or phase-scope changes, open an issue labeled `rfc` first (see [docs/governance.md](docs/governance.md)).

## Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,web]"
pytest
```

Optional: `[mcp]` for `anvil mcp`, `[local]` / `[hf]` for GPU/VLM paths.  
`fake://` powers many unit tests without a GPU.

## Product dual focus (when reviewing PRs)

| Mode | Keep true |
|------|-----------|
| **Individual** | Web UI + CLI remain usable without any agent |
| **Agentic** | New control/observe features expose **JSON/SSE** (and MCP tools when relevant)—not HTML-only |

Agent force-overrides and live acts should leave **audit/log** trails. See [docs/agentic-control.md](docs/agentic-control.md).

## PR hygiene (required)

Process detail: [docs/development-process.md](docs/development-process.md) (same shape as Mia).

- **Branch → PR → Andrew merges.** Never push to `main`; never self-merge.  
- Small, focused PRs.  
- Tests for new/changed behavior; docs when behavior, product thesis, or roadmap exit criteria change.  
- No secrets, private hostnames, or large binaries.  
- Apache-2.0 for all contributions.  

```bash
git checkout -b feat/my-change
# … work, pytest …
git push -u origin HEAD
gh pr create --title "…" --body $'…\n\nCloses #N'
# stop — wait for Andrew to merge
```

## Naming

Call the project **Anvil**. Do not use third-party product names in package names, CLI binaries, or branding.
