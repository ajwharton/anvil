# Contributing

Thanks for helping democratize post-training.

## Before you code

1. Read [start.md](start.md) (thin card).  
2. Check [docs/roadmap.md](docs/roadmap.md) for the active phase.  
3. For API or phase-scope changes, open an issue labeled `rfc` first (see [docs/governance.md](docs/governance.md)).

## Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Phase 0 needs no GPU. The `fake://` backend powers golden tests in `tests/`.

## PR hygiene (required)

Process detail: [docs/development-process.md](docs/development-process.md) (same shape as Mia).

- **Branch → PR → Andrew merges.** Never push to `main`; never self-merge.  
- Small, focused PRs.  
- Tests for new/changed behavior; docs when behavior or roadmap exit criteria change.  
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
