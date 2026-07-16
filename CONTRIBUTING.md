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

## PR hygiene

- Small, focused PRs.  
- Update docs when behavior or roadmap exit criteria change.  
- No secrets, private hostnames, or large binaries.  
- Apache-2.0 for all contributions.  

## Naming

Call the project **Anvil**. Do not use “Tinker” in package names, CLI binary names, or trademarks. “Tinker-shaped” is OK as descriptive English in docs.
