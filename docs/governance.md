# Governance

**Anvil** is an open-source project aimed at democratizing post-training (SFT/RL) via a small, stable API—and at making that surface **operable by humans and by agents** (MCP/API; user-supplied agent model). See [`product.md`](product.md) and [`agentic-control.md`](agentic-control.md).

## Principles

1. **User-owned algorithms** — Anvil provides verbs and backends; recipes and rewards stay in user land.
2. **LoRA-first** — default path is adapters + export, not full-weight FT of frontier MoEs.
3. **Train/sample consistency** — same renderer and adapter identity across train and sample.
4. **Vision and edge are in scope** — not optional forever; roadmap may sequence delivery.
5. **Dual clients** — human UI and agent tools share one SSOT; agents do not scrape HTML.
6. **Audited automation** — force-gates and method switches leave an audit trail; humans keep kill switches.
7. **No trademark confusion** — brand only as Anvil; do not imply affiliation with any commercial post-training product.
8. **Public by default** — discussion and design in the open; secrets never in-repo.

## Roles

| Role | Responsibility |
|------|----------------|
| **Maintainer** | Merge rights, releases, roadmap edits, security triage |
| **Contributor** | PRs, issues, docs, recipes under this governance |
| **User** | Runs Anvil (solo or with their agent); files bugs/RFCs; not required to contribute code |
| **Operator agent** | Uses MCP/HTTP under human policy; not a privileged role in the repo |

Initial maintainer: **Andrew Wharton** (`@ajwharton`). Additional maintainers are added by explicit PR to this file after sustained contribution.

## Decision process

| Type | How |
|------|-----|
| **Typo / docs / small fix** | PR; maintainer merges (no direct `main` push) |
| **Feature within current roadmap phase** | Branch → PR + design note if API surface changes; **Andrew merges** |
| **New phase / non-goal change** | Issue labeled `rfc` → discussion → PR updating `docs/roadmap.md` |
| **License or trademark** | Maintainer decision; no silent relicensing |
| **Security** | Private report preferred; coordinated disclosure |

**RFC issues** should answer: problem, proposal, alternatives, impact on vision/edge, exit criteria.

## PR & merge (Mia-aligned)

Full process: **[development-process.md](development-process.md)**.

- **No direct pushes to `main`.**  
- Agents open PRs; they **do not** merge.  
- **Andrew Wharton** reviews and merges in the GitHub UI.  
- GitHub is the source of truth; local trees are sandboxes.

## Contributions

- By submitting a PR, you agree your contribution is licensed under **Apache-2.0**.
- Prefer the [Developer Certificate of Origin](https://developercertificate.org/) (sign-off optional but appreciated: `Signed-off-by: Name <email>`).
- See [CONTRIBUTING.md](../CONTRIBUTING.md) and [development-process.md](development-process.md) for branch/PR/test hygiene.

## Code of conduct

Be respectful. No harassment, spam, or bad-faith trademark abuse. Maintainers may close issues/PRs that violate this. A fuller CODE_OF_CONDUCT can be added later without blocking early work.

## Relationship to other projects

| Project | Relationship |
|---------|----------------|
| **TRL / OpenRLHF / veRL / PEFT / vLLM** | Likely backends/libraries under the Anvil contract |
| **mia-rl / aiops / starwatch** | Upstream inspiration / personal lab context; not required to use Anvil |

Inspiration one-liner (only place we name the prior art): see `README.md`.

## Releases

- **v0.x** — experimental; breaking API OK with changelog note  
- **v1.0** — stable client verbs + at least one documented local backend  

Version tags: `vMAJOR.MINOR.PATCH` (semver once 0.1+ exists).

## Security

Do not file public issues with private keys, VPN configs, or home LAN inventories. For sensitive reports, contact the maintainer via GitHub private security advisory when enabled.
