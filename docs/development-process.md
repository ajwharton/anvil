# Development process

Aligned with the **Mia** lab rule: GitHub is the source of truth; the maintainer merges.

## Source of truth

Nothing is done until it is **committed, pushed on a PR branch, and merged on GitHub** by the maintainer.

Product dual focus (individuals + agentic control) lives in [`product.md`](product.md) and [`agentic-control.md`](agentic-control.md). Docs PRs that touch public facing text should keep both modes honest.

## No direct pushes to `main`

| Allowed | Not allowed |
|---------|-------------|
| Feature / fix branch → PR → review → **maintainer merge** | `git push origin main` |
| Agent opens PR, leaves it for Andrew | Agent self-merges PRs |
| Squash or merge commit as the maintainer prefers in the UI | Force-push to `main` |

## Workflow (default)

```text
1. Outcome (and optional issue for non-trivial work)
2. git checkout -b feat/…   (or fix/…, docs/…)
3. Implement + tests + docs
4. git push -u origin HEAD
5. gh pr create   # agent does this; does NOT merge
6. Andrew reviews and merges in the GitHub UI
```

## Pull request requirements

A PR is ready for Andrew when it has:

1. **Code** that matches the Outcome / issue  
2. **Tests** for new or changed behavior (no bare “trust me”)  
3. **Docs** when behavior, roadmap exit criteria, or public knobs change  
4. A short PR body: what / why; link issue with `Closes #N` when applicable  

Prefer **small, focused PRs**. Split design changes from bulk implementation when both are large.

## Issues

- **Feature / phase gate / RFC** — open a GitHub issue (or use `rfc` label) before large scope.  
- **Docs-only / tiny fix** — issue optional; PR still required.  
- Tracking-only issues need no PR.

## Review & merge

- **Andrew Wharton** (`@ajwharton`) reviews and **merges** PRs.  
- Agents and other contributors: open the PR, respond to review, **do not merge**.  
- Sole-maintainer exception for emergencies only: still prefer a PR so history stays reviewable.

## Branch naming (suggested)

| Prefix | Use |
|--------|-----|
| `feat/` | New capability |
| `fix/` | Bug fix |
| `docs/` | Docs / governance only |
| `chore/` | CI, deps, tidy |

## Agent rules (copy for sessions)

1. Never push commits to `main`.  
2. Never `gh pr merge` unless the user explicitly says to merge **this** PR.  
3. After opening a PR, stop and report the URL — wait for human merge.  
4. If already on a dirty `main` with local commits: move them to a branch before push.

## Relationship to other docs

| Doc | Role |
|-----|------|
| [governance.md](governance.md) | Principles, roles, license, RFCs |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Dev setup + PR hygiene checklist |
| [start.md](../start.md) | Session entry (red lines include this process) |
