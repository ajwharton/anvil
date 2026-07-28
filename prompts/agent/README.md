# Agent prompt pack (v0)

Portable prompts for a **post-training operator** agent that drives Anvil.

Anvil’s product is dual-focus: **individuals** use the web/CLI; **agents** use
HTTP/MCP with these prompts. The **brain** is always yours.

## Load first

| Doc | Role |
|-----|------|
| **[`docs/agent-context.md`](../../docs/agent-context.md)** | **Session brief:** surfaces, MCP tools, metrics by job type, classify→act, hard rules |
| **[`docs/recipes.md`](../../docs/recipes.md)** | Atlas vs personal book vs meta-recipes (product differentiator) |

Then these prompts (system + habits):

| File | Role |
|------|------|
| [`system_operator.md`](system_operator.md) | System role + hard rules + **atlas vs book** |
| [`watch_loop.md`](watch_loop.md) | How to monitor a live run (GRPO / SFT / DPO) |
| [`method_switch.md`](method_switch.md) | Conservative cliff → next-step suggestions |
| [`safety_policy.md`](safety_policy.md) | Force, spend, secrets, stop |

Use them:

1. **With Anvil’s harness** — `anvil agent` (loads this pack) + your model env, or  
2. **MCP only** — `anvil mcp` + Cursor / Claude Desktop / custom host + agent-context + these prompts, or  
3. **In your own harness** — paste or `@`-include and point tools at Anvil HTTP/MCP.

Also: [`docs/agentic-control.md`](../../docs/agentic-control.md) (ownership) · [`docs/product.md`](../../docs/product.md) (why).

Version these with API/tool renames—when tools change, update **agent-context + prompts** in the same PR.
