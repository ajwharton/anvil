# Agent prompt pack (v0)

Portable prompts for a **post-training operator** agent that drives Anvil.

Anvil’s product is dual-focus: **individuals** use the web/CLI; **agents** use
HTTP/MCP with these prompts. The **brain** is always yours.

Use them:

1. **With Anvil’s harness** — `anvil agent` (loads this pack) + your model env, or  
2. **MCP only** — `anvil mcp` + Cursor / Claude Desktop / custom host + these prompts as system context, or  
3. **In your own harness** — paste or `@`-include and point tools at Anvil HTTP/MCP.

See [`docs/agentic-control.md`](../../docs/agentic-control.md) and [`docs/product.md`](../../docs/product.md).

| File | Role |
|------|------|
| [`system_operator.md`](system_operator.md) | System role + hard rules |
| [`watch_loop.md`](watch_loop.md) | How to monitor a live run |
| [`method_switch.md`](method_switch.md) | Conservative cliff → next-step suggestions |
| [`safety_policy.md`](safety_policy.md) | Force, spend, secrets, stop |

Version these with API/tool renames—when tools change, update prompts in the
same PR.
