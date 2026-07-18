# Agent prompt pack (v0)

Portable prompts for a **post-training operator** agent that drives Anvil.

Use them:

1. **With Anvil’s future harness** (`anvil agent` / MCP loop)—default system load, or  
2. **In your own harness** (Cursor, Claude Code, custom orchestrator)—paste or
   `@`-include these files and point tools at Anvil HTTP/MCP.

The **brain** (which model runs the agent) is always **yours**. See
[`docs/agentic-control.md`](../../docs/agentic-control.md).

| File | Role |
|------|------|
| [`system_operator.md`](system_operator.md) | System role + hard rules |
| [`watch_loop.md`](watch_loop.md) | How to monitor a live run |
| [`method_switch.md`](method_switch.md) | Conservative cliff → next-step suggestions |
| [`safety_policy.md`](safety_policy.md) | Force, spend, secrets, stop |

Version these with API/tool renames—when tools change, update prompts in the
same PR.
