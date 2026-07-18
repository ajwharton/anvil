"""Agent control plane: HTTP client, MCP tools, optional harness.

See docs/agentic-control.md — Anvil owns tools/harness; you bring the brain.
"""

from anvil.agent.client import AnvilControlClient

__all__ = ["AnvilControlClient"]
