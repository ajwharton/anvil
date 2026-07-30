"""Agent control plane: HTTP client, MCP tools, optional harness.

See docs/agentic-control.md — Anvil owns tools/harness; you bring the brain.
"""

from anvil.agent.client import AnvilControlClient
from anvil.agent.decide import ActKind, Decision, RunClass, classify_metrics, decide_from_run_dir

__all__ = [
    "ActKind",
    "AnvilControlClient",
    "Decision",
    "RunClass",
    "classify_metrics",
    "decide_from_run_dir",
]
