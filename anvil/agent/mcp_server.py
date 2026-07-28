"""Anvil MCP server — tools over the control/observe HTTP SSOT.

Run (stdio)::

    pip install -e \".[mcp]\"
    anvil mcp --url http://127.0.0.1:7600

Point Cursor / Claude / custom clients at this process. The **brain** is the
host's model; Anvil only exposes tools. See docs/agentic-control.md.
"""

from __future__ import annotations

import json
import os
from typing import Any

from anvil.agent.client import AnvilControlClient


def _client(url: str | None = None) -> AnvilControlClient:
    return AnvilControlClient(base_url=url or os.environ.get("ANVIL_CONTROL_URL"))


def build_mcp_server(control_url: str | None = None) -> Any:
    """Build a FastMCP server instance (requires ``mcp`` package)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "anvil mcp needs the [mcp] extra: pip install -e \".[mcp]\""
        ) from e

    mcp = FastMCP(
        "anvil",
        instructions=(
            "Anvil post-training control plane. Use tools to plan recipes, "
            "manage runs, watch metrics/probes, pause/patch/resume, and audit. "
            "You are an operator; the user supplies the model that calls you."
        ),
    )
    url = control_url

    def c() -> AnvilControlClient:
        return _client(url)

    @mcp.tool()
    def anvil_health() -> str:
        """Health check for the Anvil control plane (anvil-web)."""
        return json.dumps(c().health())

    @mcp.tool()
    def anvil_overview() -> str:
        """Host overview: runs summary, models, backend label."""
        return json.dumps(c().overview())

    @mcp.tool()
    def anvil_list_recipes(group: str | None = None) -> str:
        """List catalog recipes (optional group filter)."""
        return json.dumps(c().list_recipes(group=group))

    @mcp.tool()
    def anvil_list_recipe_book(
        family: str | None = None, pattern: str | None = None
    ) -> str:
        """List personal recipe book entries (local ANVIL_RECIPE_BOOK store)."""
        return json.dumps(c().list_recipe_book(family=family, pattern=pattern))

    @mcp.tool()
    def anvil_get_recipe_book(recipe_id: str) -> str:
        """Get one personal book recipe by id."""
        return json.dumps(c().get_recipe_book(recipe_id))

    @mcp.tool()
    def anvil_list_meta_recipes() -> str:
        """List meta-recipes (stage graphs) in the personal book."""
        return json.dumps(c().list_meta_recipes())

    @mcp.tool()
    def anvil_get_meta_recipe(meta_id: str) -> str:
        """Get one meta-recipe by id."""
        return json.dumps(c().get_meta_recipe(meta_id))

    @mcp.tool()
    def anvil_suggest(base_model: str, fetch_remote: bool = False) -> str:
        """Suggest catalog + personal book recipes for a base model id."""
        return json.dumps(c().suggest(base_model, fetch_remote=fetch_remote))

    @mcp.tool()
    def anvil_plan(
        base_model: str,
        recipe_id: str | None = None,
        pattern: str | None = None,
        force: bool = False,
    ) -> str:
        """Build a RecipePlan for a model (+ optional recipe_id/pattern)."""
        return json.dumps(
            c().plan(
                base_model,
                recipe_id=recipe_id,
                pattern=pattern,
                force=force,
            )
        )

    @mcp.tool()
    def anvil_list_runs() -> str:
        """List control-plane runs (web store)."""
        return json.dumps(c().list_runs())

    @mcp.tool()
    def anvil_get_run(run_id: str) -> str:
        """Get one run: status, knobs, history, logs."""
        return json.dumps(c().get_run(run_id))

    @mcp.tool()
    def anvil_create_run(
        base_model: str,
        recipe_id: str | None = None,
        name: str | None = None,
        force: bool = False,
        rank: int | None = None,
        learning_rate: float | None = None,
        loss_fn: str | None = None,
    ) -> str:
        """Create a run from knobs (+ optional recipe_id)."""
        knobs: dict[str, Any] = {"base_model": base_model}
        if rank is not None:
            knobs["rank"] = rank
        if learning_rate is not None:
            knobs["learning_rate"] = learning_rate
        if loss_fn is not None:
            knobs["loss_fn"] = loss_fn
        return json.dumps(
            c().create_run(
                name=name,
                knobs=knobs,
                recipe_id=recipe_id,
                force=force,
            )
        )

    @mcp.tool()
    def anvil_train(run_id: str, steps: int = 1) -> str:
        """Train a run for N steps (control-plane fake/local backend)."""
        return json.dumps(c().train(run_id, steps=steps))

    @mcp.tool()
    def anvil_pause(run_id: str) -> str:
        """Pause a run (live control)."""
        return json.dumps(c().pause(run_id))

    @mcp.tool()
    def anvil_resume(run_id: str) -> str:
        """Resume a paused/created run."""
        return json.dumps(c().resume(run_id))

    @mcp.tool()
    def anvil_patch_knobs(run_id: str, knobs_json: str) -> str:
        """Patch knobs mid-run. knobs_json is a JSON object of knob→value."""
        knobs = json.loads(knobs_json)
        return json.dumps(c().patch_knobs(run_id, knobs))

    @mcp.tool()
    def anvil_export(run_id: str, format: str = "peft") -> str:
        """Export adapter for a run."""
        return json.dumps(c().export(run_id, fmt=format))

    @mcp.tool()
    def anvil_sample(run_id: str) -> str:
        """Sample from the run's current adapter (toy/control plane)."""
        return json.dumps(c().sample(run_id))

    @mcp.tool()
    def anvil_observe_list() -> str:
        """List observe run dirs that have metrics.jsonl."""
        return json.dumps(c().list_observe_runs())

    @mcp.tool()
    def anvil_observe_metrics(run_id: str, tail: int = 30) -> str:
        """Tail metrics.jsonl for an observe run_id (RL debugger)."""
        return json.dumps(c().observe_metrics(run_id, tail=tail))

    @mcp.tool()
    def anvil_observe_probes(run_id: str, tail: int = 12) -> str:
        """Tail probes.jsonl for an observe run_id."""
        return json.dumps(c().observe_probes(run_id, tail=tail))

    @mcp.tool()
    def anvil_audit(kind: str | None = None) -> str:
        """Control-plane audit trail (e.g. gate overrides)."""
        return json.dumps(c().list_audit(kind=kind))

    return mcp


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url",
        default=os.environ.get("ANVIL_CONTROL_URL", "http://127.0.0.1:7600"),
        help="anvil-web base URL",
    )
    args = p.parse_args(argv)
    mcp = build_mcp_server(args.url)
    # stdio transport for Cursor / Claude Desktop style hosts
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
