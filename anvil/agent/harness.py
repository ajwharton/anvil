"""Optional Anvil agent harness — you bring the brain model.

Loads ``prompts/agent/*`` and runs a simple OpenAI-compatible tool loop against
Anvil MCP-equivalent actions via :class:`AnvilControlClient`.

::

    export ANVIL_CONTROL_URL=http://127.0.0.1:7600
    export ANVIL_AGENT_BASE_URL=https://api.openai.com/v1   # or local vLLM
    export ANVIL_AGENT_API_KEY=...
    export ANVIL_AGENT_MODEL=gpt-4o-mini
    anvil agent --once "List runs and summarize status"

Without API keys, ``anvil agent --print-prompts`` dumps the prompt pack for
use in an external harness.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from anvil.agent.client import AnvilControlClient

_PROMPT_FILES = (
    "system_operator.md",
    "watch_loop.md",
    "method_switch.md",
    "safety_policy.md",
)


def prompt_pack_dir() -> Path:
    # repo: prompts/agent ; installed: next to package via package-data fallback
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "prompts" / "agent",
        Path.cwd() / "prompts" / "agent",
        Path(os.environ.get("ANVIL_PROMPTS", "")) / "agent",
    ]
    for c in candidates:
        if c.is_dir() and (c / "system_operator.md").is_file():
            return c
    raise FileNotFoundError(
        "prompts/agent not found; run from repo root or set ANVIL_PROMPTS"
    )


def load_prompt_pack() -> str:
    d = prompt_pack_dir()
    parts: list[str] = []
    for name in _PROMPT_FILES:
        p = d / name
        if p.is_file():
            parts.append(f"# --- {name} ---\n\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def tool_specs() -> list[dict[str, Any]]:
    """OpenAI-style tool definitions mapping to AnvilControlClient methods."""
    return [
        {
            "type": "function",
            "function": {
                "name": "anvil_overview",
                "description": "Host overview: runs and models",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "anvil_list_runs",
                "description": "List control-plane runs",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "anvil_get_run",
                "description": "Get one run by id",
                "parameters": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "anvil_suggest",
                "description": "Suggest recipes for a base model",
                "parameters": {
                    "type": "object",
                    "properties": {"base_model": {"type": "string"}},
                    "required": ["base_model"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "anvil_train",
                "description": "Train a run for N steps",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "steps": {"type": "integer", "default": 1},
                    },
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "anvil_pause",
                "description": "Pause a run",
                "parameters": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "anvil_resume",
                "description": "Resume a run",
                "parameters": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "anvil_patch_knobs",
                "description": "Patch knobs JSON on a run",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "knobs": {"type": "object"},
                    },
                    "required": ["run_id", "knobs"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "anvil_observe_metrics",
                "description": "Tail RL metrics for observe run_id",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "tail": {"type": "integer", "default": 20},
                    },
                    "required": ["run_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "anvil_audit",
                "description": "List audit events",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def dispatch_tool(client: AnvilControlClient, name: str, args: dict[str, Any]) -> str:
    table: dict[str, Callable[..., Any]] = {
        "anvil_overview": lambda: client.overview(),
        "anvil_list_runs": lambda: client.list_runs(),
        "anvil_get_run": lambda: client.get_run(args["run_id"]),
        "anvil_suggest": lambda: client.suggest(args["base_model"]),
        "anvil_train": lambda: client.train(args["run_id"], int(args.get("steps", 1))),
        "anvil_pause": lambda: client.pause(args["run_id"]),
        "anvil_resume": lambda: client.resume(args["run_id"]),
        "anvil_patch_knobs": lambda: client.patch_knobs(args["run_id"], args["knobs"]),
        "anvil_observe_metrics": lambda: client.observe_metrics(
            args["run_id"], tail=int(args.get("tail", 20))
        ),
        "anvil_audit": lambda: client.list_audit(),
    }
    if name not in table:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        return json.dumps(table[name]())
    except Exception as e:
        return json.dumps({"error": str(e)})


def _chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout: float = 120.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"agent model HTTP {e.code}: {e.read().decode()}") from e


def run_harness_once(
    user_message: str,
    *,
    control_url: str | None = None,
    max_rounds: int = 8,
) -> str:
    """One-shot agent turn with tools. Returns final assistant text."""
    base_url = os.environ.get("ANVIL_AGENT_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("ANVIL_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("ANVIL_AGENT_MODEL", "gpt-4o-mini")
    if not api_key:
        raise SystemExit(
            "Set ANVIL_AGENT_API_KEY (or OPENAI_API_KEY) and optionally "
            "ANVIL_AGENT_BASE_URL / ANVIL_AGENT_MODEL — or use --print-prompts"
        )

    client = AnvilControlClient(base_url=control_url)
    system = load_prompt_pack()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    tools = tool_specs()
    final = ""
    for _ in range(max_rounds):
        resp = _chat_completions(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            tools=tools,
        )
        choice = resp["choices"][0]["message"]
        messages.append(choice)
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            final = choice.get("content") or ""
            break
        for tc in tool_calls:
            fn = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            result = dispatch_tool(client, fn, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )
    return final
