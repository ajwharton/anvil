"""anvil CLI — serve verbs, MCP tools, optional agent harness.

Subcommands stay few; recipes and planning live in the library and web UI.
"""

from __future__ import annotations

import argparse
import os
import sys


def _build_backend(name: str, root: str | None, model: str | None = None):
    if name == "fake":
        from anvil.backends.fake import FakeBackend

        return FakeBackend(root=root)
    if name == "local":
        from anvil.backends.local import LocalBackend

        return LocalBackend(root=root)
    if name == "vllm-sample":
        if not model:
            raise SystemExit("--model is required for --backend vllm-sample")
        from anvil.workers.sample import VLLMSampleBackend

        return VLLMSampleBackend(model=model, root=root)
    raise SystemExit(f"unknown backend {name!r}; choose 'local', 'fake', or 'vllm-sample'")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="anvil", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="host the four verbs over HTTP")
    serve.add_argument(
        "--backend",
        default="local",
        choices=("local", "fake", "vllm-sample"),
        help=(
            "'local' = torch+PEFT (needs the [local] extra); 'fake' = golden-test "
            "stub; 'vllm-sample' = sampling-only worker over a vLLM engine (needs "
            "vllm + --model)"
        ),
    )
    serve.add_argument(
        "--model",
        default=None,
        help="base model id/path for --backend vllm-sample",
    )
    serve.add_argument("--root", default=None, help="state root for checkpoints/adapters")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (LAN: 0.0.0.0)")
    serve.add_argument("--port", type=int, default=8741)
    serve.add_argument(
        "--token",
        default=os.environ.get("ANVIL_TOKEN"),
        help="shared secret clients must send as Bearer (default: $ANVIL_TOKEN)",
    )

    mcp_p = sub.add_parser(
        "mcp",
        help="stdio MCP server over anvil-web control URL (needs [mcp] extra)",
    )
    mcp_p.add_argument(
        "--url",
        default=os.environ.get("ANVIL_CONTROL_URL", "http://127.0.0.1:7600"),
        help="anvil-web base URL (default $ANVIL_CONTROL_URL or :7600)",
    )

    agent_p = sub.add_parser(
        "agent",
        help="optional agent harness (you bring the model via env)",
    )
    agent_p.add_argument(
        "message",
        nargs="?",
        default=None,
        help="user message for one tool-using turn",
    )
    agent_p.add_argument(
        "--print-prompts",
        action="store_true",
        help="print portable prompt pack and exit (no API key needed)",
    )
    agent_p.add_argument(
        "--url",
        default=os.environ.get("ANVIL_CONTROL_URL", "http://127.0.0.1:7600"),
        help="anvil-web base URL",
    )
    agent_p.add_argument("--max-rounds", type=int, default=8)

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        try:
            import uvicorn
        except ImportError as e:
            raise SystemExit(
                "anvil serve needs the [serve] extra: pip install -e \".[serve]\""
            ) from e
        from anvil.serve.app import create_app

        backend = _build_backend(args.backend, args.root, args.model)
        app = create_app(backend, token=args.token)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if args.cmd == "mcp":
        from anvil.agent.mcp_server import main as mcp_main

        mcp_main(["--url", args.url])
        return

    if args.cmd == "agent":
        from anvil.agent.harness import load_prompt_pack, run_harness_once

        if args.print_prompts:
            sys.stdout.write(load_prompt_pack() + "\n")
            return
        if not args.message:
            raise SystemExit("pass a message or --print-prompts")
        text = run_harness_once(
            args.message, control_url=args.url, max_rounds=args.max_rounds
        )
        sys.stdout.write((text or "") + "\n")
        return


if __name__ == "__main__":
    main()
