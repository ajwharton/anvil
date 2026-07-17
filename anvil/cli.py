"""anvil CLI — `anvil serve` hosts the four verbs on one host.

Subcommands are deliberately few; recipes and planning live in the library and
the web control plane, not here.
"""

from __future__ import annotations

import argparse
import os


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
        help="base model id/path for --backend vllm-sample (e.g. Qwen/Qwen2.5-1.5B-Instruct)",
    )
    serve.add_argument("--root", default=None, help="state root for checkpoints/adapters")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (LAN: 0.0.0.0)")
    serve.add_argument("--port", type=int, default=8741)
    serve.add_argument(
        "--token",
        default=os.environ.get("ANVIL_TOKEN"),
        help="shared secret clients must send as Bearer (default: $ANVIL_TOKEN, else none)",
    )

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


if __name__ == "__main__":
    main()
