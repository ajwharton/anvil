"""python -m anvil.web"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Anvil web control plane")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7600)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "web extras required: pip install -e '.[web]'"
        ) from e

    uvicorn.run(
        "anvil.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
