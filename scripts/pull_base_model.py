#!/usr/bin/env python3
"""Pull a base model onto a **lab host** (forge / hammer), never into this laptop repo.

Anvil train/sample will run on lab GPUs. Weights live on NVMe:

  forge/hammer:  /mnt/data/models/<name>     # explicit snapshot (default)
                 /mnt/data/hf_cache          # optional HF hub cache layout

This script SSHes to the host and runs ``huggingface_hub.snapshot_download``
there. The public Anvil tree must not grow multi‑GB weight blobs.

Examples::

  # default: Qwen2.5-VL-3B → forge:/mnt/data/models/Qwen2.5-VL-3B-Instruct
  python scripts/pull_base_model.py

  python scripts/pull_base_model.py --host hammer
  python scripts/pull_base_model.py --preset qwen2.5-vl-7b --host forge
  python scripts/pull_base_model.py --repo Qwen/Qwen2.5-VL-3B-Instruct --local-only  # escape hatch
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def shlex_quote(s: str) -> str:
    return shlex.quote(s)


DEFAULT_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_HOST = "forge"
DEFAULT_MODELS_ROOT = "/mnt/data/models"
DEFAULT_HF_CACHE = "/mnt/data/hf_cache"
# Lab hosts: use the shared venv (PEP 668 blocks system pip).
DEFAULT_REMOTE_PYTHON = "/mnt/data/tools/hf-venv/bin/python"

KNOWN = {
    "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-3b-awq": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-7b-awq": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
}


def _repo_dirname(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def _remote_pull_script(
    *,
    repo: str,
    dest: str,
    revision: str | None,
    use_hf_cache: bool,
    hf_cache: str,
) -> str:
    """Python snippet executed on the lab host."""
    rev = repr(revision)
    if use_hf_cache:
        body = f"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download
os.environ["HF_HOME"] = {hf_cache!r}
Path({hf_cache!r}).mkdir(parents=True, exist_ok=True)
path = snapshot_download(repo_id={repo!r}, revision={rev}, cache_dir=str(Path({hf_cache!r}) / "hub"))
print(path)
"""
    else:
        body = f"""
from pathlib import Path
from huggingface_hub import snapshot_download
dest = Path({dest!r})
dest.mkdir(parents=True, exist_ok=True)
path = snapshot_download(repo_id={repo!r}, revision={rev}, local_dir=str(dest))
print(path)
"""
    return body.strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO})")
    p.add_argument("--preset", choices=sorted(KNOWN), help="Short name instead of --repo")
    p.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"SSH host for lab GPU box (default: {DEFAULT_HOST}). Use 'local' only as escape hatch.",
    )
    p.add_argument(
        "--models-root",
        default=DEFAULT_MODELS_ROOT,
        help=f"Remote directory for named snapshots (default: {DEFAULT_MODELS_ROOT})",
    )
    p.add_argument(
        "--dest-name",
        default=None,
        help="Subdir under models-root (default: last segment of repo id)",
    )
    p.add_argument(
        "--hf-cache",
        action="store_true",
        help=f"Use HF hub cache layout under {DEFAULT_HF_CACHE} instead of a named models/ snapshot",
    )
    p.add_argument("--hf-cache-root", default=DEFAULT_HF_CACHE, help="HF_HOME on the remote host")
    p.add_argument("--revision", default=None, help="Optional git revision / tag")
    p.add_argument(
        "--remote-python",
        default=DEFAULT_REMOTE_PYTHON,
        help=f"Python on lab host with huggingface_hub (default: {DEFAULT_REMOTE_PYTHON})",
    )
    p.add_argument(
        "--local-only",
        action="store_true",
        help="Download on this machine (discouraged; Mac is not the train host)",
    )
    args = p.parse_args(argv)

    repo = KNOWN[args.preset] if args.preset else args.repo
    dest_name = args.dest_name or _repo_dirname(repo)
    dest = f"{args.models_root.rstrip('/')}/{dest_name}"

    if args.local_only or args.host in {"local", "localhost", "127.0.0.1"}:
        print(
            "WARNING: local pull — prefer --host forge|hammer so weights land on lab NVMe.",
            file=sys.stderr,
        )
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("huggingface_hub required: pip install huggingface_hub", file=sys.stderr)
            return 1
        print(f"pulling {repo} locally …", flush=True)
        path = snapshot_download(repo_id=repo, revision=args.revision)
        print(f"ok: {path}")
        return 0

    remote_py = _remote_pull_script(
        repo=repo,
        dest=dest,
        revision=args.revision,
        use_hf_cache=args.hf_cache,
        hf_cache=args.hf_cache_root,
    )
    py = args.remote_python
    remote = f"""
set -euo pipefail
if [ ! -x {shlex_quote(py)} ]; then
  echo "missing remote python: {py}" >&2
  echo "create/share hf-venv on the lab host or pass --remote-python" >&2
  exit 1
fi
{shlex_quote(py)} - <<'PY'
{remote_py}
PY
"""
    print(f"pulling {repo} → {args.host}:{dest if not args.hf_cache else args.hf_cache_root}", flush=True)
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host, "bash", "-s"]
    proc = subprocess.run(cmd, input=remote, text=True)
    if proc.returncode != 0:
        print(f"remote pull failed (exit {proc.returncode})", file=sys.stderr)
        return proc.returncode
    print(f"ok: base_model={repo!r}")
    if not args.hf_cache:
        print(f"    path on {args.host}: {dest}")
    print("    client keeps endpoint/base_model strings only — no weights in the Anvil git tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
