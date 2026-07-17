"""anvil serve — host the four verbs over HTTP on one GPU host.

Thin shell: every route decodes the wire shape, calls the Backend protocol,
re-encodes. No business logic lives here — the backend owns it.

Run::

    pip install -e ".[serve]"   # (+ [local] for the real torch backend)
    anvil serve --backend local --host 0.0.0.0 --port 8741

LAN trust model: no auth in v0 (home cluster). Bind to 127.0.0.1 if the
machine is not on a trusted LAN. See docs/cluster.md.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from anvil.backends.base import Backend
from anvil.protocol.serde import (
    adam_from_wire,
    checkpoint_ref_from_wire,
    datum_from_wire,
    model_input_from_wire,
    sampling_params_from_wire,
    to_wire,
    train_config_from_wire,
)
from anvil.protocol.types import AdapterId

API = "/v1"


def _aid(raw: str | None) -> AdapterId | None:
    return AdapterId(str(raw)) if raw is not None else None


def _error_json(status: int, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


def create_app(backend: Backend, *, token: str | None = None) -> FastAPI:
    """Build the verb-serving app around one backend instance.

    token: optional shared secret; when set, requests must carry
    ``Authorization: Bearer <token>``. v0 cluster trust is LAN-first.
    """
    app = FastAPI(title="anvil-serve", version="0.1.0")

    # Register handlers for *concrete* exception classes. A blanket
    # `Exception` handler is routed by FastAPI to the outermost
    # ServerErrorMiddleware, which re-raises after responding — fine for
    # real bugs (loud 500s in tests), wrong for expected domain errors.
    # Concrete classes land on the inner ExceptionMiddleware, which
    # returns the response without re-raising.
    @app.exception_handler(KeyError)
    async def _not_found(_: Request, exc: KeyError) -> JSONResponse:
        return _error_json(404, exc)

    @app.exception_handler(ValueError)
    async def _bad_request(_: Request, exc: ValueError) -> JSONResponse:
        return _error_json(400, exc)

    @app.exception_handler(FileNotFoundError)
    async def _missing(_: Request, exc: FileNotFoundError) -> JSONResponse:
        return _error_json(400, exc)

    @app.exception_handler(NotImplementedError)
    async def _unimplemented(_: Request, exc: NotImplementedError) -> JSONResponse:
        return _error_json(501, exc)

    @app.middleware("http")
    async def _auth(request: Request, call_next: Callable[..., Any]):
        if token is not None:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {token}":
                return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return await call_next(request)

    @app.get(f"{API}/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "backend": getattr(backend, "name", type(backend).__name__)}

    @app.post(f"{API}/sessions")
    def create_session(body: dict[str, Any]) -> dict[str, Any]:
        config = train_config_from_wire(body["config"])
        adapter_id = backend.create_lora_session(config)
        return {"adapter_id": str(adapter_id)}

    @app.post(f"{API}/sessions/{{adapter_id}}/forward_backward")
    def forward_backward(adapter_id: str, body: dict[str, Any]) -> dict[str, Any]:
        out = backend.forward_backward(
            _aid(adapter_id),  # type: ignore[arg-type]
            [datum_from_wire(d) for d in body["data"]],
            body["loss_fn"],
        )
        return to_wire(out)

    @app.post(f"{API}/sessions/{{adapter_id}}/optim_step")
    def optim_step(adapter_id: str, body: dict[str, Any]) -> dict[str, Any]:
        out = backend.optim_step(_aid(adapter_id), adam_from_wire(body.get("adam") or {}))  # type: ignore[arg-type]
        return to_wire(out)

    @app.post(f"{API}/sessions/{{adapter_id}}/save_state")
    def save_state(adapter_id: str, body: dict[str, Any]) -> dict[str, Any]:
        ref = backend.save_state(_aid(adapter_id), str(body["name"]))  # type: ignore[arg-type]
        return to_wire(ref)

    @app.post(f"{API}/sessions/{{adapter_id}}/load_state")
    def load_state(adapter_id: str, body: dict[str, Any]) -> dict[str, Any]:
        backend.load_state(_aid(adapter_id), checkpoint_ref_from_wire(body["ref"]))  # type: ignore[arg-type]
        return {"ok": True}

    @app.post(f"{API}/sessions/{{adapter_id}}/snapshot_for_sample")
    def snapshot_for_sample(adapter_id: str, body: dict[str, Any]) -> dict[str, Any]:
        ref = backend.snapshot_for_sample(_aid(adapter_id), str(body["name"]))  # type: ignore[arg-type]
        return to_wire(ref)

    @app.post(f"{API}/sessions/{{adapter_id}}/export")
    def export(adapter_id: str, body: dict[str, Any]) -> dict[str, Any]:
        from anvil.protocol.types import ExportFormat

        out = backend.export_adapter(
            _aid(adapter_id),  # type: ignore[arg-type]
            ExportFormat(str(body["format"])),
            str(body["path"]),
        )
        return to_wire(out)

    @app.post(f"{API}/sample")
    def sample(body: dict[str, Any]) -> dict[str, Any]:
        out = backend.sample(
            base_model=str(body["base_model"]),
            adapter_id=_aid(body.get("adapter_id")),
            prompt=model_input_from_wire(body["prompt"]),
            sampling_params=sampling_params_from_wire(body.get("sampling_params") or {}),
            num_samples=int(body.get("num_samples", 1)),
            include_prompt_logprobs=bool(body.get("include_prompt_logprobs", False)),
        )
        return to_wire(out)

    @app.post(f"{API}/compute_logprobs")
    def compute_logprobs(body: dict[str, Any]) -> dict[str, Any]:
        out = backend.compute_logprobs(
            base_model=str(body["base_model"]),
            adapter_id=_aid(body.get("adapter_id")),
            prompt=model_input_from_wire(body["prompt"]),
        )
        return {"logprobs": to_wire(out)}

    return app


def main(argv: list[str] | None = None) -> None:
    """`python -m anvil.serve` entry (argparse lives in anvil.cli)."""
    from anvil.cli import main as cli_main

    cli_main(["serve", *(argv or [])])
