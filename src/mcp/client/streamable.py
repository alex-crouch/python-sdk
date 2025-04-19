# ───────────────────────── streamable.py (REPLACEMENT) ─────────────────────────
"""
Transport helper for the “Streamable HTTP” MCP spec (2025‑03‑26).

* Falls back to the older SSE transport automatically.
* Exposes the same `(read_stream, write_stream)` interface that `sse_client()`
  does, so fast‑agent can treat it identically.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import anyio
from anyio import Event
import httpx
from httpx_sse import EventSource
from pydantic import TypeAdapter

import mcp.types as types
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

STREAMABLE_PROTOCOL_VERSION = "2025-03-26"
SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = (
    types.LATEST_PROTOCOL_VERSION,
    STREAMABLE_PROTOCOL_VERSION,
)

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def streamable_client(
    url: str,
    headers: Optional[dict[str, Any]] = None,
    timeout: float = 5,
):
    """
    Create a **memory‑stream transport** that follows the Streamable‑HTTP spec.

    If the target server supports only the legacy SSE transport, we transparently
    fall back to `sse_client()`, so callers don’t need special‑case logic.
    """
    # ░░░░░ 1. Quick probe: is this an old SSE‑only server? ░░░░░
    if await _is_old_sse_server(url, headers=headers, timeout=timeout):
        async with sse_client(url, headers=headers) as (rs, ws):
            yield rs, ws
        return

    # ░░░░░ 2. Build the in‑process queues exposed to the caller ░░░░░
    rs_writer, rs_reader = anyio.create_memory_object_stream[
        types.JSONRPCMessage | Exception
    ](0)
    ws_writer, ws_reader = anyio.create_memory_object_stream

    # Small helper for decoding JSON (single msg or list) and pushing to reader
    async def _push_from_json(payload: str) -> None:
        items = _json_adapter.validate_json(payload)
        if isinstance(items, types.JSONRPCMessage):
            items = [items]
        for item in items:
            await rs_writer.send(item)

    # State shared between writer ↔︎ reader coroutines
    session_id: str | None = None
    reader_ready = Event()

    async with anyio.create_task_group() as tg:
        async with httpx.AsyncClient(timeout=timeout) as client:

            # ── background SSE reader (persistent GET) ────────────────────────
            async def _sse_reader_forever() -> None:
                await reader_ready.wait()  # wait until we have a session‑id
                assert session_id is not None

                sse_headers = {
                    **(headers or {}),
                    "accept": "text/event-stream",
                    "mcp-session-id": session_id,
                }

                while True:  # auto‑reconnect loop
                    try:
                        resp = await client.get(url, headers=sse_headers)
                        resp.raise_for_status()
                        async for sse in EventSource(resp).aiter_sse():
                            if sse.event == "message":
                                await _push_from_json(sse.data)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("SSE stream dropped – retrying in 1 s: %s", exc)
                        await anyio.sleep(1)

            tg.start_soon(_sse_reader_forever)

            # ── writer coroutine (POSTs outbound messages) ────────────────────
            async def _post_writer() -> None:
                nonlocal session_id
                async with ws_reader:
                    async for msg in ws_reader:
                        resp = await client.post(
                            url,
                            json=msg.model_dump(
                                mode="json", by_alias=True, exclude_none=True
                            ),
                            headers={
                                **(headers or {}),
                                "accept": "application/json, text/event-stream",
                                **({"mcp-session-id": session_id} if session_id else {}),
                            },
                        )
                        resp.raise_for_status()

                        # First response carries the session‑id
                        if sid := resp.headers.get("mcp-session-id"):
                            session_id = sid
                            reader_ready.set()

                        # Some servers reply with JSON directly in the POST response
                        ctype = resp.headers.get("content-type", "")
                        if ctype.startswith("application/json"):
                            await _push_from_json(resp.text)
                        elif ctype.startswith("text/event-stream"):
                            # Rare case: server streams back SSE in same reply
                            tg.start_soon(
                                lambda: _relay_inline_sse(resp)  # noqa: E731
                            )

            # Helper for inline SSE in POST response
            async def _relay_inline_sse(response: httpx.Response) -> None:
                async for sse in EventSource(response).aiter_sse():
                    if sse.event == "message":
                        await _push_from_json(sse.data)

            tg.start_soon(_post_writer)

            try:  # ── hand control to caller ──
                yield rs_reader, ws_writer
            finally:
                tg.cancel_scope.cancel()

    # Close our internal writers  (makes .aiter() finish on caller side)
    await rs_writer.aclose()
    await ws_writer.aclose()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_json_adapter: TypeAdapter[
    types.JSONRPCMessage | list[types.JSONRPCMessage]
] = TypeAdapter(types.JSONRPCMessage | list[types.JSONRPCMessage])


async def _is_old_sse_server(
    url: str,
    headers: Optional[dict[str, Any]],
    timeout: float,
) -> bool:
    """
    Probe the endpoint: if it rejects a streamable InitializeRequest (4xx) we
    assume it’s a legacy SSE server so we can fall back gracefully.

    Spec: https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#backwards-compatibility
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        init_req = types.InitializeRequest(
            method="initialize",
            params=types.InitializeRequestParams(
                protocolVersion=STREAMABLE_PROTOCOL_VERSION,
                capabilities=types.ClientCapabilities(),
                clientInfo=types.Implementation(name="mcp", version="0.1.0"),
            ),
        )
        resp = await client.post(
            url,
            json=types.JSONRPCRequest(
                jsonrpc="2.0",
                id=1,
                method=init_req.method,
                params=init_req.params.model_dump(
                    by_alias=True, mode="json", exclude_none=True
                ),
            ).model_dump(by_alias=True, mode="json", exclude_none=True),
            headers={
                **(headers or {}),
                "accept": "application/json, text/event-stream",
            },
        )
        return 400 <= resp.status_code < 500