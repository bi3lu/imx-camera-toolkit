"""Serve the optional MCP SDK over one application's private POSIX pipes."""

from __future__ import annotations

import logging
import os
import stat
import sys
from collections.abc import AsyncIterator, Collection, Iterator
from contextlib import ExitStack, contextmanager
from functools import partial
from threading import Lock

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.dispatcher import coerce_request_id
from mcp.shared.message import ServerMessageMetadata, SessionMessage
from mcp_types import (
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    RequestId,
    jsonrpc_message_adapter,
)

from .host import MCPHost
from .sdk_server import DEFAULT_TOOLS, OBSERVER_SCOPE, create_camera_server

_MAX_REQUEST_BYTES = 64 * 1024
_MAX_ACTIVE_REQUESTS = 8
_WRITE_TIMEOUT_SECONDS = 1.0
_CLAIM_LOCK = Lock()
_LOGGER = logging.getLogger(__name__)


@contextmanager
def _claim_pipes() -> Iterator[tuple[int, int]]:
    """Own the wire until shutdown and permanently divert process stdio.

    Keep diagnostics off the wire through camera teardown and process exit,
    including writes from native libraries and inherited standard descriptors.
    """
    with ExitStack() as stack:
        with _CLAIM_LOCK:
            if os.name != "posix" or not all(
                stat.S_ISFIFO(os.fstat(fd).st_mode) for fd in (0, 1)
            ):
                raise ValueError("MCP stdio requires private POSIX stdin/stdout pipes")

            wire_in = os.dup(0)
            stack.callback(os.close, wire_in)
            wire_out = os.dup(1)
            stack.callback(os.close, wire_out)

            with open(os.devnull, "rb") as null:
                os.dup2(2, 1)
                os.dup2(null.fileno(), 0)

            sys.stdout.flush()
            os.set_blocking(wire_in, False)
            os.set_blocking(wire_out, False)

        yield wire_in, wire_out


async def _request_lines(fd: int) -> AsyncIterator[bytes]:
    """Read newline-framed bytes without waiting on a thread or an unbounded line."""
    pending = b""

    while True:
        await anyio.wait_readable(fd)

        try:
            chunk = os.read(fd, 8192)

        except BlockingIOError:
            continue

        if not chunk:
            return

        *lines, pending = (pending + chunk).split(b"\n")

        for line in lines:
            if len(line) > _MAX_REQUEST_BYTES:
                raise ValueError("MCP stdio request exceeds 64 KiB")

            yield line

        if len(pending) > _MAX_REQUEST_BYTES:
            raise ValueError("MCP stdio request exceeds 64 KiB")


async def _forget_cancelled(active: set[RequestId], request_id: RequestId) -> None:
    """Release admission after the SDK settles a request without a response."""
    active.discard(request_id)


async def _read_requests(
    fd: int,
    sink: MemoryObjectSendStream[SessionMessage | Exception],
    active: set[RequestId],
) -> None:
    """Apply the pipe-wide request budget before handing messages to the SDK."""
    with sink:
        try:
            async for line in _request_lines(fd):
                try:
                    message = jsonrpc_message_adapter.validate_json(line, by_name=False)

                except ValueError:
                    continue

                metadata = None

                if isinstance(message, JSONRPCRequest):
                    request_id = coerce_request_id(message.id)

                    if request_id in active or len(active) >= _MAX_ACTIVE_REQUESTS:
                        raise ValueError(
                            "MCP stdio has a duplicate or excess active request"
                        )

                    active.add(request_id)
                    metadata = ServerMessageMetadata(
                        on_request_unanswered=partial(
                            _forget_cancelled, active, request_id
                        )
                    )

                await sink.send(SessionMessage(message, metadata))

        except ValueError as error:
            _LOGGER.warning("%s; closing the local MCP connection", error)


async def _write_responses(
    fd: int,
    source: MemoryObjectReceiveStream[SessionMessage],
    active: set[RequestId],
    cancel_scope: anyio.CancelScope,
) -> None:
    """Serialize one complete SDK response at a time, with bounded backpressure."""
    with source:
        try:
            async for item in source:
                message = item.message
                line = message.model_dump_json(by_alias=True, exclude_unset=True)
                remaining = memoryview(line.encode("utf-8") + b"\n")

                with anyio.move_on_after(_WRITE_TIMEOUT_SECONDS) as write_scope:
                    while remaining:
                        try:
                            remaining = remaining[os.write(fd, remaining) :]

                        except BlockingIOError:
                            await anyio.wait_writable(fd)

                if write_scope.cancelled_caught:
                    _LOGGER.warning("MCP stdio response write timed out")
                    cancel_scope.cancel()
                    return

                if (
                    isinstance(message, (JSONRPCResponse, JSONRPCError))
                    and message.id is not None
                ):
                    active.discard(coerce_request_id(message.id))

        except BrokenPipeError:
            cancel_scope.cancel()


async def serve_camera_stdio(
    host: MCPHost,
    *,
    enabled_tools: Collection[str] = DEFAULT_TOOLS,
    grants: Collection[str] = frozenset({OBSERVER_SCOPE}),
) -> None:
    """Enter the host and serve one connection until EOF or a failed pipe.

    Use this from a subprocess entry point on POSIX with dedicated stdin
    and stdout pipes. The process permanently redirects ordinary stdin to
    the null device and stdout to stderr. Relaunch in a fresh process.
    A borrowed camera must already be managed by the calling application.
    SDK transport tasks exit before the host drains its reads and releases
    any explicitly owned capture.

    Args:
        host: A configured, unentered host for the application's primary camera.
        enabled_tools: Fixed tool names permitted for this launch.
        grants: Fixed access scopes assigned by the trusted launcher.

    Raises:
        ValueError: If the launch tool set or standard descriptors are invalid.
    """
    server = create_camera_server(host, enabled_tools=enabled_tools, grants=grants)
    options = server.create_initialization_options()

    with _claim_pipes() as (wire_in, wire_out):
        async with host:
            incoming, requests = anyio.create_memory_object_stream[
                SessionMessage | Exception
            ]()
            responses, outgoing = anyio.create_memory_object_stream[SessionMessage]()
            active: set[RequestId] = set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(_read_requests, wire_in, incoming, active)
                tasks.start_soon(
                    _write_responses, wire_out, outgoing, active, tasks.cancel_scope
                )
                await server.run(requests, responses, options)
