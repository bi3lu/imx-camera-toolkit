"""Register local tools on the optional MCP SDK 2.3 server."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Collection
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.resources import files
from math import ceil
from typing import Any

from mcp import MCPError as ProtocolError
from mcp.server import ServerRequestContext
from mcp.server.context import CallNext, HandlerResult
from mcp.server.lowlevel import Server
from mcp_types import (
    INVALID_PARAMS,
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
    ImageContent,
    JSONRPCResponse,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from imx_camera_toolkit import __version__

from .errors import MCPError, map_tool_errors
from .host import MCPHost
from .validation import (
    EMPTY_ARGUMENTS,
    parse_snapshot_input,
    validate_camera_input,
    validate_inference_status_input,
)

OBSERVER_SCOPE = "mcp:camera:status"
SNAPSHOT_SCOPE = "mcp:snapshot:read"
SNAPSHOT_TOOL = "camera_snapshot"
INFERENCE_SCOPE = "mcp:inference:read"
INFERENCE_STATUS_TOOL = "inference_status"
_REQUEST_TIMEOUT_NS = 1_000_000_000
_CALLS_PER_SECOND = 30
_CALL_BURST = 8
_NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True, slots=True)
class _ToolSpec:
    """Bind advertised contract metadata to its required grant."""

    scope: str
    description: str
    input: str
    output: str
    dependencies: tuple[str, ...] = ()


_TOOLS = {
    "camera_status": _ToolSpec(
        OBSERVER_SCOPE,
        "Read the primary camera state and capture diagnostics.",
        "CameraInput",
        "CameraStatus",
    ),
    "camera_capabilities": _ToolSpec(
        OBSERVER_SCOPE,
        "Read cached sensor, capture, control, and JPEG capabilities.",
        "CameraInput",
        "CameraCapabilities",
        (
            "SensorMode",
            "ControlCapabilities",
            "ExposureBounds",
            "GainBounds",
            "SnapshotCapabilities",
        ),
    ),
    SNAPSHOT_TOOL: _ToolSpec(
        SNAPSHOT_SCOPE,
        "Return one fresh JPEG from the primary camera's existing preview source.",
        "SnapshotInput",
        "SnapshotMetadata",
    ),
    INFERENCE_STATUS_TOOL: _ToolSpec(
        INFERENCE_SCOPE,
        "Read cached status for the application's existing inference worker.",
        "InferenceStatusInput",
        "InferenceStatus",
        ("InferenceFrameCount", "InferenceDuration"),
    ),
}
DEFAULT_TOOLS = frozenset({"camera_status", "camera_capabilities"})


def _tool(name: str, spec: _ToolSpec, schemas: dict[str, Any]) -> Tool:
    """Publish standalone schemas for offline client validation."""
    output = deepcopy(schemas[spec.output])
    if spec.dependencies:
        output["$defs"] = {key: deepcopy(schemas[key]) for key in spec.dependencies}
    return Tool(
        name=name,
        description=spec.description,
        input_schema=deepcopy(schemas[spec.input]),
        output_schema=output,
    )


def _tool_error(error: MCPError) -> CallToolResult:
    """Return only the contract's approved public error fields."""
    return CallToolResult(
        is_error=True,
        content=[TextContent(type="text", text=json.dumps(error.to_payload()))],
    )


def create_camera_server(
    host: MCPHost,
    *,
    enabled_tools: Collection[str] = DEFAULT_TOOLS,
    grants: Collection[str] = frozenset({OBSERVER_SCOPE}),
) -> Server[Any]:
    """Create the SDK server for an application-managed host and fixed launch grants.

    Construct the server before opening capture, then run it over the local
    transport inside the entered host's lifespan. Dispatch uses one asyncio
    event loop. The server does not enter or close the host.

    Raises:
        ValueError: If the configured tool set contains an unknown tool name.
    """
    enabled = frozenset(enabled_tools)
    if not enabled <= _TOOLS.keys():
        raise ValueError("unknown configured MCP tool")
    scopes = frozenset(grants)
    schemas = json.loads(
        files(__package__)
        .joinpath("tool-contracts.schema.json")
        .read_text(encoding="utf-8")
    )["$defs"]
    visible = [
        _tool(name, spec, schemas)
        for name, spec in _TOOLS.items()
        if name in enabled and spec.scope in scopes
    ]
    limits = host.snapshot_limits
    credit = float(_CALL_BURST)
    last_admission_ns = time.monotonic_ns()

    async def list_tools(
        _context: ServerRequestContext[Any], params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        """Expose only the enabled tools allowed by the launch grants."""
        if params is not None and params.cursor is not None:
            raise ProtocolError(INVALID_PARAMS, "Invalid cursor")
        return ListToolsResult(tools=visible)

    async def call_tool(
        _context: ServerRequestContext[Any], params: CallToolRequestParams
    ) -> CallToolResult:
        """Authorize and validate before consuming adapter capacity."""
        nonlocal credit, last_admission_ns
        name = params.name
        if name not in enabled:
            raise ProtocolError(INVALID_PARAMS, "Unknown tool")
        arguments = (
            params.arguments
            if "arguments" in params.model_fields_set
            else EMPTY_ARGUMENTS
        )

        try:
            with map_tool_errors():
                if _TOOLS[name].scope not in scopes:
                    raise MCPError("permission_denied")
                if name == SNAPSHOT_TOOL:
                    parse_snapshot_input(arguments, limits)
                elif name == INFERENCE_STATUS_TOOL:
                    validate_inference_status_input(arguments)
                else:
                    validate_camera_input(arguments)

                now = time.monotonic_ns()
                credit = min(
                    _CALL_BURST,
                    credit
                    + (now - last_admission_ns) * _CALLS_PER_SECOND / _NS_PER_SECOND,
                )
                last_admission_ns = now
                if credit < 1:
                    delay_ms = ceil((1 - credit) * 1000 / _CALLS_PER_SECOND)
                    raise MCPError("rate_limited", retry_after_ms=delay_ms)
                credit -= 1

                image = None
                if name == SNAPSHOT_TOOL:
                    deadline_ns = now + _REQUEST_TIMEOUT_NS
                    snapshot = await host.snapshot(arguments, deadline_ns=deadline_ns)
                    size = len(snapshot.jpeg)
                    if (
                        size > limits.max_jpeg_bytes
                        or 4 * ((size + 2) // 3) > limits.max_response_bytes
                    ):
                        raise MCPError("payload_too_large")
                    data = asdict(snapshot.metadata)
                    image = ImageContent(
                        type="image",
                        mime_type="image/jpeg",
                        data=base64.b64encode(snapshot.jpeg).decode("ascii"),
                    )
                elif name == "camera_status":
                    data = asdict(host.status(arguments))
                elif name == INFERENCE_STATUS_TOOL:
                    data = asdict(host.inference_status(arguments))
                else:
                    data = asdict(host.capabilities(arguments))

                content: list[ContentBlock] = [
                    TextContent(type="text", text=json.dumps(data, allow_nan=False))
                ]
                if image is not None:
                    content.append(image)
                    if time.monotonic_ns() >= deadline_ns:
                        raise MCPError("request_timeout")
                return CallToolResult(
                    content=content, structured_content=data, is_error=False
                )
        except MCPError as error:
            return _tool_error(error)

    async def check_snapshot_size(
        ctx: ServerRequestContext[Any], call_next: CallNext
    ) -> HandlerResult:
        """Check the complete successful SDK response, including its actual ID."""
        result = await call_next(ctx)
        if (
            ctx.method == "tools/call"
            and ctx.params is not None
            and ctx.params.get("name") == SNAPSHOT_TOOL
            and ctx.request_id is not None
            and isinstance(result, dict)
            and result.get("isError") is False
        ):
            response = JSONRPCResponse(jsonrpc="2.0", id=ctx.request_id, result=result)
            size = len(
                response.model_dump_json(by_alias=True, exclude_unset=True).encode(
                    "utf-8"
                )
            )
            if size > limits.max_response_bytes:
                failure = _tool_error(MCPError("payload_too_large"))
                # Preserve the SDK's protocol version envelope and server metadata.
                return {
                    key: value
                    for key, value in result.items()
                    if key not in {"content", "structuredContent", "isError"}
                } | failure.model_dump(
                    include={"content", "is_error"}, by_alias=True, mode="json"
                )
        return result

    server: Server[Any] = Server(
        "imx-camera-toolkit",
        version=__version__,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )
    server.middleware.append(check_snapshot_size)
    return server
