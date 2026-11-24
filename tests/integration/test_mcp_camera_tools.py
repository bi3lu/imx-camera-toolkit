"""Exercise the registered camera tools with the pinned MCP client and stdio."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from imx_camera_toolkit import CameraStats
from imx_camera_toolkit._internal.camera.publishing.jpeg_slot import (
    JPEGPreview,
    JPEGPublication,
)
from imx_camera_toolkit._internal.mcp.camera_adapter import CameraReadAdapter
from imx_camera_toolkit._internal.mcp.contracts import HARD_SNAPSHOT_LIMITS
from imx_camera_toolkit._internal.mcp.host import MCPHost
from imx_camera_toolkit._internal.mcp.snapshot_adapter import CameraSnapshotAdapter

if TYPE_CHECKING:
    from mcp_types import CallToolResult

_ROOT = Path(__file__).resolve().parents[2]
_PHOTO = _ROOT / "examples/mcp_runtime_spike/synthetic.jpg"
_TOOLS = ("camera_status", "camera_capabilities", "camera_snapshot")
_OBSERVER = "mcp:camera:status"
_VISION = "mcp:snapshot:read"
_REVISION = "2026-07-28"
_WIRE_LIMIT = 2048


class _Camera:
    """Supply one synthetic frame and record every adapter source read."""

    def __init__(self, clock: int, read_limit: int) -> None:
        self.reads = 0
        self.stats_reads = 0
        self.photo = _PHOTO.read_bytes()
        self.clock = clock
        self.read_limit = read_limit

    def stats(self) -> CameraStats:
        """Supply cached diagnostics for the running source."""
        self.stats_reads += 1
        return CameraStats(
            1, 0, 30.0, None, 0, 0, True, state="running", has_published_frame=True
        )

    def _jpeg_preview(self) -> JPEGPreview:
        """Supply one fixed publication with deterministic acquisition age."""
        self.reads += 1
        if self.reads > self.read_limit:
            raise RuntimeError("private failure /private/camera")
        publication = JPEGPublication(self.photo, 3, 1, 1, 65, self.clock)
        return JPEGPreview("running", True, True, 1, 1, 65, publication)


async def _serve(profile: str) -> None:
    """Enter a borrowed test host and run it on a disposable stdio transport."""
    from time import monotonic_ns
    from types import SimpleNamespace

    from mcp.server.stdio import stdio_server

    from imx_camera_toolkit._internal.mcp import sdk_server, snapshot_adapter

    # Keep the test JPEG age independent of process and pipe timing.
    clock = monotonic_ns()
    camera = _Camera(clock, read_limit=3 if profile == "wire" else 2)
    read = CameraReadAdapter(camera, backend="cpu")
    snapshot = CameraSnapshotAdapter(
        camera, limits=replace(HARD_SNAPSHOT_LIMITS, max_response_bytes=_WIRE_LIMIT)
    )
    host = MCPHost(read, snapshot)
    grants = {
        "denied": {"admin"},
        "observer": {_OBSERVER},
    }.get(profile, {_OBSERVER, _VISION})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            snapshot_adapter, "time", SimpleNamespace(monotonic_ns=lambda: clock)
        )
        if profile == "rate":
            patch.setattr(
                sdk_server, "time", SimpleNamespace(monotonic_ns=lambda: clock)
            )
        if profile == "default":
            server = sdk_server.create_camera_server(host)
        else:
            server = sdk_server.create_camera_server(
                host, enabled_tools=_TOOLS, grants=grants
            )
        async with host, stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )
    if profile in {"default", "observer", "denied"}:
        assert camera.reads == 0
    if profile == "rate":
        assert camera.stats_reads == 8


def _server_args(profile: str) -> list[str]:
    """Run this test module's synthetic server using the client's interpreter."""
    return [
        "-c",
        "import asyncio, runpy, sys; "
        "asyncio.run(runpy.run_path(sys.argv[1])['_serve'](sys.argv[2]))",
        str(Path(__file__).resolve()),
        profile,
    ]


def _error(result: CallToolResult, code: str) -> dict[str, Any]:
    """Check the public domain error envelope."""
    from mcp_types import TextContent

    assert result.is_error is True
    assert result.structured_content is None
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, TextContent)
    payload: dict[str, Any] = json.loads(block.text)
    assert payload["code"] == code
    assert "/private/camera" not in block.text
    return payload


@pytest.mark.integration
def test_sdk_client_discovers_and_calls_camera_tools() -> None:
    """Verify default, denied, and explicitly granted launches with a real client."""
    pytest.importorskip("mcp")
    from mcp import Client, MCPError, StdioServerParameters
    from mcp_types import INVALID_PARAMS, ImageContent, TextContent

    async def scenario() -> None:
        for profile in ("default", "observer", "denied", "granted"):
            server = StdioServerParameters(
                command=sys.executable, args=_server_args(profile), cwd=_ROOT
            )

            async with Client(server) as client:
                assert client.protocol_version == _REVISION
                tools = await client.list_tools()
                expected = {
                    "default": _TOOLS[:2],
                    "observer": _TOOLS[:2],
                    "denied": (),
                    "granted": _TOOLS,
                }
                assert tuple(tool.name for tool in tools.tools) == expected[profile]
                assert all(
                    tool.description and tool.output_schema for tool in tools.tools
                )

                if profile == "default":
                    with pytest.raises(MCPError) as disabled:
                        await client.call_tool("camera_snapshot")
                    with pytest.raises(MCPError) as unknown:
                        await client.call_tool("camera_unknown")
                    assert disabled.value.error == unknown.value.error
                    assert disabled.value.error.code == INVALID_PARAMS
                    assert (await client.call_tool("camera_status")).structured_content[
                        "state"
                    ] == "ready"

                elif profile in {"observer", "denied"}:
                    denied_tools = _TOOLS if profile == "denied" else _TOOLS[2:]
                    for name in denied_tools:
                        _error(
                            await client.call_tool(name, {"camera_id": "other"}),
                            "permission_denied",
                        )

                else:
                    for arguments in (
                        {"wait_ms": True},
                        {"wait_ms": 1.0},
                        {"camera_id": "other"},
                    ):
                        _error(
                            await client.call_tool("camera_snapshot", arguments),
                            "invalid_argument",
                        )
                    capability = (
                        await client.call_tool("camera_capabilities")
                    ).structured_content
                    assert capability["frame_format"] == "BGR_CPU"
                    assert capability["snapshot"]["source_supported"] is True
                    result = await client.call_tool(
                        "camera_snapshot", {"jpeg_quality": 65}
                    )
                    assert result.is_error is False
                    text, image = result.content
                    assert isinstance(text, TextContent)
                    assert isinstance(image, ImageContent)
                    assert image.mime_type == "image/jpeg"
                    assert (
                        base64.b64decode(image.data, validate=True)
                        == _PHOTO.read_bytes()
                    )
                    assert json.loads(text.text) == result.structured_content
                    assert result.structured_content["frame_sequence"] == 3
                    _error(await client.call_tool("camera_snapshot"), "internal_error")

    asyncio.run(asyncio.wait_for(scenario(), 15))


@pytest.mark.integration
def test_rate_budget_only_admits_valid_calls() -> None:
    """Keep invalid requests free and reject an exhausted local burst."""
    pytest.importorskip("mcp")
    from mcp import Client, StdioServerParameters

    async def scenario() -> None:
        server = StdioServerParameters(
            command=sys.executable, args=_server_args("rate"), cwd=_ROOT
        )
        async with Client(server) as client:
            for _ in range(8):
                _error(
                    await client.call_tool("camera_status", {"camera_id": "other"}),
                    "invalid_argument",
                )
                assert not (await client.call_tool("camera_status")).is_error
            result = await client.call_tool("camera_status")
            payload = _error(result, "rate_limited")
            assert payload["details"]["retry_after_ms"] == 34

    asyncio.run(asyncio.wait_for(scenario(), 15))


@pytest.mark.integration
def test_wire_budget_counts_the_actual_serialized_response() -> None:
    """Accept the exact UTF-8 limit and replace an oversized image with an error."""
    pytest.importorskip("mcp")

    async def scenario() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            *_server_args("wire"),
            cwd=_ROOT,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        stdin, stdout = process.stdin, process.stdout
        assert stdin is not None and stdout is not None

        async def call(request_id: str, arguments: object = None) -> bytes:
            """Return the exact encoded message before newline framing."""
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": "camera_snapshot",
                    "arguments": arguments,
                    "_meta": {
                        "io.modelcontextprotocol/protocolVersion": _REVISION,
                        "io.modelcontextprotocol/clientCapabilities": {},
                    },
                },
            }
            stdin.write(json.dumps(request).encode("utf-8") + b"\n")
            await stdin.drain()
            return (await stdout.readline()).removesuffix(b"\n")

        try:
            invalid = json.loads(await call("null"))["result"]
            assert invalid["isError"] is True
            assert (
                json.loads(invalid["content"][0]["text"])["code"] == "invalid_argument"
            )
            malformed = json.loads(await call("malformed", "invalid protocol object"))
            assert malformed["error"]["code"] == -32602
            initial = await call("a", {})
            assert json.loads(initial)["result"]["isError"] is False
            remaining = _WIRE_LIMIT - len(initial)
            assert remaining > 0
            exact = await call("ą" * (remaining // 2) + "b" * (1 + remaining % 2), {})
            assert len(exact) == _WIRE_LIMIT
            assert json.loads(exact)["result"]["isError"] is False

            oversized = json.loads(await call("z" * (remaining + 2), {}))["result"]
            assert oversized["isError"] is True
            assert "structuredContent" not in oversized
            assert len(oversized["content"]) == 1
            assert (
                json.loads(oversized["content"][0]["text"])["code"]
                == "payload_too_large"
            )
            stdin.close()
            assert await process.wait() == 0
        finally:
            if process.returncode is None:
                process.kill()
                await asyncio.wait_for(process.wait(), 5)

    asyncio.run(asyncio.wait_for(scenario(), 15))
