"""Verify the local MCP release profile against synthetic contract fixtures."""

import argparse
import asyncio
import base64
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from anyio import CancelScope, Event, fail_after, sleep_forever
from mcp import Client, MCPError, StdioServerParameters
from mcp.server import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp_types import (
    INVALID_PARAMS,
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    Tool,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "docs" / "mcp"
SCHEMA = json.loads(
    (ROOT / "imx_camera_toolkit/_internal/mcp/tool-contracts.schema.json").read_text()
)
EXAMPLES = json.loads((CONTRACTS / "tool-contracts.examples.json").read_text())
CASES = {case["name"]: case for case in EXAMPLES["cases"]}
NAME = "imx-mcp-profile-probe"
REVISION = "2026-07-28"
UNSUPPORTED = "2099-01-01"
DEADLINE = 10


def tool_schema(name: str) -> dict[str, Any]:
    """Expand the named schema with the shared definitions for SDK validation."""
    return {"$defs": SCHEMA["$defs"], **SCHEMA["$defs"][name]}


async def serve() -> None:
    """Serve fixed fixtures and a cooperative wait over private stdio pipes."""
    waiting = Event()
    finished = Event()
    defaults = {
        "camera_status": "status_ready",
        "camera_capabilities": "capabilities_known",
        "camera_snapshot": "snapshot_fresh",
    }

    async def list_tools(
        _context: object, _params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        if waiting.is_set():
            await finished.wait()

        return ListToolsResult(
            tools=[
                Tool(
                    name=name,
                    input_schema=tool_schema(spec["input"]),
                    output_schema=tool_schema(spec["success"]),
                )
                for name, spec in sorted(EXAMPLES["tools"].items())
            ]
        )

    async def call_tool(
        context: ServerRequestContext[object], params: CallToolRequestParams
    ) -> CallToolResult:
        if params.name not in defaults:
            raise MCPError(INVALID_PARAMS, "Unknown tool")

        arguments = params.arguments or {}
        case = defaults[params.name]

        if params.name == "camera_snapshot":
            if arguments == CASES["snapshot_stale"]["arguments"]:
                case = "snapshot_stale"

            if arguments.get("wait_ms") == 1000:
                waiting.set()

                try:
                    await context.session.report_progress(0)
                    await sleep_forever()

                finally:
                    finished.set()

        expected = CASES[case]["expected"]
        data = expected["value"]
        blocks = [{"type": "text", "text": json.dumps(data)}]

        if "jpeg" in expected:
            jpeg = (CONTRACTS / expected["jpeg"]).read_bytes()
            blocks.insert(
                0,
                {
                    "type": "image",
                    "mimeType": "image/jpeg",
                    "data": base64.b64encode(jpeg).decode("ascii"),
                },
            )

        result: dict[str, Any] = {
            "content": blocks,
            "isError": expected["kind"] == "domain_error",
        }

        if not result["isError"]:
            result["structuredContent"] = data

        return CallToolResult.model_validate(result)

    app = Server(NAME, on_list_tools=list_tools, on_call_tool=call_tool)
    print("probe started", file=sys.stderr)

    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

    print("probe stopped", file=sys.stderr)


def require(condition: bool, message: str) -> None:
    """Fail the probe even when Python optimizes assertions away."""
    if not condition:
        raise RuntimeError(message)


async def check_results(client: Client, jpeg_output: Path | None) -> None:
    """Check tool discovery, image decoding and the two error channels."""
    tools = (await client.list_tools()).tools
    require([tool.name for tool in tools] == sorted(EXAMPLES["tools"]), "tool list")
    require(all(tool.output_schema is not None for tool in tools), "output schemas")

    for case in (
        "status_ready",
        "capabilities_known",
        "snapshot_fresh",
        "snapshot_stale",
    ):
        fixture = CASES[case]
        expected = fixture["expected"]
        result = await client.call_tool(fixture["tool"], fixture["arguments"])
        is_error = expected["kind"] == "domain_error"
        require(result.is_error == is_error, case)
        (text,) = (block for block in result.content if block.type == "text")
        require(json.loads(text.text) == expected["value"], case)
        require(
            result.structured_content == (None if is_error else expected["value"]), case
        )

        if "jpeg" in expected:
            (image,) = (block for block in result.content if block.type == "image")
            jpeg = base64.b64decode(image.data, validate=True)
            require(len(result.content) == 2 and image.mime_type == "image/jpeg", case)
            require(jpeg == (CONTRACTS / expected["jpeg"]).read_bytes(), case)
            require(len(jpeg) == expected["value"]["jpeg_bytes"], case)

            if jpeg_output is not None:
                with jpeg_output.open("xb") as output:
                    output.write(jpeg)

        else:
            require(len(result.content) == 1, case)

    try:
        await client.call_tool("camera_unknown")

    except MCPError as error:
        require(error.code == INVALID_PARAMS, "unknown tool protocol error")

    else:
        raise RuntimeError("Unknown tool completed without a protocol error")


async def check_cancel(client: Client) -> None:
    """Cancel after server progress, then verify the wait released on that pipe."""

    async def on_progress(
        progress: float, total: float | None, message: str | None
    ) -> None:
        cancel.cancel()

    with CancelScope() as cancel:
        await client.call_tool(
            "camera_snapshot", {"wait_ms": 1000}, progress_callback=on_progress
        )

    require(cancel.cancelled_caught, "client did not cancel the pending call")
    require(len((await client.list_tools()).tools) == 3, "server wait did not release")


async def check_wire() -> None:
    """Verify the unsupported-version response, retry and EOF shutdown directly."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        __file__,
        "--serve",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdin, stdout, stderr = process.stdin, process.stdout, process.stderr

    if stdin is None or stdout is None or stderr is None:
        raise RuntimeError("Probe subprocess is missing stdio pipes")

    try:
        with fail_after(DEADLINE):
            for request_id, revision in enumerate((UNSUPPORTED, REVISION), 1):
                request = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "server/discover",
                    "params": {
                        "_meta": {
                            "io.modelcontextprotocol/protocolVersion": revision,
                            "io.modelcontextprotocol/clientCapabilities": {},
                        }
                    },
                }
                stdin.write(json.dumps(request).encode("utf-8") + b"\n")
                await stdin.drain()
                line = await stdout.readline()
                require(line.endswith(b"\n"), "missing response newline")
                reply = json.loads(line)
                require(
                    reply["jsonrpc"] == "2.0" and reply["id"] == request_id, "framing"
                )

                if revision == UNSUPPORTED:
                    error = reply["error"]
                    require(error["code"] == -32022, "unsupported revision error code")
                    require(
                        error["data"]["requested"] == UNSUPPORTED
                        and error["data"]["supported"] == [REVISION],
                        "unsupported revision error data",
                    )

                else:
                    require(
                        reply["result"]["supportedVersions"] == [REVISION], "wire retry"
                    )

            stdin.close()
            require(await stdout.read() == b"", "unexpected data on protocol stdout")
            require(
                (await stderr.read()).splitlines()
                == [b"probe started", b"probe stopped"],
                "stderr diagnostics or graceful cleanup missing",
            )
            require(await process.wait() == 0, "server failed after EOF")

    finally:
        if process.returncode is None:
            process.kill()
            with fail_after(DEADLINE):
                await process.wait()


async def probe(jpeg_output: Path | None) -> None:
    """Run the official SDK client over two fresh child processes."""
    server = StdioServerParameters(command=sys.executable, args=[__file__, "--serve"])

    with fail_after(DEADLINE):
        async with Client(server) as client:
            require(client.protocol_version == REVISION, "client protocol revision")
            require(
                client.server_info is not None and client.server_info.name == NAME,
                "discovery",
            )
            await check_results(client, jpeg_output)
            await check_cancel(client)

        async with Client(server) as client:
            require(
                (await client.call_tool("camera_status")).structured_content
                == CASES["status_ready"]["expected"]["value"],
                "call after relaunch",
            )

    await check_wire()
    print(
        f"MCP SDK {version('mcp')} / {REVISION}: tools, JPEG and metadata, errors, cancellation, relaunch and EOF passed"
    )


def main() -> None:
    """Start the fixture subprocess or run the profile checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--jpeg-output", type=Path, help="Write the received fixture JPEG"
    )
    args = parser.parse_args()
    asyncio.run(serve() if args.serve else probe(args.jpeg_output))


if __name__ == "__main__":
    main()
