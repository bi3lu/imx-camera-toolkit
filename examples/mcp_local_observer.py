"""Read camera status, capabilities, and an optional JPEG over local MCP."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any

from anyio import fail_after
from mcp import Client, MCPError, StdioServerParameters
from mcp_types import CallToolResult, ImageContent, TextContent

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


_CLIENT_ERRORS = (MCPError, OSError, RuntimeError, ValueError)


def _error_messages(error: Exception) -> list[str]:
    """Extract expected failures from the SDK's stdio task groups."""
    if isinstance(error, ExceptionGroup):
        return [
            message for child in error.exceptions for message in _error_messages(child)
        ]

    if isinstance(error, _CLIENT_ERRORS):
        return [str(error)]

    raise error


def _metadata(tool: str, result: CallToolResult) -> dict[str, Any]:
    """Return structured metadata or report the tool's error block."""
    if result.is_error:
        detail = next(
            (block.text for block in result.content if isinstance(block, TextContent)),
            "No error details returned.",
        )
        raise RuntimeError(f"{tool} failed: {detail}")

    metadata = result.structured_content

    if not isinstance(metadata, dict):
        raise RuntimeError(f"{tool} returned no structured metadata.")

    return metadata


async def observe(sensor_id: int, snapshot_output: Path | None) -> None:
    """Launch one camera, read its tools, then close the child before exiting.

    Args:
        sensor_id: Zero-based Argus sensor ID on this Jetson.
        snapshot_output: Destination for a new JPEG, or None for observer mode.
    """
    executable = Path(sys.executable).absolute().with_name("imx-camera")

    if not executable.is_file():
        raise RuntimeError(
            f"Install the toolkit in the client environment: {executable}"
        )

    arguments = ["mcp", "--sensor-id", str(sensor_id)]
    expected_tools = {"camera_status", "camera_capabilities"}

    if snapshot_output is not None:
        arguments.append("--enable-snapshot")
        expected_tools.add("camera_snapshot")

    server = StdioServerParameters(command=str(executable), args=arguments)

    with fail_after(30):
        async with Client(server) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}

            if tools != expected_tools:
                raise RuntimeError(f"Unexpected tools: {', '.join(sorted(tools))}")

            protocol = client.protocol_version

            while True:
                status = _metadata(
                    "camera_status", await client.call_tool("camera_status")
                )
                if status["state"] == "ready":
                    break
                if status["state"] not in {"starting", "recovering"}:
                    raise RuntimeError(f"Camera state is {status['state']}.")
                await asyncio.sleep(0.1)

            report = {
                "protocol": protocol,
                "tools": sorted(tools),
                "status": status,
                "capabilities": _metadata(
                    "camera_capabilities", await client.call_tool("camera_capabilities")
                ),
            }

            if snapshot_output is not None:
                result = await client.call_tool(
                    "camera_snapshot", {"max_age_ms": 2000, "wait_ms": 500}
                )
                metadata = _metadata("camera_snapshot", result)
                images = [
                    block for block in result.content if isinstance(block, ImageContent)
                ]

                if len(images) != 1 or images[0].mime_type != "image/jpeg":
                    raise RuntimeError("camera_snapshot must return one JPEG image.")

                jpeg = base64.b64decode(images[0].data, validate=True)

                if len(jpeg) != metadata["jpeg_bytes"]:
                    raise RuntimeError("JPEG size differs from the snapshot metadata.")

                report["snapshot"] = metadata

                with snapshot_output.open("xb") as output:
                    output.write(jpeg)

    print(json.dumps(report, indent=2))


def main() -> None:
    """Parse launch options and report expected failures on stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-id", type=int, default=0, help="Argus CSI sensor ID")
    parser.add_argument(
        "--snapshot-output",
        type=Path,
        help="enable snapshots and save one JPEG to a new file",
    )
    args = parser.parse_args()

    if args.sensor_id < 0:
        parser.error("--sensor-id must be non-negative")

    try:
        asyncio.run(observe(args.sensor_id, args.snapshot_output))

    except TimeoutError:
        parser.exit(1, "Timed out waiting for the camera or MCP response.\n")

    except (*_CLIENT_ERRORS, ExceptionGroup) as error:
        parser.exit(1, "\n".join(_error_messages(error)) + "\n")


if __name__ == "__main__":
    main()
