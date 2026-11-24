"""Launch the local MCP profile with one explicitly owned CPU camera."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from importlib.metadata import version

from ..camera.camera import Camera, CameraConfig
from ..camera.errors import CameraError
from .camera_adapter import CameraReadAdapter
from .host import MCPHost
from .snapshot_adapter import CameraSnapshotAdapter

_SDK_VERSION = "2.3.0"


def _sensor_id(value: str) -> int:
    """Parse an Argus sensor ID before loading the optional runtime."""
    sensor_id = int(value)
    if sensor_id < 0:
        raise argparse.ArgumentTypeError(
            "sensor ID must be greater than or equal to zero"
        )
    return sensor_id


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the standalone camera options to an MCP command parser."""
    parser.add_argument(
        "--sensor-id", type=_sensor_id, default=0, help="Argus CSI sensor ID"
    )
    parser.add_argument(
        "--enable-snapshot",
        action="store_true",
        help="enable the JPEG preview source and grant its MCP snapshot tool",
    )


def run(*, sensor_id: int, enable_snapshot: bool) -> int:
    """Validate the optional runtime and serve one camera until disconnect."""
    config = CameraConfig(sensor_id=sensor_id, enable_preview=enable_snapshot)
    try:
        if version("mcp") != _SDK_VERSION or version("mcp-types") != _SDK_VERSION:
            raise ImportError(f"MCP requires mcp and mcp-types {_SDK_VERSION}")

        from .sdk_server import (
            DEFAULT_TOOLS,
            OBSERVER_SCOPE,
            SNAPSHOT_SCOPE,
            SNAPSHOT_TOOL,
        )
        from .stdio import serve_camera_stdio

    except ImportError:
        print(
            f"The local MCP command requires a working MCP SDK {_SDK_VERSION}. "
            "Install the toolkit's mcp extra in this interpreter:\n"
            f'  uv pip install --python "{sys.executable}" "imx-camera-toolkit[mcp]"',
            file=sys.stderr,
        )
        return 1

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    try:
        camera = Camera(config)
        host = MCPHost(
            CameraReadAdapter(camera, backend="cpu"),
            CameraSnapshotAdapter(camera),
            owned_camera=camera,
        )
        tools = set(DEFAULT_TOOLS)
        grants = {OBSERVER_SCOPE}

        if enable_snapshot:
            tools.add(SNAPSHOT_TOOL)
            grants.add(SNAPSHOT_SCOPE)

        asyncio.run(serve_camera_stdio(host, enabled_tools=tools, grants=grants))

    except (CameraError, OSError, ValueError) as error:
        print(f"MCP startup or transport failed: {error}", file=sys.stderr)
        return 1

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone module with the same options as the installed CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    return run(sensor_id=args.sensor_id, enable_snapshot=args.enable_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
