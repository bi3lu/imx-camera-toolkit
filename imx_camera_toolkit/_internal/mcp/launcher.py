"""Launch the local MCP profile with one explicitly owned CPU camera."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

from ..camera.camera import Camera, CameraConfig
from ..camera.errors import CameraError
from .camera_adapter import CameraReadAdapter
from .host import MCPHost
from .snapshot_adapter import CameraSnapshotAdapter


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the launch and serve one camera until the client disconnects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-id", type=int, default=0, help="Argus CSI sensor ID")
    parser.add_argument(
        "--enable-snapshot",
        action="store_true",
        help="enable the JPEG preview source and grant its MCP snapshot tool",
    )
    args = parser.parse_args(argv)
    try:
        config = CameraConfig(
            sensor_id=args.sensor_id, enable_preview=args.enable_snapshot
        )

    except ValueError as error:
        parser.error(str(error))

    try:
        from .sdk_server import (
            DEFAULT_TOOLS,
            OBSERVER_SCOPE,
            SNAPSHOT_SCOPE,
            SNAPSHOT_TOOL,
        )
        from .stdio import serve_camera_stdio

    except ModuleNotFoundError:
        print(
            "Install the local MCP runtime with "
            f'uv pip install --python "{sys.executable}" "mcp==2.3.0".',
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

        if args.enable_snapshot:
            tools.add(SNAPSHOT_TOOL)
            grants.add(SNAPSHOT_SCOPE)

        asyncio.run(serve_camera_stdio(host, enabled_tools=tools, grants=grants))

    except (CameraError, OSError, ValueError) as error:
        print(f"MCP startup or transport failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
