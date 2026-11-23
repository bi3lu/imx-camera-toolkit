"""Serve one application-owned camera over stdio for the SDK comparison."""

import argparse
import os
import sys
from contextlib import ExitStack, redirect_stdout
from importlib.metadata import version
from pathlib import Path

from imx_camera_toolkit import Camera, CameraConfig
from imx_camera_toolkit.testing import MockCamera

FIXTURE = Path(__file__).with_name("synthetic.jpg")


def serve(camera: Camera | MockCamera) -> None:
    """Borrow a running camera and expose two read-only smoke tools."""
    if version("mcp").startswith("1."):
        from mcp.server.fastmcp import FastMCP, Image

        server = FastMCP("imx-camera-runtime-spike")

    else:
        from mcp.server import MCPServer
        from mcp.server.mcpserver import Image

        server = MCPServer("imx-camera-runtime-spike")

    @server.tool()
    def camera_status() -> dict[str, bool | int | str | None]:
        """Return running state, backend and captured-frame count."""
        stats = camera.stats()
        return {
            "running": stats.running,
            "backend": camera.active_backend,
            "captured_frames": stats.captured_frames,
        }

    @server.tool()
    def camera_snapshot() -> Image:
        """Return a JPEG from the existing publisher after a bounded wait."""
        _, jpeg = camera.wait_for_jpeg(0, timeout=3.0)
        if jpeg is None:
            raise RuntimeError("JPEG unavailable before the snapshot deadline")

        return Image(data=jpeg, format="jpeg")

    server.run(transport="stdio")


def main() -> None:
    """Own one camera for the lifetime of this subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jetson", action="store_true")
    parser.add_argument("--sensor-id", type=int, default=0)
    args = parser.parse_args()

    if args.sensor_id < 0:
        parser.error("--sensor-id must be nonnegative")

    stdout_fd = sys.stdout.fileno()
    sys.stdout.flush()

    with (
        os.fdopen(os.dup(stdout_fd), "w", encoding="utf-8") as protocol_stream,
        ExitStack() as resources,
    ):
        # Keep native Argus output off the MCP wire, including during camera startup.
        os.dup2(sys.stderr.fileno(), stdout_fd)

        with redirect_stdout(protocol_stream):
            if args.jetson:
                camera = resources.enter_context(
                    Camera(
                        config=CameraConfig(sensor_id=args.sensor_id),
                        enable_preview=True,
                    )
                )

            else:
                camera = MockCamera(output_width=1, output_height=1)
                resources.callback(camera.stop)
                camera.publish_jpeg(FIXTURE.read_bytes())

            serve(camera)


if __name__ == "__main__":
    main()
