"""Verify MCP discovery, tool listing, status and one JPEG over real stdio."""

import argparse
import asyncio
import base64
import json
import sys
from importlib.metadata import version
from pathlib import Path

from anyio import fail_after
from mcp import Client, StdioServerParameters
from server import FIXTURE

SERVER = Path(__file__).with_name("server.py")


def validate_jpeg(jpeg: bytes, jetson: bool) -> None:
    """Verify the fixture bytes or decode the physical camera image in memory."""
    if not jetson:
        if jpeg != FIXTURE.read_bytes():
            raise RuntimeError("MCP returned different bytes from the mock publisher")

        return

    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    if frame is None or frame.shape != (720, 1280, 3):
        raise RuntimeError("MCP snapshot did not decode as a 1280x720 BGR image")


async def smoke(server_python: str, jetson: bool, sensor_id: int, legacy: bool) -> None:
    """Run the SDK client and verify the expected server replies."""
    server_args = [str(SERVER)]

    if jetson:
        server_args.extend(("--jetson", "--sensor-id", str(sensor_id)))

    server = StdioServerParameters(command=server_python, args=server_args)

    with fail_after(30):
        async with Client(server, mode="legacy" if legacy else "auto") as client:
            if (
                client.server_info is None
                or client.server_info.name != "imx-camera-runtime-spike"
            ):
                raise RuntimeError("MCP discovery returned an unexpected server")

            tools = sorted(tool.name for tool in (await client.list_tools()).tools)

            if tools != ["camera_snapshot", "camera_status"]:
                raise RuntimeError("MCP returned an unexpected tool list")

            snapshot = await client.call_tool("camera_snapshot")

            if snapshot.is_error or len(snapshot.content) != 1:
                raise RuntimeError("MCP did not return a single snapshot")

            image = snapshot.content[0]

            if image.type != "image" or image.mime_type != "image/jpeg":
                raise RuntimeError("MCP snapshot did not contain JPEG image content")

            jpeg = base64.b64decode(image.data, validate=True)
            validate_jpeg(jpeg, jetson)

            status = await client.call_tool("camera_status")
            data = status.structured_content

            if (
                status.is_error
                or not isinstance(data, dict)
                or data.get("running") is not True
            ):
                raise RuntimeError("MCP status did not report a running camera")

            frames = data.get("captured_frames")

            if type(frames) is not int or frames < 1:
                raise RuntimeError("MCP status did not report captured frames")

            backend = data.get("backend")

            if not backend or (backend == "mock") == jetson:
                raise RuntimeError(
                    "MCP status did not report the requested camera backend"
                )

            evidence = {
                "client_sdk": version("mcp"),
                "protocol": client.protocol_version,
                "tools": tools,
                "status": data,
                "jpeg_bytes": len(jpeg),
            }

    print(json.dumps(evidence, sort_keys=True))


def main() -> None:
    """Launch the chosen server interpreter and print evidence without pixels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument(
        "--legacy", action="store_true", help="Use the initialize handshake"
    )
    parser.add_argument("--jetson", action="store_true")
    parser.add_argument("--sensor-id", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(smoke(args.server_python, args.jetson, args.sensor_id, args.legacy))


if __name__ == "__main__":
    main()
