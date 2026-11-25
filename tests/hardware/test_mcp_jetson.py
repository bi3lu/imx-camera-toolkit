"""Exercise the installed local MCP command against a physical CSI sensor."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from importlib import import_module
from pathlib import Path

import pytest

pytestmark = pytest.mark.hardware


@pytest.mark.parametrize("enable_snapshot", [False, True], ids=["observer", "snapshot"])
def test_packaged_mcp_cpu_camera(enable_snapshot: bool) -> None:
    """Launch the installed CLI over private pipes and close it after capture."""
    sensor = os.environ.get("IMX_CAMERA_SENSOR")
    if sensor not in {"IMX219", "IMX477"}:
        pytest.skip("set IMX_CAMERA_SENSOR to the physical IMX219 or IMX477 sensor")

    pytest.importorskip("mcp")
    from mcp import Client, StdioServerParameters
    from mcp_types import ImageContent, TextContent

    sensor_id = os.environ.get("IMX_CAMERA_SENSOR_ID", "0")
    executable = Path(sys.executable).with_name("imx-camera")
    assert (
        executable.is_file()
    ), f"install the toolkit in {sys.executable}'s environment"
    arguments = ["mcp", "--sensor-id", sensor_id]
    expected_tools = {"camera_status", "camera_capabilities"}
    if enable_snapshot:
        arguments.append("--enable-snapshot")
        expected_tools.add("camera_snapshot")

    async def smoke() -> None:
        server = StdioServerParameters(command=str(executable), args=arguments)
        async with Client(server) as client:
            assert client.protocol_version == "2026-07-28"
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == expected_tools

            while True:
                status = await client.call_tool("camera_status")
                assert not status.is_error
                assert status.structured_content is not None
                assert status.structured_content["backend"] == "cpu"
                state = status.structured_content["state"]
                if state == "ready":
                    break
                assert state in {"starting", "recovering"}
                await asyncio.sleep(0.1)

            metadata = None
            if enable_snapshot:
                import cv2

                np = import_module("numpy")
                result = await client.call_tool(
                    "camera_snapshot", {"max_age_ms": 2000, "wait_ms": 500}
                )
                assert not result.is_error
                metadata = result.structured_content
                assert metadata is not None
                text, image = result.content
                assert isinstance(text, TextContent)
                assert json.loads(text.text) == metadata
                assert isinstance(image, ImageContent)
                assert image.mime_type == metadata["mime_type"] == "image/jpeg"
                data = base64.b64decode(image.data, validate=True)
                assert len(data) == metadata["jpeg_bytes"] <= 524_288
                decoded = cv2.imdecode(
                    np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                assert decoded is not None
                assert decoded.shape == (metadata["height"], metadata["width"], 3)
                assert metadata["sequence_domain"] == "jpeg_publication"
                assert metadata["age_clock"] == "server_monotonic"
                assert 0 <= metadata["age_ms"] <= 2000

                fixed_source = await client.call_tool(
                    "camera_snapshot", {"max_width": 1}
                )
                assert fixed_source.is_error
                (error_block,) = fixed_source.content
                assert isinstance(error_block, TextContent)
                error = json.loads(error_block.text)
                assert error["code"] == "snapshot_unavailable"
                assert error["details"]["reason"] == "constraints_unsupported"

            capabilities = await client.call_tool("camera_capabilities")
            assert not capabilities.is_error
            assert capabilities.structured_content is not None
            assert capabilities.structured_content["frame_format"] == "BGR_CPU"
            snapshot = capabilities.structured_content["snapshot"]
            assert snapshot["configured"] is enable_snapshot
            if enable_snapshot:
                assert snapshot["source_supported"] is True

            print(
                json.dumps(
                    {
                        "protocol": client.protocol_version,
                        "sensor": sensor,
                        "sensor_id": int(sensor_id),
                        "tools": sorted(expected_tools),
                        "status": status.structured_content,
                        "capabilities": capabilities.structured_content,
                        "snapshot": metadata,
                    },
                    sort_keys=True,
                )
            )

    asyncio.run(asyncio.wait_for(smoke(), timeout=30))
