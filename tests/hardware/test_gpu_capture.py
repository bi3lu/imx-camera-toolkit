"""Opt-in NVMM validation against a physical Jetson CSI camera."""

from __future__ import annotations

import os
import time

import pytest

from imx_camera_toolkit import CameraConfig, FrameFormat, GpuCamera, MemoryType

pytestmark = pytest.mark.hardware

SENSOR = os.environ.get("IMX_CAMERA_SENSOR")
SENSOR_ID_TEXT = os.environ.get("IMX_CAMERA_SENSOR_ID", "0")

if SENSOR not in {"IMX219", "IMX477"}:
    pytest.skip(
        "set IMX_CAMERA_SENSOR to IMX219 or IMX477 for hardware validation",
        allow_module_level=True,
    )

try:
    SENSOR_ID = int(SENSOR_ID_TEXT)

except ValueError:
    pytest.skip(
        "IMX_CAMERA_SENSOR_ID must be an integer",
        allow_module_level=True,
    )


@pytest.mark.parametrize(("width", "height"), [(1280, 720), (1920, 1080)])
def test_physical_sensor_delivers_nvmm_at_30_fps(width: int, height: int) -> None:
    """Exercise both tee branches and measure unique NVMM frame delivery."""
    config = CameraConfig(
        sensor_id=SENSOR_ID,
        capture_width=width,
        capture_height=height,
        output_width=width,
        output_height=height,
        fps=30,
        enable_preview=True,
    )
    timestamps_ns: list[int] = []
    previous_sequence = 0

    with GpuCamera(config) as camera:
        deadline = time.monotonic() + 10.0

        while len(timestamps_ns) < 60 and time.monotonic() < deadline:
            frame = camera.read(timeout=0.5)

            if frame is None or frame.sequence == previous_sequence:
                continue

            assert frame.format is FrameFormat.NV12_NVMM
            assert frame.memory_type is MemoryType.NVMM
            assert frame.width == width
            assert frame.height == height
            assert frame.payload() is not None
            timestamps_ns.append(frame.timestamp_ns)
            previous_sequence = frame.sequence

        _, jpeg = camera.wait_for_jpeg(0, timeout=2.0)

    assert len(timestamps_ns) == 60, f"{SENSOR} did not deliver 60 NVMM frames"
    elapsed_seconds = (timestamps_ns[-1] - timestamps_ns[0]) / 1_000_000_000
    observed_fps = (len(timestamps_ns) - 1) / elapsed_seconds
    assert 27.0 <= observed_fps <= 33.0
    assert jpeg is not None and jpeg.startswith(b"\xff\xd8")
