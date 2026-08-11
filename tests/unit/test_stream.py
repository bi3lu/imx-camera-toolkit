"""Unit tests for MJPEG framing with the mock camera contract."""

from __future__ import annotations

from imx_camera_toolkit._internal.stream.stream import MJPEGStream
from imx_camera_toolkit._internal.testing.mock_camera import MockCamera


def test_mjpeg_stream_yields_latest_mock_camera_frame() -> None:
    """The first stream item must frame an already available JPEG immediately."""
    camera = MockCamera()
    camera.publish_jpeg(b"jpeg")

    part = next(iter(MJPEGStream(camera, boundary="test", timeout=0.01)))

    assert part.startswith(b"--test\r\n")
    assert b"Content-Type: image/jpeg" in part
    assert part.endswith(b"jpeg\r\n")
