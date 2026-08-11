"""Unit tests for the deterministic mock camera."""

from __future__ import annotations

import pytest

from imx_camera_toolkit._internal.testing.mock_camera import MockCamera


def test_mock_camera_publishes_and_waits_for_jpeg() -> None:
    """A published JPEG must be observable with its assigned frame number."""
    camera = MockCamera()

    assert camera.publish_jpeg(b"jpeg") == 1
    assert camera.wait_for_jpeg(0, timeout=0) == (1, b"jpeg")
    assert camera.frames_captured == 1
    assert camera.frames_encoded == 1
    assert camera.stats().captured_frames == 1
    assert camera.stats().last_frame_timestamp_ns is not None


def test_mock_camera_rejects_publication_after_stop() -> None:
    """Stopped cameras must not silently accept synthetic frames."""
    camera = MockCamera()
    camera.stop()

    with pytest.raises(RuntimeError, match="not running"):
        camera.publish_jpeg(b"jpeg")
