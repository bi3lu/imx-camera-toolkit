"""Tests for the stable raw-frame camera read contract."""

from __future__ import annotations

import threading
import time

import pytest

from imx_camera_toolkit import Camera, CameraFrame


def _running_camera() -> Camera:
    """Create a camera with capture marked active without opening hardware."""
    camera = Camera()
    camera._running.set()
    return camera


def test_read_returns_newest_frame_and_avoids_a_consumer_queue() -> None:
    """Only the newest raw frame must be exposed to an external consumer."""
    camera = _running_camera()
    camera._raw_publisher.publish(bytearray(b"older"))
    camera._raw_publisher.publish(bytearray(b"newest"))

    frame = camera.read(timeout=0, copy=False)

    assert isinstance(frame, CameraFrame)
    assert frame.sequence == 2
    assert frame.image == bytearray(b"newest")


def test_read_copy_controls_image_ownership() -> None:
    """Default reads must copy while zero-copy reads retain the shared image."""
    camera = _running_camera()
    image = bytearray(b"frame")
    camera._raw_publisher.publish(image)

    copied_frame = camera.read(timeout=0)
    shared_frame = camera.read(timeout=0, copy=False)

    assert copied_frame is not None
    assert copied_frame.image == image
    assert copied_frame.image is not image
    assert shared_frame is not None
    assert shared_frame.image is image


def test_read_returns_none_after_timeout_without_a_raw_frame() -> None:
    """An active camera with no available raw frame must honor the timeout."""
    camera = _running_camera()

    assert camera.read(timeout=0) is None


def test_read_waits_for_the_first_published_raw_frame() -> None:
    """A camera with no initial frame must wait until a frame is published."""
    camera = _running_camera()

    def publish_frame() -> None:
        """Publish one frame after read has started waiting."""
        time.sleep(0.01)
        camera._raw_publisher.publish(bytearray(b"first"))

    publisher = threading.Thread(target=publish_frame)
    publisher.start()
    frame = camera.read(timeout=0.5, copy=False)
    publisher.join()

    assert frame is not None
    assert frame.sequence == 1
    assert frame.image == bytearray(b"first")


def test_read_requires_an_active_camera() -> None:
    """Callers must start capture before requesting a raw frame."""
    with pytest.raises(RuntimeError, match="camera is not running"):
        Camera().read(timeout=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout": -1}, "timeout"),
        ({"timeout": float("inf")}, "timeout"),
        ({"copy": 1}, "copy"),
    ],
)
def test_read_validates_arguments(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Invalid read options must fail before waiting for a frame."""
    camera = _running_camera()

    with pytest.raises(ValueError, match=message):
        camera.read(**kwargs)  # type: ignore[arg-type]
