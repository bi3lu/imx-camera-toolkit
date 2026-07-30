"""Tests for the stable raw-frame camera read contract."""

from __future__ import annotations

import threading
import time

import pytest

from imx_camera_toolkit import Camera, CameraDependencyError, CameraFrame, Frame
from packages.camera import camera as camera_module
from packages.camera.backends.gstreamer import GStreamerCaptureBackend
from packages.camera.config import loader as config_loader
from packages.camera.publishing.jpeg import JPEGPublisher


def _running_camera() -> Camera:
    """Create a camera with capture marked active without opening hardware."""
    camera = Camera()
    camera._running.set()
    return camera


def _publish(camera: Camera, image: object) -> None:
    """Publish a raw BGR payload with the camera's resolved metadata."""
    camera._raw_publisher.publish(
        image,
        width=camera.config.output_width,
        height=camera.config.output_height,
    )


def test_read_returns_newest_frame_and_avoids_a_consumer_queue() -> None:
    """Only the newest raw frame must be exposed to an external consumer."""
    camera = _running_camera()
    _publish(camera, bytearray(b"older"))
    _publish(camera, bytearray(b"newest"))

    frame = camera.read(timeout=0, copy=False)

    assert isinstance(frame, CameraFrame)
    assert isinstance(frame, Frame)
    assert frame.sequence == 2
    assert frame.image == bytearray(b"newest")
    assert frame.timestamp_ns > 0
    assert frame.capture_timestamp_ns is None
    assert frame.width == camera.config.output_width
    assert frame.height == camera.config.output_height
    assert frame.format == "BGR"


def test_read_copy_controls_image_ownership() -> None:
    """Default reads must copy while zero-copy reads retain the shared image."""
    camera = _running_camera()
    image = bytearray(b"frame")
    _publish(camera, image)

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
        _publish(camera, bytearray(b"first"))

    publisher = threading.Thread(target=publish_frame)
    publisher.start()
    frame = camera.read(timeout=0.5, copy=False)
    publisher.join()

    assert frame is not None
    assert frame.sequence == 1
    assert frame.image == bytearray(b"first")


def test_read_image_returns_only_the_requested_image_payload() -> None:
    """Compatibility helper must preserve read's copy semantics."""
    camera = _running_camera()
    image = bytearray(b"frame")
    _publish(camera, image)

    shared_image = camera.read_image(timeout=0, copy=False)

    assert shared_image is image


def test_latest_frame_returns_the_current_raw_frame_without_waiting() -> None:
    """Latest-frame access must retain raw metadata and copy configuration."""
    camera = _running_camera()
    image = bytearray(b"frame")
    _publish(camera, image)

    frame = camera.latest_frame(copy=False)

    assert frame is not None
    assert frame.image is image
    assert frame.sequence == 1


def test_latest_jpeg_returns_the_current_preview_payload() -> None:
    """Latest JPEG access must expose preview data independently from raw frames."""
    camera = Camera(enable_preview=True)
    camera._publisher._jpeg = b"preview"

    assert camera.latest_jpeg() == b"preview"


def test_disabled_preview_skips_jpeg_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw-only capture must not invoke the JPEG publisher."""
    camera = Camera(enable_preview=False)

    def fail_publish(self: JPEGPublisher, frame: object) -> bool:
        """Fail if raw-only capture attempts JPEG encoding."""
        raise AssertionError("JPEG encoding must be disabled")

    monkeypatch.setattr(JPEGPublisher, "publish", fail_publish)

    camera._publish_frame(bytearray(b"raw"), timestamp_ns=123)

    frame = camera.latest_frame(copy=False)
    assert frame is not None
    assert frame.image == bytearray(b"raw")
    assert camera.latest_jpeg() is None
    assert camera.frames_encoded == 0
    assert camera.preview_enabled is False


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


def test_camera_rejects_non_boolean_preview_configuration() -> None:
    """JPEG preview configuration must be an explicit boolean option."""
    with pytest.raises(ValueError, match="enable_preview"):
        Camera(enable_preview=1)  # type: ignore[arg-type]


def test_camera_reports_a_missing_system_opencv_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing capture runtime must raise the public dependency exception."""
    monkeypatch.setattr(camera_module, "opencv_available", lambda: False)
    monkeypatch.setattr(
        GStreamerCaptureBackend,
        "available",
        lambda: False,
    )

    with pytest.raises(CameraDependencyError, match="System OpenCV"):
        Camera().start()


def test_camera_config_uses_defaults_when_pyyaml_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core camera construction must not require a YAML parser dependency."""
    monkeypatch.setattr(config_loader, "yaml", None)

    assert config_loader.load_camera_config() == config_loader.DEFAULT_CAMERA_CONFIG
