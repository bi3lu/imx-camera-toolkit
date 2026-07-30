"""Tests for generic, model-agnostic browser preview transport."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from imx_camera_toolkit import Camera
from imx_camera_toolkit.preview import PreviewServer
from packages.camera.models import Frame
from packages.camera.publishing import JPEGPublisher


def test_preview_server_publishes_opaque_images_without_model_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic preview publication must forward only image payloads to JPEG."""
    published: list[object] = []

    def fake_publish(self: JPEGPublisher, image: object) -> bool:
        """Record the opaque image instead of invoking system OpenCV."""
        published.append(image)
        return True

    monkeypatch.setattr(JPEGPublisher, "publish", fake_publish)
    server = PreviewServer()
    image = bytearray(b"annotated-image")
    frame = Frame(
        image=image,
        sequence=1,
        timestamp_ns=1,
        capture_timestamp_ns=None,
        width=1,
        height=1,
        format="BGR",
    )

    assert server.publish(image) is True
    assert server.publish(frame) is True
    assert published == [image, image]


def test_preview_server_shares_camera_source_without_starting_capture() -> None:
    """A camera source must be reused without a second capture lifecycle."""
    camera = Camera()
    server = PreviewServer(source=camera)

    server.start()
    try:
        assert server.source is camera
        assert camera.running is False
        assert camera._backend is None
    finally:
        server.stop()


def test_preview_server_creates_a_transport_only_fastapi_application() -> None:
    """The generic HTTP app must expose images without camera-control routes."""
    server = PreviewServer()
    application = server.create_app()

    with TestClient(application) as client:
        response = client.get("/")
        unavailable = client.get("/api/camera/snapshot")
        controls = client.get("/api/camera/control")

    assert response.status_code == 200
    assert unavailable.status_code == 503
    assert controls.status_code == 404
