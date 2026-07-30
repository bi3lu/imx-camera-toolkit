"""Integration tests for FastAPI endpoints with no Jetson hardware."""

from __future__ import annotations

from fastapi.testclient import TestClient

from packages.api.api import create_app
from packages.testing.mock_camera import MockCamera


def test_health_and_snapshot_with_mock_camera() -> None:
    """API health and snapshot endpoints must use one supplied mock camera."""
    camera = MockCamera(auto_start=False)
    application = create_app(camera)  # type: ignore[arg-type]

    with TestClient(application) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["camera_running"] is True

        camera.publish_jpeg(b"\xff\xd8mock\xff\xd9")
        snapshot = client.get("/api/camera/snapshot")
        assert snapshot.status_code == 200
        assert snapshot.headers["content-type"] == "image/jpeg"
        assert snapshot.content == b"\xff\xd8mock\xff\xd9"


def test_browser_view_is_served_with_mock_camera() -> None:
    """The customizable view must be available without opening hardware."""
    application = create_app(MockCamera())  # type: ignore[arg-type]

    with TestClient(application) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "/api/camera/mjpeg" in response.text
