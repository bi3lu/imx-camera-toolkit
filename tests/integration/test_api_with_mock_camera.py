"""Integration tests for FastAPI endpoints with no Jetson hardware."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from packages.api.api import create_app
from packages.testing.mock_camera import MockCamera


def test_health_and_snapshot_with_mock_camera() -> None:
    """API health and snapshot endpoints must use one supplied mock camera."""
    camera = MockCamera(auto_start=False)
    camera.record_stage_latency("inference", 2_000_000)
    camera.record_consumer_drop("inference", 3)
    application = create_app(camera)  # type: ignore[arg-type]

    with TestClient(application) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["camera_running"] is True
        assert health.json()["dropped_frames"] == 0
        assert health.json()["capture_fps"] == 0.0
        assert health.json()["last_frame_timestamp_ns"] is None
        assert health.json()["consecutive_failures"] == 0
        assert health.json()["active_backend"] == "mock"
        assert health.json()["frame_format"] == "BGR_CPU"
        assert health.json()["frame_memory_type"] == "CPU"
        assert health.json()["frame_resolution"] == {
            "width": 1280,
            "height": 720,
        }
        assert health.json()["consumer_dropped_frames"] == {"inference": 3}
        assert set(health.json()["stage_latency_ns"]) == {
            "transfer",
            "inference",
            "encoder",
            "end_to_end",
        }
        assert health.json()["stage_latency_ns"]["inference"] == {
            "samples": 1,
            "last": 2_000_000,
            "mean": 2_000_000.0,
            "max": 2_000_000,
        }

        camera.publish_jpeg(b"\xff\xd8mock\xff\xd9")
        snapshot = client.get("/api/camera/snapshot")
        assert snapshot.status_code == 200
        assert snapshot.headers["content-type"] == "image/jpeg"
        assert snapshot.content == b"\xff\xd8mock\xff\xd9"


def test_browser_view_is_served_with_mock_camera() -> None:
    """The default simple browser view must be available without hardware."""
    application = create_app(MockCamera())  # type: ignore[arg-type]

    with TestClient(application) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "/api/camera/mjpeg" in response.text
    assert "Camera controls" not in response.text
    assert application.state.view_mode == "simple"


def test_advanced_browser_view_includes_runtime_camera_controls() -> None:
    """The advanced bundled view must expose the runtime camera-control panel."""
    application = create_app(
        MockCamera(),  # type: ignore[arg-type]
        view_mode="advanced",
    )

    with TestClient(application) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "/api/camera/mjpeg" in response.text
    assert "Camera controls" in response.text
    assert "/api/camera/control" in response.text
    assert application.state.view_mode == "advanced"


def test_unknown_bundled_view_mode_is_rejected() -> None:
    """Invalid view variants must fail during application construction."""
    with pytest.raises(ValueError, match="unknown camera view mode"):
        create_app(MockCamera(), view_mode="unknown")  # type: ignore[arg-type]


def test_api_can_leave_an_existing_camera_lifecycle_to_the_application() -> None:
    """A shared application camera must not be started or stopped by FastAPI."""
    camera = MockCamera(auto_start=False)
    application = create_app(camera, manage_camera=False)  # type: ignore[arg-type]

    with TestClient(application):
        assert camera.running is False

    assert camera.running is False
