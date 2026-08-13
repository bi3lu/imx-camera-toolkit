"""Integration tests for FastAPI endpoints with no Jetson hardware."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

import pytest
from fastapi import Response

from imx_camera_toolkit import GpuCamera
from imx_camera_toolkit._internal.api.api import create_app
from imx_camera_toolkit._internal.testing.mock_camera import MockCamera


def _endpoint(application: Any, path: str) -> Callable[..., Any]:
    """Resolve one registered endpoint without Starlette's test portal."""
    for route in application.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint  # type: ignore[no-any-return]
    raise LookupError(path)


def _run_lifespan(application: Any, operation: Callable[[], None]) -> None:
    """Execute assertions inside the application's explicit lifespan."""

    async def run() -> None:
        async with application.router.lifespan_context(application):
            operation()

    asyncio.run(run())


def test_health_and_snapshot_with_mock_camera() -> None:
    """API health and snapshot endpoints must use one supplied mock camera."""
    camera = MockCamera(auto_start=False)
    camera.record_stage_latency("inference", 2_000_000)
    camera.record_consumer_drop("inference", 3)
    application = create_app(camera)  # type: ignore[arg-type]

    def verify() -> None:
        """Verify health and a subsequently published snapshot."""
        health = _endpoint(application, "/api/health")()
        assert health["camera_running"] is True
        assert health["dropped_frames"] == 0
        assert health["capture_fps"] == 0.0
        assert health["last_frame_timestamp_ns"] is None
        assert health["consecutive_failures"] == 0
        assert health["active_backend"] == "mock"
        assert health["frame_format"] == "BGR_CPU"
        assert health["frame_memory_type"] == "CPU"
        assert health["frame_resolution"] == {
            "width": 1280,
            "height": 720,
        }
        assert health["consumer_dropped_frames"] == {"inference": 3}
        assert set(health["stage_latency_ns"]) == {
            "transfer",
            "inference",
            "encoder",
            "end_to_end",
        }
        assert health["stage_latency_ns"]["inference"] == {
            "samples": 1,
            "last": 2_000_000,
            "mean": 2_000_000.0,
            "max": 2_000_000,
        }

        camera.publish_jpeg(b"\xff\xd8mock\xff\xd9")
        snapshot = cast(
            Response,
            _endpoint(application, "/api/camera/snapshot")(-1),
        )
        assert snapshot.status_code == 200
        assert snapshot.media_type == "image/jpeg"
        assert snapshot.body == b"\xff\xd8mock\xff\xd9"

    _run_lifespan(application, verify)


def test_browser_view_is_served_with_mock_camera() -> None:
    """The default simple browser view must be available without hardware."""
    application = create_app(MockCamera())  # type: ignore[arg-type]

    def request() -> None:
        """Load the bundled simple view."""
        response = cast(Response, _endpoint(application, "/")())
        assert response.status_code == 200
        text = bytes(response.body).decode("utf-8")
        assert "/api/camera/mjpeg" in text
        assert "Camera controls" not in text

    _run_lifespan(application, request)
    assert application.state.view_mode == "simple"


def test_advanced_browser_view_includes_runtime_camera_controls() -> None:
    """The advanced bundled view must expose the runtime camera-control panel."""
    application = create_app(
        MockCamera(),  # type: ignore[arg-type]
        view_mode="advanced",
    )

    def request() -> None:
        """Load the bundled advanced view."""
        response = cast(Response, _endpoint(application, "/")())
        assert response.status_code == 200
        text = bytes(response.body).decode("utf-8")
        assert "/api/camera/mjpeg" in text
        assert "Camera controls" in text
        assert "/api/camera/control" in text

    _run_lifespan(application, request)
    assert application.state.view_mode == "advanced"


def test_unknown_bundled_view_mode_is_rejected() -> None:
    """Invalid view variants must fail during application construction."""
    with pytest.raises(ValueError, match="unknown camera view mode"):
        create_app(MockCamera(), view_mode="unknown")  # type: ignore[arg-type]


def test_api_can_leave_an_existing_camera_lifecycle_to_the_application() -> None:
    """A shared application camera must not be started or stopped by FastAPI."""
    camera = MockCamera(auto_start=False)
    application = create_app(camera, manage_camera=False)  # type: ignore[arg-type]

    def verify() -> None:
        """Observe lifecycle state while the application is active."""
        assert camera.running is False

    _run_lifespan(application, verify)

    assert camera.running is False


def test_mjpeg_api_accepts_the_stable_gpu_camera_contract() -> None:
    """GPU hardware JPEG and runtime controls must use the shared camera API."""
    camera = GpuCamera(enable_preview=True)
    application = create_app(camera, manage_camera=False)

    health = _endpoint(application, "/api/health")()
    controls = _endpoint(application, "/api/camera/control")()

    assert health["frame_format"] == "NV12_NVMM"
    assert health["frame_memory_type"] == "NVMM"
    assert controls["software_hdr"]["supported"] is False
