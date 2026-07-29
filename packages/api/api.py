"""FastAPI application exposing IMX camera snapshots and MJPEG streams."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from packages.camera.camera import Camera
from packages.stream.stream import MJPEGStream

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import Response, StreamingResponse

except ImportError:
    FastAPI: Any | None = None
    HTTPException: Any | None = None
    Query: Any | None = None
    Response: Any | None = None
    StreamingResponse: Any | None = None


SNAPSHOT_TIMEOUT = 2.0
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _camera_status(camera: Camera) -> dict[str, object]:
    """Return JSON-serializable state and capture metrics for a camera.

    Args:
        camera: Camera instance to inspect.

    Returns:
        Health status, frame availability, capture metrics, and the last
        background capture error when one exists.
    """
    last_error = str(camera.last_error) if camera.last_error is not None else None

    if camera.running:
        status = "ok"

    elif last_error is not None:
        status = "error"

    else:
        status = "unavailable"

    return {
        "status": status,
        "camera_running": camera.running,
        "frame_available": camera.frame_available,
        "frame_number": camera.frame_number,
        "frames_captured": camera.frames_captured,
        "frames_encoded": camera.frames_encoded,
        "last_frame_time": camera.last_frame_time,
        "last_error": last_error,
    }


def create_app(camera: Camera | None = None) -> Any:
    """Create a FastAPI application backed by one shared camera instance.

    The camera is started during the FastAPI lifespan startup event and stopped
    during shutdown. It is deliberately shared by the snapshot and MJPEG
    endpoints instead of being created once per HTTP request.

    Args:
        camera: Camera to expose. When omitted, creates a default ``Camera``.

    Returns:
        Configured FastAPI application.

    Raises:
        RuntimeError: If FastAPI is unavailable in the current environment.
    """
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI is not installed. Add FastAPI to the project's uv "
            "dependencies before creating the API application."
        )

    shared_camera = camera if camera is not None else Camera()

    @asynccontextmanager
    async def lifespan(_: Any):
        """Start the shared camera for the lifetime of the API application."""
        shared_camera.start()

        try:
            yield

        finally:
            shared_camera.stop()

    application = FastAPI(
        title="IMX Camera API",
        description="Snapshots and MJPEG streaming for an NVIDIA Jetson CSI camera.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/")
    def index() -> dict[str, str]:
        """Return links to the camera API endpoints.

        Returns:
            Available camera API endpoint paths.
        """
        return {
            "health": "/api/health",
            "snapshot": "/api/camera/snapshot",
            "mjpeg": "/api/camera/mjpeg",
        }

    @application.get("/api/health")
    def health() -> dict[str, object]:
        """Return camera health and capture metrics.

        Returns:
            JSON-serializable state for the shared camera.
        """
        return _camera_status(shared_camera)

    @application.get("/api/camera/snapshot")
    def camera_snapshot(after: int = Query(default=-1, ge=-1)) -> Any:
        """Return the newest JPEG camera frame.

        Args:
            after: Previously consumed frame number. A newer frame is awaited
                for up to two seconds when this value is non-negative.

        Returns:
            JPEG response with the current frame number in ``X-Frame-Number``.

        Raises:
            HTTPException: If the camera has not produced a frame.
        """
        frame_number, jpeg = shared_camera.wait_for_jpeg(
            previous_frame_number=after,
            timeout=SNAPSHOT_TIMEOUT,
        )
        if jpeg is None:
            raise HTTPException(status_code=503, detail="Camera frame unavailable")

        if after >= 0 and frame_number == after:
            return Response(status_code=204, headers=NO_CACHE_HEADERS)

        headers = {
            **NO_CACHE_HEADERS,
            "X-Frame-Number": str(frame_number),
        }

        return Response(content=jpeg, media_type="image/jpeg", headers=headers)

    @application.get("/api/camera/mjpeg")
    def camera_mjpeg() -> Any:
        """Return a live MJPEG response from the shared camera.

        Returns:
            Multipart HTTP response that emits only the newest JPEG frames.
        """
        stream = MJPEGStream(shared_camera)
        return StreamingResponse(
            stream,
            media_type=stream.content_type,
            headers=NO_CACHE_HEADERS,
        )

    return application


app = create_app() if FastAPI is not None else None
