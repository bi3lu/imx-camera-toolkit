"""Convenient browser-preview facade for a single IMX camera."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn

from packages.preview import PreviewServer, PreviewSource

from .api import ViewMode, create_app
from .camera import Camera, CameraConfig

__all__ = [
    "CameraPreview",
    "PreviewServer",
    "PreviewSource",
    "create_preview_app",
    "preview",
    "serve",
]


@dataclass(frozen=True)
class CameraPreview:
    """Configure and run a browser preview using the existing camera API.

    The facade composes :class:`Camera` with the FastAPI application factory.
    Camera startup and cleanup remain owned by the API lifespan handler, so the
    camera is opened when Uvicorn starts and stopped during graceful shutdown.

    Args:
        sensor_id: Zero-based CSI sensor identifier.
        width: Capture and output frame width, in pixels.
        height: Capture and output frame height, in pixels.
        fps: Capture and JPEG encoding limit, in frames per second.
        host: Address on which to expose the preview server.
        port: TCP port on which to expose the preview server.
    """

    sensor_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    host: str = "0.0.0.0"
    port: int = 8000

    def __post_init__(self) -> None:
        """Validate the preview configuration before creating camera resources.

        Raises:
            ValueError: If a camera setting, host, or port is invalid.
        """
        for field_name in ("sensor_id", "width", "height", "fps", "port"):
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")

        if self.sensor_id < 0:
            raise ValueError("sensor_id must be greater than or equal to zero")

        if min(self.width, self.height, self.fps) <= 0:
            raise ValueError("width, height, and fps must be greater than zero")

        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")

        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host must be a non-empty string")

    def create_application(self) -> Any:
        """Create the configured FastAPI application without starting it.

        Returns:
            FastAPI application that owns this preview's camera lifecycle.
        """
        camera = Camera(
            config=CameraConfig(
                sensor_id=self.sensor_id,
                capture_width=self.width,
                capture_height=self.height,
                output_width=self.width,
                output_height=self.height,
                fps=self.fps,
                max_fps=float(self.fps),
                enable_preview=True,
            )
        )
        return create_app(camera, view_mode="simple")

    def run(self) -> None:
        """Start the browser preview server until it is stopped.

        The API lifespan starts the camera before serving requests and releases
        its resources when the Uvicorn server shuts down.
        """
        uvicorn.run(self.create_application(), host=self.host, port=self.port)


def preview(
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Start a simple browser preview for an IMX camera.

    Args:
        sensor_id: Zero-based CSI sensor identifier.
        width: Capture and output frame width, in pixels.
        height: Capture and output frame height, in pixels.
        fps: Capture and JPEG encoding limit, in frames per second.
        host: Address on which to expose the preview server.
        port: TCP port on which to expose the preview server.
    """
    CameraPreview(
        sensor_id=sensor_id,
        width=width,
        height=height,
        fps=fps,
        host=host,
        port=port,
    ).run()


def create_preview_app(
    camera: Camera,
    *,
    config_path: str | Path | None = None,
    view_mode: ViewMode = "simple",
    view_path: str | Path | None = None,
) -> Any:
    """Create a browser preview application using an existing camera.

    The caller retains camera lifecycle ownership. In particular, this helper
    neither starts nor stops ``camera``, so one capture pipeline can serve
    inference, raw-frame consumers, snapshots, MJPEG preview, diagnostics, and
    other application components concurrently. JPEG preview is enabled on the
    supplied camera without recreating its capture backend.

    Args:
        camera: Existing camera instance shared with the application.
        config_path: Optional path to the API YAML configuration file.
        view_mode: Bundled preview view: ``"simple"`` or ``"advanced"``.
        view_path: Optional custom browser-view template path.

    Returns:
        FastAPI application backed by the supplied camera.

    Raises:
        TypeError: If ``camera`` is not a :class:`Camera` instance.
        ValueError: If view configuration is invalid.
    """
    if not isinstance(camera, Camera):
        raise TypeError("camera must be a Camera instance")

    camera.set_preview_enabled(True)
    return create_app(
        camera,
        config_path=config_path,
        view_mode=view_mode,
        view_path=view_path,
        manage_camera=False,
    )


def serve(
    source: PreviewSource,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    quality: int = 65,
    max_fps: float = 30.0,
) -> None:
    """Serve opaque images from an existing source through a browser preview.

    The caller retains source lifecycle ownership. In particular, passing a
    :class:`Camera` does not open or close its capture pipeline; start it with a
    context manager or application lifecycle before calling this function.

    Args:
        source: Existing latest-frame source, such as a started ``Camera``.
        host: Address on which to expose the HTTP server.
        port: TCP port on which to expose the HTTP server.
        quality: JPEG quality from 0 to 100.
        max_fps: Maximum JPEG encoding rate in frames per second.

    Raises:
        ValueError: If server or JPEG settings are invalid.
    """
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")

    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 1 <= port <= 65535
    ):
        raise ValueError("port must be between 1 and 65535")

    server = PreviewServer(source=source, quality=quality, max_fps=max_fps)
    uvicorn.run(server.create_app(), host=host, port=port)
