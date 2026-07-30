"""Convenient browser-preview facade for a single IMX camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import uvicorn

from .api import create_app
from .camera import Camera


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
            sensor_id=self.sensor_id,
            capture_width=self.width,
            capture_height=self.height,
            output_width=self.width,
            output_height=self.height,
            capture_fps=self.fps,
            max_fps=float(self.fps),
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
