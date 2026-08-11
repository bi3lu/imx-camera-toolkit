"""Convenient browser-preview facade for a single IMX camera."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import uvicorn

from imx_camera_toolkit._internal.api.security import SecurityConfig
from imx_camera_toolkit._internal.preview import PreviewServer, PreviewSource

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
    host: str = "127.0.0.1"
    port: int = 8000
    allow_remote: bool = False
    field_mode: bool = False
    token_file: Path | None = None
    allowed_hosts: tuple[str, ...] = ()
    behind_tls_proxy: bool = False
    ssl_certfile: Path | None = None
    ssl_keyfile: Path | None = None

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

        for name in ("allow_remote", "field_mode", "behind_tls_proxy"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

        remote = not _is_loopback_host(self.host)
        if remote and not (self.allow_remote or self.field_mode):
            raise ValueError(
                "remote bind requires allow_remote=True or field_mode=True"
            )

        if (self.ssl_certfile is None) != (self.ssl_keyfile is None):
            raise ValueError("ssl_certfile and ssl_keyfile must be provided together")

        if self.field_mode and self.token_file is None:
            raise ValueError("field mode requires a token_file")

        if (
            self.field_mode
            and remote
            and not (self.behind_tls_proxy or self.ssl_certfile is not None)
        ):
            raise ValueError("remote field mode requires TLS or behind_tls_proxy=True")

        if any(not host.strip() for host in self.allowed_hosts):
            raise ValueError("allowed_hosts must contain non-empty host names")
        if self.field_mode and "*" in self.allowed_hosts:
            raise ValueError("field mode does not allow a wildcard host")

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
        return create_app(
            camera,
            view_mode="simple",
            security_config=self._security_config(),
        )

    def _security_config(self) -> SecurityConfig:
        """Resolve the fail-closed field policy before camera startup."""
        if not self.field_mode and self.token_file is None:
            return SecurityConfig()

        allowed_hosts = self.allowed_hosts or _default_allowed_hosts(self.host)
        require_https = self.field_mode and not _is_loopback_host(self.host)

        if self.token_file is not None:
            return SecurityConfig.from_token_file(
                self.token_file,
                field_mode=self.field_mode,
                allowed_hosts=allowed_hosts,
                require_https=require_https,
            )

        return SecurityConfig(allowed_hosts=allowed_hosts)

    def run(self) -> None:
        """Start the browser preview server until it is stopped.

        The API lifespan starts the camera before serving requests and releases
        its resources when the Uvicorn server shuts down.
        """
        application = self.create_application()
        if self.ssl_certfile is not None and self.ssl_keyfile is not None:
            uvicorn.run(
                application,
                host=self.host,
                port=self.port,
                ssl_certfile=self.ssl_certfile,
                ssl_keyfile=self.ssl_keyfile,
            )
        else:
            uvicorn.run(application, host=self.host, port=self.port)


def _is_loopback_host(host: str) -> bool:
    """Return whether a bind address is restricted to this device."""
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False


def _default_allowed_hosts(host: str) -> tuple[str, ...]:
    """Derive a conservative Host allowlist from a concrete bind address."""
    normalized = host.strip()
    if normalized in {"0.0.0.0", "::", "[::]"}:  # noqa: S104
        raise ValueError(
            "wildcard field bind requires at least one allowed_hosts entry"
        )
    return (normalized, "localhost", "127.0.0.1", "[::1]")


def preview(
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    host: str = "127.0.0.1",
    port: int = 8000,
    allow_remote: bool = False,
    field_mode: bool = False,
    token_file: str | Path | None = None,
    allowed_hosts: tuple[str, ...] = (),
    behind_tls_proxy: bool = False,
    ssl_certfile: str | Path | None = None,
    ssl_keyfile: str | Path | None = None,
) -> None:
    """Start a simple browser preview for an IMX camera.

    Args:
        sensor_id: Zero-based CSI sensor identifier.
        width: Capture and output frame width, in pixels.
        height: Capture and output frame height, in pixels.
        fps: Capture and JPEG encoding limit, in frames per second.
        host: Address on which to expose the preview server.
        port: TCP port on which to expose the preview server.
        allow_remote: Explicitly allow a non-loopback bind in development mode.
        field_mode: Enable authentication and HTTP hardening.
        token_file: Protected JSON file containing hashed bearer-token grants.
        allowed_hosts: Accepted HTTP Host header values in field mode.
        behind_tls_proxy: Trust that a local reverse proxy terminates TLS.
        ssl_certfile: TLS certificate for direct Uvicorn termination.
        ssl_keyfile: TLS private key for direct Uvicorn termination.
    """
    CameraPreview(
        sensor_id=sensor_id,
        width=width,
        height=height,
        fps=fps,
        host=host,
        port=port,
        allow_remote=allow_remote,
        field_mode=field_mode,
        token_file=None if token_file is None else Path(token_file),
        allowed_hosts=allowed_hosts,
        behind_tls_proxy=behind_tls_proxy,
        ssl_certfile=None if ssl_certfile is None else Path(ssl_certfile),
        ssl_keyfile=None if ssl_keyfile is None else Path(ssl_keyfile),
    ).run()


def create_preview_app(
    camera: Camera,
    *,
    config_path: str | Path | None = None,
    view_mode: ViewMode = "simple",
    view_path: str | Path | None = None,
    security_config: SecurityConfig | None = None,
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
        security_config: Optional authentication and deployment policy.

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
        security_config=security_config,
    )


def serve(
    source: PreviewSource,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    quality: int = 65,
    max_fps: float = 30.0,
    allow_remote: bool = False,
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
        allow_remote: Explicitly permit binding outside loopback.

    Raises:
        ValueError: If server or JPEG settings are invalid.
    """
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a non-empty string")

    if not _is_loopback_host(host) and not allow_remote:
        raise ValueError("remote bind requires allow_remote=True")

    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    server = PreviewServer(source=source, quality=quality, max_fps=max_fps)
    uvicorn.run(server.create_app(), host=host, port=port)
