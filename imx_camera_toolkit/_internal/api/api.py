"""FastAPI application exposing IMX camera snapshots and MJPEG streams."""

from __future__ import annotations

import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal, TypeAlias

from fastapi import Body, FastAPI, HTTPException, Query, Security
from fastapi.responses import HTMLResponse, Response, StreamingResponse

try:
    import yaml

except ImportError:
    yaml = None

from imx_camera_toolkit._internal.camera.camera import Camera, CameraConfig
from imx_camera_toolkit._internal.camera.gpu_camera import GpuCamera
from imx_camera_toolkit._internal.camera_control.camera_control import (
    CameraController,
    ProfileNotFoundError,
    UnsupportedControlError,
)
from imx_camera_toolkit._internal.stream.stream import MJPEGStream

from .security import (
    SecurityConfig,
    apply_security_middleware,
    build_authorizer,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yml")
ViewMode: TypeAlias = Literal["simple", "advanced"]
DEFAULT_VIEW_MODE: ViewMode = "simple"
DEFAULT_VIEW_PATHS: dict[ViewMode, Path] = {
    "simple": Path(__file__).parents[3] / "view" / "simple.html",
    "advanced": Path(__file__).parents[3] / "view" / "advanced.html",
}
DEFAULT_VIEW_PATH = DEFAULT_VIEW_PATHS[DEFAULT_VIEW_MODE]
CAMERA_STREAM_PATH = "/api/camera/mjpeg"
CAMERA_STREAM_TEMPLATE = "{{ camera_stream_url }}"


@dataclass(frozen=True)
class APIConfig:
    """Validated settings used to create the FastAPI application.

    Attributes:
        title: Application title displayed in OpenAPI documentation.
        description: Application description displayed in OpenAPI documentation.
        version: Application version displayed in OpenAPI documentation.
        snapshot_timeout: Maximum snapshot wait time, in seconds.
    """

    title: str = "IMX Camera API"
    description: str = "Snapshots and MJPEG streaming for an NVIDIA Jetson CSI camera."
    version: str = "0.7.0"
    snapshot_timeout: float = 2.0


DEFAULT_API_CONFIG = APIConfig()

NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _validate_api_config(config: APIConfig) -> None:
    """Validate values used to create the API application.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If API metadata or the snapshot timeout is invalid.
    """
    for field_name in ("title", "description", "version"):
        value = getattr(config, field_name)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")

    if isinstance(config.snapshot_timeout, bool) or not isinstance(
        config.snapshot_timeout, (int, float)
    ):
        raise ValueError("snapshot_timeout must be a number")

    if not math.isfinite(config.snapshot_timeout) or not (
        0 < config.snapshot_timeout <= 60
    ):
        raise ValueError("snapshot_timeout must be finite and between 0 and 60 seconds")


def _read_config_values(config_data: dict[str, Any]) -> APIConfig:
    """Convert a parsed YAML mapping into a validated API configuration.

    Args:
        config_data: Mapping stored under ``api_config`` in the YAML file.

    Returns:
        Validated API configuration.

    Raises:
        ValueError: If keys are unknown or values have invalid types or ranges.
    """
    valid_keys = set(DEFAULT_API_CONFIG.__dataclass_fields__)
    unknown_keys = set(config_data) - valid_keys

    if unknown_keys:
        formatted_keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unknown API configuration key(s): {formatted_keys}")

    config = APIConfig(
        title=config_data.get("title", DEFAULT_API_CONFIG.title),
        description=config_data.get("description", DEFAULT_API_CONFIG.description),
        version=config_data.get("version", DEFAULT_API_CONFIG.version),
        snapshot_timeout=config_data.get(
            "snapshot_timeout", DEFAULT_API_CONFIG.snapshot_timeout
        ),
    )

    _validate_api_config(config)
    return config


def load_api_config(
    config_path: str | Path | None = None,
    *,
    strict: bool = False,
) -> APIConfig:
    """Load API settings from YAML, falling back to built-in defaults.

    Args:
        config_path: Path to a YAML file. When omitted, uses the ``config.yml``
            located next to this module.
        strict: Raise on missing, unreadable, or invalid configuration instead
            of falling back to development defaults.

    Returns:
        A validated configuration. Built-in defaults are returned when the file
        is missing, cannot be read, is malformed, or contains invalid values.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    try:
        raw_config = path.read_text(encoding="utf-8")

    except FileNotFoundError:
        if strict:
            raise

        return DEFAULT_API_CONFIG

    except OSError as error:
        if strict:
            raise RuntimeError(f"Could not read API configuration {path}") from error

        logger.warning("Could not read API configuration %s: %s", path, error)
        return DEFAULT_API_CONFIG

    if yaml is None:
        if strict:
            raise RuntimeError("PyYAML is required for strict API configuration")

        logger.warning("PyYAML is unavailable; using built-in API defaults")
        return DEFAULT_API_CONFIG

    try:
        parsed_config = yaml.safe_load(raw_config)

        if not isinstance(parsed_config, dict):
            raise ValueError("the YAML document must be a mapping")

        config_data = parsed_config.get("api_config")

        if not isinstance(config_data, dict):
            raise ValueError("api_config must be a mapping")

        return _read_config_values(config_data)

    except (ValueError, yaml.YAMLError) as error:
        if strict:
            raise ValueError(f"Invalid API configuration {path}: {error}") from error
        logger.warning("Invalid API configuration %s: %s", path, error)
        return DEFAULT_API_CONFIG


class _CameraStreamImageParser(HTMLParser):
    """Detect the image element reserved for the MJPEG camera stream."""

    def __init__(self) -> None:
        """Initialize the parser without an identified stream image."""
        super().__init__()
        self.has_stream_image = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Inspect image tags for the required camera stream marker.

        Args:
            tag: Parsed HTML tag name.
            attrs: Attributes associated with the tag.
        """
        if tag != "img":
            return

        attributes = dict(attrs)

        if (
            "data-camera-stream" in attributes
            and attributes.get("src") == CAMERA_STREAM_TEMPLATE
        ):
            self.has_stream_image = True


def _resolve_view_path(
    view_mode: ViewMode,
    view_path: str | Path | None,
) -> Path:
    """Resolve a bundled view mode or an explicit custom template path.

    Args:
        view_mode: Bundled view variant selected when ``view_path`` is omitted.
        view_path: Explicit user-provided template path.

    Returns:
        Resolved path to the selected HTML template.

    Raises:
        ValueError: If ``view_mode`` is not a supported bundled variant.
    """
    if view_path is not None:
        return Path(view_path)

    if view_mode not in DEFAULT_VIEW_PATHS:
        supported_modes = ", ".join(DEFAULT_VIEW_PATHS)
        raise ValueError(
            f"unknown camera view mode {view_mode!r}; use one of: {supported_modes}"
        )

    return DEFAULT_VIEW_PATHS[view_mode]


def load_camera_view(
    view_path: str | Path | None = None,
    *,
    view_mode: ViewMode = DEFAULT_VIEW_MODE,
) -> str:
    """Load a camera view template and insert the MJPEG stream URL.

    The template must include an ``img`` tag with both
    ``data-camera-stream`` and ``src="{{ camera_stream_url }}"``. Other HTML,
    CSS, and JavaScript are left unchanged, so users can fully style the view.

    Args:
        view_path: Path to an HTML template. Overrides ``view_mode`` when set.
        view_mode: Bundled template variant: ``"simple"`` for preview only or
            ``"advanced"`` for preview with runtime camera controls.

    Returns:
        HTML ready to serve to the browser.

    Raises:
        RuntimeError: If the view file cannot be read.
        ValueError: If ``view_mode`` is unsupported or the required camera
            stream image element is absent.
    """
    path = _resolve_view_path(view_mode, view_path)

    try:
        html = path.read_text(encoding="utf-8")

    except OSError as error:
        raise RuntimeError(f"Could not read camera view {path}: {error}") from error

    parser = _CameraStreamImageParser()
    parser.feed(html)
    parser.close()

    if not parser.has_stream_image:
        raise ValueError(
            "camera view must contain "
            '<img data-camera-stream src="{{ camera_stream_url }}">'
        )

    return html.replace(CAMERA_STREAM_TEMPLATE, CAMERA_STREAM_PATH)


def _camera_status(camera: Camera | GpuCamera) -> dict[str, object]:
    """Return JSON-serializable state and capture metrics for a camera.

    Args:
        camera: Camera instance to inspect.

    Returns:
        Health status, frame availability, capture metrics, and the last
        background capture error when one exists.
    """
    diagnostics = camera.stats()
    width, height = camera.frame_resolution
    last_error = str(camera.last_error) if camera.last_error is not None else None

    if diagnostics.running:
        status = "ok"

    elif last_error is not None:
        status = "error"

    else:
        status = "unavailable"

    return {
        "status": status,
        "camera_running": diagnostics.running,
        "frame_available": camera.frame_available,
        "frame_number": camera.frame_number,
        "frames_captured": diagnostics.captured_frames,
        "dropped_frames": diagnostics.dropped_frames,
        "capture_fps": diagnostics.capture_fps,
        "last_frame_timestamp_ns": diagnostics.last_frame_timestamp_ns,
        "last_capture_timestamp_ns": diagnostics.last_capture_timestamp_ns,
        "active_backend": camera.active_backend,
        "frame_format": camera.frame_format.value,
        "frame_memory_type": camera.memory_type.value,
        "frame_resolution": {"width": width, "height": height},
        "consumer_dropped_frames": dict(diagnostics.consumer_dropped_frames),
        "stage_latency_ns": {
            stage: {
                "samples": metrics.samples,
                "last": metrics.last_duration_ns,
                "mean": metrics.mean_duration_ns,
                "max": metrics.max_duration_ns,
            }
            for stage, metrics in (
                ("transfer", diagnostics.pipeline.transfer),
                ("inference", diagnostics.pipeline.inference),
                ("encoder", diagnostics.pipeline.encoder),
                ("end_to_end", diagnostics.pipeline.end_to_end),
            )
        },
        "frames_encoded": camera.frames_encoded,
        "last_frame_time": camera.last_frame_time,
        "last_error": last_error,
        "recovery_attempts": camera.recovery_attempts,
        "recoveries": diagnostics.recovery_count,
        "consecutive_failures": diagnostics.consecutive_failures,
        "last_recovery_error": (
            str(camera.last_recovery_error)
            if camera.last_recovery_error is not None
            else None
        ),
    }


def create_app(
    camera: Camera | GpuCamera | None = None,
    *,
    config_path: str | Path | None = None,
    view_mode: ViewMode = DEFAULT_VIEW_MODE,
    view_path: str | Path | None = None,
    manage_camera: bool = True,
    security_config: SecurityConfig | None = None,
) -> Any:
    """Create a FastAPI application backed by one shared camera instance.

    The camera is deliberately shared by the snapshot and MJPEG endpoints
    instead of being created once per HTTP request. By default, the FastAPI
    lifespan owns startup and shutdown. Applications that already own a camera
    can set ``manage_camera=False``.

    Args:
        camera: CPU or GPU camera to expose. When omitted, creates a default
            CPU ``Camera`` for backward compatibility.
        config_path: Optional path to a YAML API configuration file.
        view_mode: Bundled camera view: ``"simple"`` for preview only or
            ``"advanced"`` for preview with runtime control panel.
        view_path: Optional path to the browser camera view template.
        manage_camera: Whether the API lifespan starts and stops ``camera``.
        security_config: Authentication and field-deployment policy.

    Returns:
        Configured FastAPI application.

    Raises:
        RuntimeError: If FastAPI is unavailable in the current environment.
    """
    if not isinstance(manage_camera, bool):
        raise ValueError("manage_camera must be a boolean")

    shared_camera = (
        camera if camera is not None else Camera(CameraConfig(enable_preview=True))
    )
    camera_controller = CameraController(
        runtime_handler=lambda update: shared_camera.apply_argus_properties(
            update.source_properties,
            restart_required=update.restart_required,
        )
    )
    resolved_security = security_config or SecurityConfig()
    config = load_api_config(config_path, strict=resolved_security.field_mode)
    resolved_view_path = _resolve_view_path(view_mode, view_path)
    authorize = build_authorizer(resolved_security)

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        """Optionally manage the shared camera for this application lifespan."""
        if manage_camera:
            shared_camera.start()

        try:
            yield

        finally:
            if manage_camera:
                shared_camera.stop()

    application = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        lifespan=lifespan,
        docs_url="/docs" if resolved_security.docs_enabled else None,
        redoc_url="/redoc" if resolved_security.docs_enabled else None,
        openapi_url="/openapi.json" if resolved_security.docs_enabled else None,
    )
    apply_security_middleware(application, resolved_security)
    application.state.config = config
    application.state.security_config = resolved_security
    application.state.view_mode = view_mode
    application.state.view_path = resolved_view_path
    application.state.manage_camera = manage_camera
    application.state.camera_controller = camera_controller

    @application.get(
        "/",
        response_class=HTMLResponse,
        include_in_schema=False,
        dependencies=[Security(authorize, scopes=["stream:read"])],
    )
    def index() -> Any:
        """Return the customizable browser camera view.

        Returns:
            HTML containing the live MJPEG camera image.

        Raises:
            HTTPException: If the view file cannot be loaded or has no valid
                camera stream image tag.
        """
        try:
            return HTMLResponse(
                content=load_camera_view(resolved_view_path),
                headers=NO_CACHE_HEADERS,
            )

        except (RuntimeError, ValueError) as error:
            logger.error("Could not serve camera view: %s", error)
            raise HTTPException(status_code=500, detail=str(error)) from error

    @application.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        """Return a non-diagnostic process liveness response."""
        return {"status": "ok"}

    @application.get(
        "/debug/health",
        dependencies=[Security(authorize, scopes=["admin"])],
    )
    @application.get(
        "/api/health",
        deprecated=True,
        dependencies=[Security(authorize, scopes=["admin"])],
    )
    def health() -> dict[str, object]:
        """Return camera health and capture metrics.

        Returns:
            JSON-serializable state for the shared camera.
        """
        return _camera_status(shared_camera)

    @application.get(
        "/api/camera/control",
        dependencies=[Security(authorize, scopes=["camera:read"])],
    )
    def camera_control_state() -> dict[str, object]:
        """Return runtime camera-control settings and capabilities.

        Returns:
            JSON-ready control state, available source properties, and profiles.
        """
        state = camera_controller.get_runtime_state()
        state["capture_fps"] = shared_camera.config.fps
        state["software_hdr"] = shared_camera.software_hdr_state
        return state

    @application.patch(
        "/api/camera/control",
        dependencies=[Security(authorize, scopes=["camera:control"])],
    )
    def update_camera_control(values: dict[str, Any] = Body(...)) -> dict[str, object]:
        """Apply a validated camera-control update at runtime.

        Exposure, gain, white balance, and denoising are applied to the live
        Argus source. Sensor-mode and HDR changes restart capture because they
        alter the sensor operating mode. Keys omitted from the request preserve
        their current values; ``null`` restores automatic exposure, gain,
        denoise strength, or sensor-mode selection where applicable.

        Args:
            values: Partial ``CameraController.update`` settings mapping.

        Returns:
            The committed update, including generated Argus properties.

        Raises:
            HTTPException: If a key, value, capability, or pipeline update is
                invalid.
        """
        valid_keys = set(camera_controller.settings.__dataclass_fields__)
        unknown_keys = set(values) - valid_keys

        if unknown_keys:
            formatted_keys = ", ".join(sorted(unknown_keys))
            raise HTTPException(
                status_code=422,
                detail=f"unknown camera-control key(s): {formatted_keys}",
            )

        try:
            return camera_controller.update(**values).as_dict()

        except UnsupportedControlError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        except RuntimeError as error:
            logger.exception("Could not apply camera-control update")
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.get(
        "/api/camera/software-hdr",
        dependencies=[Security(authorize, scopes=["camera:read"])],
    )
    def software_hdr_state() -> dict[str, object]:
        """Return the active Jetson-side exposure-fusion HDR state.

        Returns:
            Enabled state, configured base exposure, and resolved brackets.
        """
        return shared_camera.software_hdr_state

    @application.put(
        "/api/camera/software-hdr",
        dependencies=[Security(authorize, scopes=["camera:control"])],
    )
    def configure_software_hdr(values: dict[str, Any] = Body(...)) -> dict[str, object]:
        """Configure three-exposure HDR fusion for sensors without native HDR.

        Args:
            values: ``enabled`` plus optional ``base_exposure_us`` and
                ``settle_frames`` values.

        Returns:
            The committed software HDR state.

        Raises:
            HTTPException: If the request is invalid or sensor controls fail.
        """
        valid_keys = {"enabled", "base_exposure_us", "settle_frames"}
        unknown_keys = set(values) - valid_keys

        if unknown_keys:
            formatted_keys = ", ".join(sorted(unknown_keys))
            raise HTTPException(
                status_code=422,
                detail=f"unknown software HDR key(s): {formatted_keys}",
            )

        if "enabled" not in values:
            raise HTTPException(status_code=422, detail="enabled is required")

        try:
            return shared_camera.configure_software_hdr(
                enabled=values["enabled"],
                base_exposure_us=values.get("base_exposure_us"),
                settle_frames=values.get("settle_frames"),
            )

        except (ValueError, RuntimeError) as error:
            logger.warning("Could not configure software HDR: %s", error)
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.get(
        "/api/camera/control/profiles",
        dependencies=[Security(authorize, scopes=["camera:read"])],
    )
    def list_camera_control_profiles() -> dict[str, list[str]]:
        """Return names of available in-memory camera-control profiles.

        Returns:
            Profile names ordered lexicographically.
        """
        return {
            "profiles": [profile.name for profile in camera_controller.list_profiles()]
        }

    @application.put(
        "/api/camera/control/profiles/{name}",
        dependencies=[Security(authorize, scopes=["profiles:write"])],
    )
    def save_camera_control_profile(name: str) -> dict[str, object]:
        """Store the current runtime settings as a named profile.

        Args:
            name: Profile identifier.

        Returns:
            Name and settings held by the new profile.

        Raises:
            HTTPException: If the profile name is invalid.
        """
        try:
            profile = camera_controller.save_profile(name)

        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return {
            "name": profile.name,
            "settings": camera_controller.get_runtime_state()["settings"],
        }

    @application.post(
        "/api/camera/control/profiles/{name}/apply",
        dependencies=[
            Security(
                authorize,
                scopes=["profiles:write", "camera:control"],
            )
        ],
    )
    def apply_camera_control_profile(name: str) -> dict[str, object]:
        """Apply one named profile to the active camera.

        Args:
            name: Profile identifier.

        Returns:
            The committed runtime update.

        Raises:
            HTTPException: If the profile is absent or the update cannot apply.
        """
        try:
            return camera_controller.apply_profile(name).as_dict()

        except ProfileNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail=f"unknown profile: {name}",
            ) from error

        except RuntimeError as error:
            logger.exception("Could not apply camera-control profile %s", name)
            raise HTTPException(status_code=503, detail=str(error)) from error

    @application.delete(
        "/api/camera/control/profiles/{name}",
        status_code=204,
        dependencies=[Security(authorize, scopes=["profiles:write"])],
    )
    def delete_camera_control_profile(name: str) -> None:
        """Remove one in-memory camera-control profile.

        Args:
            name: Profile identifier.

        Raises:
            HTTPException: If the profile is absent.
        """
        try:
            camera_controller.delete_profile(name)

        except ProfileNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail=f"unknown profile: {name}",
            ) from error

    @application.get(
        "/api/camera/snapshot",
        dependencies=[Security(authorize, scopes=["stream:read"])],
    )
    def camera_snapshot(
        after: int = Query(default=-1, ge=-1, le=(1 << 63) - 1),
    ) -> Any:
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
            timeout=config.snapshot_timeout,
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

    @application.get(
        "/api/camera/mjpeg",
        dependencies=[Security(authorize, scopes=["stream:read"])],
    )
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
