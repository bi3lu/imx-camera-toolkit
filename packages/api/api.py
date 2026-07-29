"""FastAPI application exposing IMX camera snapshots and MJPEG streams."""

from __future__ import annotations

import logging

from contextlib import asynccontextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from packages.camera.camera import Camera
from packages.camera_control.camera_control import (
    CameraController,
    ProfileNotFoundError,
    UnsupportedControlError,
)
from packages.stream.stream import MJPEGStream

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yml")
DEFAULT_VIEW_PATH = Path(__file__).parents[2] / "view" / "index.html"
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
    version: str = "0.1.0"
    snapshot_timeout: float = 2.0


DEFAULT_API_CONFIG = APIConfig()

try:
    from fastapi import Body, FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, Response, StreamingResponse

except ImportError:
    FastAPI: Any | None = None
    Body: Any | None = None
    HTTPException: Any | None = None
    Query: Any | None = None
    HTMLResponse: Any | None = None
    Response: Any | None = None
    StreamingResponse: Any | None = None

try:
    import yaml

except ImportError:
    yaml: Any | None = None

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

    if config.snapshot_timeout <= 0:
        raise ValueError("snapshot_timeout must be greater than zero")


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


def load_api_config(config_path: str | Path | None = None) -> APIConfig:
    """Load API settings from YAML, falling back to built-in defaults.

    Args:
        config_path: Path to a YAML file. When omitted, uses the ``config.yml``
            located next to this module.

    Returns:
        A validated configuration. Built-in defaults are returned when the file
        is missing, cannot be read, is malformed, or contains invalid values.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    try:
        raw_config = path.read_text(encoding="utf-8")

    except FileNotFoundError:
        return DEFAULT_API_CONFIG

    except OSError as error:
        logger.warning("Could not read API configuration %s: %s", path, error)
        return DEFAULT_API_CONFIG

    if yaml is None:
        logger.warning("PyYAML is unavailable; using built-in API configuration defaults")
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


def load_camera_view(view_path: str | Path | None = None) -> str:
    """Load a camera view template and insert the MJPEG stream URL.

    The template must include an ``img`` tag with both
    ``data-camera-stream`` and ``src="{{ camera_stream_url }}"``. Other HTML,
    CSS, and JavaScript are left unchanged, so users can fully style the view.

    Args:
        view_path: Path to an HTML template. When omitted, uses
            ``view/index.html`` at the project root.

    Returns:
        HTML ready to serve to the browser.

    Raises:
        RuntimeError: If the view file cannot be read.
        ValueError: If the required camera stream image element is absent.
    """
    path = Path(view_path) if view_path is not None else DEFAULT_VIEW_PATH

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


def create_app(
    camera: Camera | None = None,
    *,
    config_path: str | Path | None = None,
    view_path: str | Path | None = None,
) -> Any:
    """Create a FastAPI application backed by one shared camera instance.

    The camera is started during the FastAPI lifespan startup event and stopped
    during shutdown. It is deliberately shared by the snapshot and MJPEG
    endpoints instead of being created once per HTTP request.

    Args:
        camera: Camera to expose. When omitted, creates a default ``Camera``.
        config_path: Optional path to a YAML API configuration file.
        view_path: Optional path to the browser camera view template.

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
    camera_controller = CameraController(
        runtime_handler=lambda update: shared_camera.apply_argus_properties(
            update.source_properties,
            restart_required=update.restart_required,
        )
    )
    config = load_api_config(config_path)
    resolved_view_path = Path(view_path) if view_path is not None else DEFAULT_VIEW_PATH

    @asynccontextmanager
    async def lifespan(_: Any):
        """Start the shared camera for the lifetime of the API application."""
        shared_camera.start()

        try:
            yield

        finally:
            shared_camera.stop()

    application = FastAPI(
        title=config.title,
        description=config.description,
        version=config.version,
        lifespan=lifespan,
    )
    application.state.config = config
    application.state.view_path = resolved_view_path
    application.state.camera_controller = camera_controller

    @application.get("/", response_class=HTMLResponse, include_in_schema=False)
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

    @application.get("/api/health")
    def health() -> dict[str, object]:
        """Return camera health and capture metrics.

        Returns:
            JSON-serializable state for the shared camera.
        """
        return _camera_status(shared_camera)

    @application.get("/api/camera/control")
    def camera_control_state() -> dict[str, object]:
        """Return runtime camera-control settings and capabilities.

        Returns:
            JSON-ready control state, available source properties, and profiles.
        """
        state = camera_controller.get_runtime_state()
        state["capture_fps"] = shared_camera.config.capture_fps
        state["software_hdr"] = shared_camera.software_hdr_state
        return state

    @application.patch("/api/camera/control")
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

    @application.get("/api/camera/software-hdr")
    def software_hdr_state() -> dict[str, object]:
        """Return the active Jetson-side exposure-fusion HDR state.

        Returns:
            Enabled state, configured base exposure, and resolved brackets.
        """
        return shared_camera.software_hdr_state

    @application.put("/api/camera/software-hdr")
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

    @application.get("/api/camera/control/profiles")
    def list_camera_control_profiles() -> dict[str, list[str]]:
        """Return names of available in-memory camera-control profiles.

        Returns:
            Profile names ordered lexicographically.
        """
        return {
            "profiles": [profile.name for profile in camera_controller.list_profiles()]
        }

    @application.put("/api/camera/control/profiles/{name}")
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

    @application.post("/api/camera/control/profiles/{name}/apply")
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

    @application.delete("/api/camera/control/profiles/{name}", status_code=204)
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
