"""Configuration loading for NVIDIA Jetson IMX cameras."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml

except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "config.yml"

@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Validated static settings used to create an IMX camera pipeline.

    Attributes:
        sensor_id: Zero-based CSI sensor identifier used by Argus.
        capture_width: Width captured directly from the sensor, in pixels.
        capture_height: Height captured directly from the sensor, in pixels.
        output_width: Width of frames delivered to OpenCV, in pixels.
        output_height: Height of frames delivered to OpenCV, in pixels.
        fps: Camera capture rate, in frames per second.
        flip_method: NVIDIA ``nvvidconv`` flip transformation, from 0 to 7.
        sensor_mode: Optional Argus sensor mode. ``None`` lets Argus choose.
        enable_preview: Whether JPEG preview encoding is enabled.
        quality: JPEG quality from 0 to 100 when preview is enabled.
        max_fps: Optional JPEG encoding limit. When omitted, ``fps`` is used.
    """

    sensor_id: int = 0
    capture_width: int = 1920
    capture_height: int = 1080
    output_width: int = 1280
    output_height: int = 720
    fps: int = 30
    flip_method: int = 0
    sensor_mode: int | None = None
    enable_preview: bool = False
    quality: int = 65
    max_fps: float | None = None

    def __post_init__(self) -> None:
        """Validate static configuration values at construction time."""
        validate_camera_config(self)

    @property
    def preview_fps(self) -> float:
        """float: Resolved maximum rate used by the JPEG preview encoder."""
        return float(self.fps) if self.max_fps is None else float(self.max_fps)

    @classmethod
    def from_profile(cls, name: str) -> CameraConfig:
        """Create a configuration from a curated hardware profile.

        Profiles contain only static camera settings. JPEG preview, runtime
        controls, and other application-level behavior retain the defaults of
        the profile's ``CameraConfig`` and can be overridden in ``Camera``.

        Args:
            name: Profile identifier, for example ``"imx219-1080p"``.

        Returns:
            Validated static configuration for the selected profile.

        Raises:
            TypeError: If ``name`` is not a string.
            ValueError: If the requested profile is unavailable.
        """
        from ..profiles import get_camera_profile

        return get_camera_profile(name).config


def validate_camera_config(config: CameraConfig) -> None:
    """Validate values that are not checked by pipeline construction.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If a configuration value is invalid.
    """
    integer_fields = (
        "quality",
        "sensor_id",
        "capture_width",
        "capture_height",
        "output_width",
        "output_height",
        "fps",
        "flip_method",
    )
    for field_name in integer_fields:
        value = getattr(config, field_name)

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")

    if config.sensor_mode is not None and (
        isinstance(config.sensor_mode, bool) or not isinstance(config.sensor_mode, int)
    ):
        raise ValueError("sensor_mode must be an integer or None")

    if not isinstance(config.enable_preview, bool):
        raise ValueError("enable_preview must be a boolean")

    if config.max_fps is not None and (
        isinstance(config.max_fps, bool)
        or not isinstance(config.max_fps, (int, float))
    ):
        raise ValueError("max_fps must be a number or None")

    if not 0 <= config.quality <= 100:
        raise ValueError("quality must be between 0 and 100")

    if config.max_fps is not None and config.max_fps <= 0:
        raise ValueError("max_fps must be greater than zero")

    if config.sensor_id < 0:
        raise ValueError("sensor_id must be greater than or equal to zero")

    if min(
        config.capture_width,
        config.capture_height,
        config.output_width,
        config.output_height,
        config.fps,
    ) <= 0:
        raise ValueError("frame dimensions and framerate must be greater than zero")

    if not 0 <= config.flip_method <= 7:
        raise ValueError("flip_method must be between 0 and 7")

    if config.sensor_mode is not None and config.sensor_mode < 0:
        raise ValueError("sensor_mode must be greater than or equal to zero")


DEFAULT_CAMERA_CONFIG = CameraConfig()


def _read_config_values(config_data: dict[str, Any]) -> CameraConfig:
    """Convert a parsed YAML mapping into a validated configuration."""
    defaults = DEFAULT_CAMERA_CONFIG
    valid_keys = set(defaults.__dataclass_fields__) | {"capture_fps"}
    unknown_keys = set(config_data) - valid_keys

    if unknown_keys:
        formatted_keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unknown camera configuration key(s): {formatted_keys}")

    if "fps" in config_data and "capture_fps" in config_data:
        raise ValueError("use either fps or legacy capture_fps, not both")

    for key in valid_keys - {"capture_fps"}:
        value = config_data.get(key, getattr(defaults, key))
        default_value = getattr(defaults, key)

        if key == "sensor_mode":
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError("sensor_mode must be an integer or null")

        elif key == "enable_preview":
            if not isinstance(value, bool):
                raise ValueError("enable_preview must be a boolean")

        elif key == "max_fps":
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError("max_fps must be a number or null")

        elif isinstance(default_value, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")

        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")

    config = CameraConfig(
        sensor_id=config_data.get("sensor_id", defaults.sensor_id),
        capture_width=config_data.get("capture_width", defaults.capture_width),
        capture_height=config_data.get("capture_height", defaults.capture_height),
        output_width=config_data.get("output_width", defaults.output_width),
        output_height=config_data.get("output_height", defaults.output_height),
        fps=config_data.get("fps", config_data.get("capture_fps", defaults.fps)),
        flip_method=config_data.get("flip_method", defaults.flip_method),
        sensor_mode=config_data.get("sensor_mode", defaults.sensor_mode),
        enable_preview=config_data.get("enable_preview", defaults.enable_preview),
        quality=config_data.get("quality", defaults.quality),
        max_fps=config_data.get("max_fps", defaults.max_fps),
    )
    return config


def load_camera_config(config_path: str | Path | None = None) -> CameraConfig:
    """Load camera settings from YAML, falling back to built-in defaults.

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
        return DEFAULT_CAMERA_CONFIG

    except OSError as error:
        logger.warning("Could not read camera configuration %s: %s", path, error)
        return DEFAULT_CAMERA_CONFIG

    if yaml is None:
        logger.warning("PyYAML is unavailable; using built-in camera defaults")
        return DEFAULT_CAMERA_CONFIG

    try:
        parsed_config = yaml.safe_load(raw_config)

        if not isinstance(parsed_config, dict):
            raise ValueError("the YAML document must be a mapping")

        config_data = parsed_config.get("camera_config")

        if not isinstance(config_data, dict):
            raise ValueError("camera_config must be a mapping")

        return _read_config_values(config_data)

    except (ValueError, yaml.YAMLError) as error:
        logger.warning("Invalid camera configuration %s: %s", path, error)
        return DEFAULT_CAMERA_CONFIG
