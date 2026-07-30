"""YAML configuration loading for camera-control defaults."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..controls import build_argus_control_properties
from ..capabilities import DEFAULT_CAPABILITIES
from ..models import (
    CameraCapabilities,
    CameraSettings,
    DenoiseMode,
    SensorMode,
    WhiteBalanceMode,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "config.yml"

try:
    import yaml

except ImportError:
    yaml: Any | None = None


@dataclass(frozen=True)
class CameraControlConfig:
    """Settings loaded from the camera-control YAML configuration.

    Args:
        capabilities: Argus properties and sensor modes declared for the
            active camera.
        initial_settings: Control state used when a controller is created.
    """

    capabilities: CameraCapabilities = DEFAULT_CAPABILITIES
    initial_settings: CameraSettings = CameraSettings()


DEFAULT_CAMERA_CONTROL_CONFIG = CameraControlConfig()


def _parse_sensor_mode(config_data: object) -> SensorMode:
    """Convert one YAML sensor-mode mapping to a validated descriptor."""
    if not isinstance(config_data, dict):
        raise ValueError("each sensor mode must be a mapping")

    valid_keys = {"index", "width", "height", "max_fps", "hdr", "name"}
    unknown_keys = set(config_data) - valid_keys

    if unknown_keys:
        formatted_keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unknown sensor-mode key(s): {formatted_keys}")

    if "index" not in config_data:
        raise ValueError("each sensor mode requires an index")

    if "hdr" in config_data and not isinstance(config_data["hdr"], bool):
        raise ValueError("sensor mode hdr must be a boolean")

    return SensorMode(
        index=config_data["index"],
        width=config_data.get("width"),
        height=config_data.get("height"),
        max_fps=config_data.get("max_fps"),
        hdr=config_data.get("hdr", False),
        name=config_data.get("name"),
    )


def _parse_initial_settings(config_data: object) -> CameraSettings:
    """Convert a YAML initial-settings mapping to validated camera settings."""
    if not isinstance(config_data, dict):
        raise ValueError("initial_settings must be a mapping")

    valid_keys = set(CameraSettings.__dataclass_fields__)
    unknown_keys = set(config_data) - valid_keys

    if unknown_keys:
        formatted_keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unknown initial-settings key(s): {formatted_keys}")

    defaults = CameraSettings()
    awb_mode = config_data.get("awb_mode", defaults.awb_mode)
    denoise_mode = config_data.get("denoise_mode", defaults.denoise_mode)

    if isinstance(awb_mode, str):
        try:
            awb_mode = WhiteBalanceMode(awb_mode)

        except ValueError as error:
            raise ValueError(f"invalid awb_mode: {awb_mode}") from error

    if isinstance(denoise_mode, str):
        try:
            denoise_mode = DenoiseMode(denoise_mode)

        except ValueError as error:
            raise ValueError(f"invalid denoise_mode: {denoise_mode}") from error

    return CameraSettings(
        exposure_us=config_data.get("exposure_us", defaults.exposure_us),
        gain=config_data.get("gain", defaults.gain),
        awb_mode=awb_mode,
        awb_locked=config_data.get("awb_locked", defaults.awb_locked),
        denoise_mode=denoise_mode,
        denoise_strength=config_data.get(
            "denoise_strength", defaults.denoise_strength
        ),
        sensor_mode=config_data.get("sensor_mode", defaults.sensor_mode),
        hdr_enabled=config_data.get("hdr_enabled", defaults.hdr_enabled),
    )


def _read_config_values(config_data: dict[str, Any]) -> CameraControlConfig:
    """Convert a parsed YAML mapping to a validated control configuration."""
    valid_keys = {"model", "source_properties", "sensor_modes", "initial_settings"}
    unknown_keys = set(config_data) - valid_keys

    if unknown_keys:
        formatted_keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unknown camera-control configuration key(s): {formatted_keys}")

    defaults = DEFAULT_CAMERA_CONTROL_CONFIG
    model = config_data.get("model", defaults.capabilities.model)
    source_properties = config_data.get(
        "source_properties", sorted(defaults.capabilities.source_properties)
    )
    sensor_modes_data = config_data.get("sensor_modes", [])
    initial_settings_data = config_data.get("initial_settings", {})

    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be a non-empty string or null")

    if not isinstance(source_properties, list) or not all(
        isinstance(property_name, str) and property_name.strip()
        for property_name in source_properties
    ):
        raise ValueError("source_properties must be a list of non-empty strings")

    if not isinstance(sensor_modes_data, list):
        raise ValueError("sensor_modes must be a list")

    capabilities = CameraCapabilities(
        source_properties=frozenset(source_properties),
        sensor_modes=tuple(_parse_sensor_mode(mode) for mode in sensor_modes_data),
        model=model,
    )
    initial_settings = _parse_initial_settings(initial_settings_data)
    build_argus_control_properties(initial_settings, capabilities)
    return CameraControlConfig(capabilities, initial_settings)


def load_camera_control_config(
    config_path: str | Path | None = None,
) -> CameraControlConfig:
    """Load camera-control settings from YAML with safe built-in fallbacks.

    Args:
        config_path: YAML configuration path. When omitted, uses the
            ``config.yml`` file next to this module.

    Returns:
        A validated configuration, or built-in defaults when the file is
        missing, unreadable, malformed, or invalid.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    try:
        raw_config = path.read_text(encoding="utf-8")

    except FileNotFoundError:
        return DEFAULT_CAMERA_CONTROL_CONFIG

    except OSError as error:
        logger.warning("Could not read camera-control configuration %s: %s", path, error)
        return DEFAULT_CAMERA_CONTROL_CONFIG

    if yaml is None:
        logger.warning(
            "PyYAML is unavailable; using built-in camera-control configuration defaults"
        )
        return DEFAULT_CAMERA_CONTROL_CONFIG

    try:
        parsed_config = yaml.safe_load(raw_config)

        if not isinstance(parsed_config, dict):
            raise ValueError("the YAML document must be a mapping")

        config_data = parsed_config.get("camera_control_config")

        if not isinstance(config_data, dict):
            raise ValueError("camera_control_config must be a mapping")

        return _read_config_values(config_data)

    except (ValueError, yaml.YAMLError) as error:
        logger.warning("Invalid camera-control configuration %s: %s", path, error)
        return DEFAULT_CAMERA_CONTROL_CONFIG
