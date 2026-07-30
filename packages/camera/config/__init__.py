"""Camera configuration models and YAML loading."""

from .loader import (
    CameraConfig,
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_CONFIG_PATH,
    load_camera_config,
    validate_camera_config,
)

__all__ = [
    "CameraConfig",
    "DEFAULT_CAMERA_CONFIG",
    "DEFAULT_CONFIG_PATH",
    "load_camera_config",
    "validate_camera_config",
]
