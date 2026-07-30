"""Compatibility facade for NVIDIA Argus camera-control primitives.

The implementation is split into focused modules. Existing applications may
continue importing all public symbols from this module.
"""

from .controls import build_argus_control_properties
from .capabilities import (
    ARGUS_CONTROL_PROPERTIES,
    DEFAULT_CAPABILITIES,
    discover_argus_capabilities,
)
from .config import (
    CameraControlConfig,
    DEFAULT_CAMERA_CONTROL_CONFIG,
    DEFAULT_CONFIG_PATH,
    load_camera_control_config,
)
from .runtime import CameraController, UNSET
from .models import (
    CameraCapabilities,
    CameraControlUpdate,
    CameraProfile,
    CameraSettings,
    DenoiseMode,
    ProfileNotFoundError,
    RuntimeHandler,
    SensorMode,
    UnsupportedControlError,
    WhiteBalanceMode,
)

__all__ = [
    "ARGUS_CONTROL_PROPERTIES",
    "CameraCapabilities",
    "CameraControlConfig",
    "CameraController",
    "CameraControlUpdate",
    "CameraProfile",
    "CameraSettings",
    "DEFAULT_CAMERA_CONTROL_CONFIG",
    "DEFAULT_CAPABILITIES",
    "DEFAULT_CONFIG_PATH",
    "DenoiseMode",
    "ProfileNotFoundError",
    "RuntimeHandler",
    "SensorMode",
    "UNSET",
    "UnsupportedControlError",
    "WhiteBalanceMode",
    "build_argus_control_properties",
    "discover_argus_capabilities",
    "load_camera_control_config",
]
