"""Stable runtime camera-control namespace.

The aliases retain the concise names used by application-facing examples while
delegating to the existing NVIDIA Argus control implementation.
"""

from imx_camera_toolkit.camera_control import CameraController, CameraSettings

CameraControls = CameraController
"""Alias for the runtime controller of one camera's settings."""

ExposureConfig = CameraSettings
"""Alias for the immutable exposure and related camera settings model."""

__all__ = ["CameraControls", "ExposureConfig"]
