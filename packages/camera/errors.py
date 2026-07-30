"""Stable exceptions raised by the public camera API."""


class CameraError(RuntimeError):
    """Base class for camera capture, lifecycle, and runtime failures."""


class CameraDependencyError(CameraError):
    """Raised when the local JetPack camera runtime is unavailable."""


class CameraOpenError(CameraError):
    """Raised when a capture backend cannot open the configured camera."""


class CameraReadError(CameraError):
    """Raised when a raw camera frame cannot be read safely."""


class CameraTimeoutError(CameraReadError):
    """Raised by APIs that require a frame before a configured deadline."""


class CameraConfigurationError(CameraError, ValueError):
    """Raised when a camera configuration or API option is invalid."""


class CameraRecoveryError(CameraError):
    """Raised when the camera cannot recover from a capture failure."""
