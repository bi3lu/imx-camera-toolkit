"""Immutable data models for NVIDIA Argus camera controls."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite


class DenoiseMode(str, Enum):
    """Temporal noise-reduction modes provided by ``nvarguscamerasrc``."""

    OFF = "off"
    FAST = "fast"
    HIGH_QUALITY = "high_quality"


class WhiteBalanceMode(str, Enum):
    """White-balance modes provided by NVIDIA Argus."""

    OFF = "off"
    AUTO = "auto"
    INCANDESCENT = "incandescent"
    FLUORESCENT = "fluorescent"
    WARM_FLUORESCENT = "warm-fluorescent"
    DAYLIGHT = "daylight"
    CLOUDY_DAYLIGHT = "cloudy-daylight"
    TWILIGHT = "twilight"
    SHADE = "shade"
    MANUAL = "manual"


class UnsupportedControlError(ValueError):
    """Raised when a requested control is unavailable on the active sensor."""


class ProfileNotFoundError(KeyError):
    """Raised when a requested in-memory camera profile does not exist."""


@dataclass(frozen=True)
class SensorMode:
    """One sensor mode made available by a sensor driver.

    Args:
        index: Argus sensor-mode index passed to ``nvarguscamerasrc``.
        width: Native frame width in pixels, when known.
        height: Native frame height in pixels, when known.
        max_fps: Maximum frame rate, when known.
        hdr: Whether this mode is an HDR or WDR sensor mode.
        name: Optional human-readable identifier.
    """

    index: int
    width: int | None = None
    height: int | None = None
    max_fps: float | None = None
    hdr: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate a sensor-mode descriptor."""
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise ValueError("sensor mode index must be an integer")

        if not 0 <= self.index <= 255:
            raise ValueError("sensor mode index must be between 0 and 255")

        for field_name in ("width", "height"):
            value = getattr(self, field_name)

            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"sensor mode {field_name} must be a positive integer")

        if self.max_fps is not None and (
            isinstance(self.max_fps, bool)
            or not isinstance(self.max_fps, (int, float))
            or not isfinite(self.max_fps)
            or not 0 < self.max_fps <= 1_000
        ):
            raise ValueError("sensor mode max_fps must be a positive number")

        if not isinstance(self.hdr, bool):
            raise ValueError("sensor mode hdr must be a boolean")

        if self.name is not None and not self.name.strip():
            raise ValueError("sensor mode name must not be empty")


@dataclass(frozen=True)
class CameraCapabilities:
    """Controls and sensor modes known to be supported by a camera.

    Args:
        source_properties: GStreamer properties exposed by the Argus source.
        sensor_modes: Known sensor modes for the selected sensor.
        model: Optional sensor model, for example ``"IMX219"``.
    """

    source_properties: frozenset[str]
    sensor_modes: tuple[SensorMode, ...] = ()
    model: str | None = None

    def __post_init__(self) -> None:
        """Normalize and validate capability metadata."""
        properties = frozenset(
            property_name.strip().lower()
            for property_name in self.source_properties
            if property_name.strip()
        )
        object.__setattr__(self, "source_properties", properties)
        indices = [mode.index for mode in self.sensor_modes]

        if len(indices) != len(set(indices)):
            raise ValueError("sensor mode indices must be unique")

        if self.model is not None and not self.model.strip():
            raise ValueError("camera model must not be empty")

    @property
    def hdr_supported(self) -> bool:
        """Return whether declared sensor modes include an HDR mode."""
        return any(mode.hdr for mode in self.sensor_modes)

    @property
    def sensor_mode_metadata_available(self) -> bool:
        """Return whether complete mode descriptors were supplied."""
        return bool(self.sensor_modes)

    def supports(self, property_name: str) -> bool:
        """Return whether an Argus source property is known to be available."""
        return property_name.lower() in self.source_properties

    def get_sensor_mode(self, index: int | None) -> SensorMode | None:
        """Return known metadata for one mode, or ``None`` when unavailable."""
        return next((mode for mode in self.sensor_modes if mode.index == index), None)


@dataclass(frozen=True)
class CameraSettings:
    """Validated desired state for one NVIDIA Argus camera.

    Exposure is expressed in microseconds for the public API. A fixed exposure
    or gain is represented as a one-value Argus range.

    Args:
        exposure_us: Fixed exposure duration in microseconds, or ``None``.
        gain: Fixed analog gain, or ``None`` for automatic gain.
        awb_mode: Desired white-balance mode.
        awb_locked: Whether auto white balance is locked.
        denoise_mode: Temporal noise-reduction quality mode.
        denoise_strength: Denoising strength from 0.0 to 1.0, or ``None``.
        sensor_mode: Sensor mode index, or ``None`` for Argus auto-selection.
        hdr_enabled: Whether an HDR sensor mode is requested.
    """

    exposure_us: int | None = None
    gain: float | None = None
    awb_mode: WhiteBalanceMode = WhiteBalanceMode.AUTO
    awb_locked: bool = False
    denoise_mode: DenoiseMode = DenoiseMode.FAST
    denoise_strength: float | None = None
    sensor_mode: int | None = None
    hdr_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate independent setting ranges and types."""
        if self.exposure_us is not None and (
            isinstance(self.exposure_us, bool)
            or not isinstance(self.exposure_us, int)
            or not 0 < self.exposure_us <= 10_000_000
        ):
            raise ValueError("exposure_us must be between 1 and 10000000 or None")

        if self.gain is not None and (
            isinstance(self.gain, bool)
            or not isinstance(self.gain, (int, float))
            or not isfinite(self.gain)
            or not 0 < self.gain <= 1_024
        ):
            raise ValueError("gain must be finite and between 0 and 1024 or None")

        if not isinstance(self.awb_locked, bool):
            raise ValueError("awb_locked must be a boolean")

        if not isinstance(self.awb_mode, WhiteBalanceMode):
            raise ValueError("awb_mode must be a WhiteBalanceMode")

        if not isinstance(self.denoise_mode, DenoiseMode):
            raise ValueError("denoise_mode must be a DenoiseMode")

        if self.denoise_strength is not None and (
            isinstance(self.denoise_strength, bool)
            or not isinstance(self.denoise_strength, (int, float))
            or not isfinite(self.denoise_strength)
            or not 0.0 <= self.denoise_strength <= 1.0
        ):
            raise ValueError("denoise_strength must be between 0.0 and 1.0")

        if self.sensor_mode is not None and (
            isinstance(self.sensor_mode, bool)
            or not isinstance(self.sensor_mode, int)
            or not 0 <= self.sensor_mode <= 255
        ):
            raise ValueError("sensor_mode must be between 0 and 255 or None")

        if not isinstance(self.hdr_enabled, bool):
            raise ValueError("hdr_enabled must be a boolean")


@dataclass(frozen=True)
class CameraProfile:
    """A named, in-memory snapshot of camera settings."""

    name: str
    settings: CameraSettings

    def __post_init__(self) -> None:
        """Validate a profile name and its settings type."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.name):
            raise ValueError(
                "profile name must be 1-64 characters containing only letters, "
                "numbers, dots, underscores, and hyphens"
            )

        if not isinstance(self.settings, CameraSettings):
            raise ValueError("profile settings must be a CameraSettings instance")


@dataclass(frozen=True)
class CameraControlUpdate:
    """One atomic camera-control state transition."""

    revision: int
    previous: CameraSettings
    settings: CameraSettings
    changed_fields: tuple[str, ...]
    source_properties: tuple[str, ...]
    restart_required: bool = False

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable update representation."""
        return {
            "revision": self.revision,
            "previous": settings_to_dict(self.previous),
            "settings": settings_to_dict(self.settings),
            "changed_fields": list(self.changed_fields),
            "source_properties": list(self.source_properties),
            "restart_required": self.restart_required,
        }


RuntimeHandler = Callable[[CameraControlUpdate], None]


def settings_to_dict(settings: CameraSettings) -> dict[str, object]:
    """Convert settings to JSON-ready values without enum instances."""
    values = asdict(settings)
    values["awb_mode"] = settings.awb_mode.value
    values["denoise_mode"] = settings.denoise_mode.value
    return values


def capabilities_to_dict(capabilities: CameraCapabilities) -> dict[str, object]:
    """Convert capabilities to JSON-ready values."""
    return {
        "model": capabilities.model,
        "source_properties": sorted(capabilities.source_properties),
        "hdr_supported": capabilities.hdr_supported,
        "sensor_mode_metadata_available": capabilities.sensor_mode_metadata_available,
        "sensor_modes": [asdict(mode) for mode in capabilities.sensor_modes],
    }
