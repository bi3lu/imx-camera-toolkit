"""Curated hardware configuration profiles for supported IMX cameras.

Profiles intentionally describe only static sensor and frame-layout settings.
They do not contain runtime controls, AI settings, JPEG quality, or network
configuration. The catalog is implemented without a YAML parser so profile
selection remains available in the dependency-free core package.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config.loader import CameraConfig


class CameraProfileStatus(str, Enum):
    """Verification status assigned to a curated hardware profile."""

    TESTED = "tested"
    COMMUNITY_TESTED = "community-tested"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class CameraProfile:
    """Named static configuration for one verified camera operating mode.

    Attributes:
        name: Stable identifier accepted by :meth:`CameraConfig.from_profile`.
        sensor: Exact sensor or camera module associated with the profile.
        status: Evidence level for the documented configuration.
        config: Static camera configuration for this operating mode.
        aliases: Additional accepted profile identifiers, not shown in listings.
    """

    name: str
    sensor: str
    status: CameraProfileStatus
    config: CameraConfig
    aliases: tuple[str, ...] = ()

    def hardware_settings(self) -> dict[str, object]:
        """Return the profile's portable hardware-only settings mapping.

        Returns:
            Mapping with the sensor, capture, and output sections used by the
            documented YAML profile format.
        """
        return {
            "sensor_id": self.config.sensor_id,
            "sensor_mode": self.config.sensor_mode,
            "capture": {
                "width": self.config.capture_width,
                "height": self.config.capture_height,
                "fps": self.config.fps,
            },
            "output": {
                "width": self.config.output_width,
                "height": self.config.output_height,
            },
        }


PROFILES = (
    CameraProfile(
        name="imx219-1080p",
        sensor="IMX219-77",
        status=CameraProfileStatus.TESTED,
        config=CameraConfig(
            sensor_id=0,
            sensor_mode=2,
            capture_width=1920,
            capture_height=1080,
            output_width=1280,
            output_height=720,
            fps=30,
        ),
        aliases=("imx219-77-1080p",),
    ),
)


def list_camera_profiles() -> tuple[CameraProfile, ...]:
    """Return all curated hardware profiles in stable name order."""
    return PROFILES


def get_camera_profile(name: str) -> CameraProfile:
    """Return a curated hardware profile by its name or documented alias.

    Args:
        name: Profile identifier, for example ``"imx219-1080p"``.

    Returns:
        Matching immutable camera profile.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If no curated profile has the requested name.
    """
    if not isinstance(name, str):
        raise TypeError("profile name must be a string")

    normalized_name = name.strip().lower()

    for profile in PROFILES:
        if normalized_name == profile.name or normalized_name in profile.aliases:
            return profile

    available_profiles = ", ".join(profile.name for profile in PROFILES)
    raise ValueError(
        f"unknown camera profile {name!r}; available profiles: {available_profiles}"
    )
