"""Validation and property generation for NVIDIA Argus controls."""

from __future__ import annotations

from enum import Enum

from ..capabilities import DEFAULT_CAPABILITIES
from ..models import (
    CameraCapabilities,
    CameraSettings,
    DenoiseMode,
    UnsupportedControlError,
)


def build_argus_control_properties(
    settings: CameraSettings,
    capabilities: CameraCapabilities = DEFAULT_CAPABILITIES,
) -> tuple[str, ...]:
    """Build safe ``nvarguscamerasrc`` property assignments from settings.

    Raises:
        UnsupportedControlError: If the camera cannot provide a requested
            feature or sensor mode.
    """
    validate_settings_capabilities(settings, capabilities)
    properties: list[str] = []

    if settings.sensor_mode is not None:
        properties.append(f"sensor-mode={settings.sensor_mode}")

    exposure_or_gain_fixed = (
        settings.exposure_us is not None or settings.gain is not None
    )

    if settings.exposure_us is not None:
        exposure_ns = settings.exposure_us * 1_000
        properties.append(f'exposuretimerange="{exposure_ns} {exposure_ns}"')

    if settings.gain is not None:
        gain = format_number(settings.gain)
        properties.append(f'gainrange="{gain} {gain}"')

    if exposure_or_gain_fixed:
        properties.append('ispdigitalgainrange="1 1"')
        properties.append("aelock=false")

    properties.append(f"wbmode={settings.awb_mode.value}")
    properties.append(f"awblock={format_bool(settings.awb_locked)}")
    properties.append(f"tnr-mode={denoise_mode_value(settings.denoise_mode)}")

    if settings.denoise_strength is not None:
        properties.append(f"tnr-strength={format_number(settings.denoise_strength)}")

    return tuple(properties)


def validate_settings_capabilities(
    settings: CameraSettings,
    capabilities: CameraCapabilities,
) -> None:
    """Validate settings that depend on available source properties."""
    required_properties = {"wbmode", "awblock", "tnr-mode"}

    if settings.sensor_mode is not None:
        required_properties.add("sensor-mode")

    if settings.exposure_us is not None:
        required_properties.add("exposuretimerange")

    if settings.gain is not None:
        required_properties.add("gainrange")

    if settings.exposure_us is not None or settings.gain is not None:
        required_properties.update({"aelock", "ispdigitalgainrange"})

    if settings.denoise_strength is not None:
        required_properties.add("tnr-strength")

    missing = sorted(
        property_name
        for property_name in required_properties
        if not capabilities.supports(property_name)
    )

    if missing:
        raise UnsupportedControlError(
            "camera does not support required control(s): " + ", ".join(missing)
        )

    if settings.sensor_mode is not None and capabilities.sensor_mode_metadata_available:
        mode = capabilities.get_sensor_mode(settings.sensor_mode)

        if mode is None:
            raise UnsupportedControlError(
                f"sensor mode {settings.sensor_mode} is not declared by capabilities"
            )

        if mode.hdr != settings.hdr_enabled:
            expected = "HDR" if settings.hdr_enabled else "non-HDR"
            raise UnsupportedControlError(
                f"sensor mode {settings.sensor_mode} is not a declared {expected} mode"
            )

    if settings.hdr_enabled:
        if not capabilities.hdr_supported:
            raise UnsupportedControlError("camera does not declare an HDR sensor mode")

        if settings.sensor_mode is None:
            raise UnsupportedControlError("HDR requires an explicit HDR sensor mode")


def coerce_enum(enum_type: type[Enum], value: object, field_name: str) -> Enum:
    """Convert a string or enum instance to one supported enum value."""
    if isinstance(value, enum_type):
        return value

    if isinstance(value, str):
        try:
            return enum_type(value)

        except ValueError as error:
            choices = ", ".join(member.value for member in enum_type)
            raise ValueError(f"{field_name} must be one of: {choices}") from error

    raise ValueError(f"{field_name} must be a {enum_type.__name__} or string")


def denoise_mode_value(mode: DenoiseMode) -> int:
    """Return the integer representation used by ``tnr-mode``."""
    return {
        DenoiseMode.OFF: 0,
        DenoiseMode.FAST: 1,
        DenoiseMode.HIGH_QUALITY: 2,
    }[mode]


def format_bool(value: bool) -> str:
    """Format a boolean as a GStreamer-compatible literal."""
    return "true" if value else "false"


def format_number(value: int | float) -> str:
    """Format a number without unnecessary trailing zeroes."""
    return format(value, "g")
