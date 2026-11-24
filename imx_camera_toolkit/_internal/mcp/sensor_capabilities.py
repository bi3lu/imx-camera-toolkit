"""Project cached sensor metadata into the allowlisted MCP contract."""

from __future__ import annotations

import re

from ..camera.profiles.registry import CameraProfile
from ..camera_control.capabilities import DEFAULT_CAPABILITIES
from ..camera_control.models import CameraCapabilities as ArgusCapabilities
from .contracts import (
    MAX_SAFE_INTEGER,
    CameraCapabilities,
    ControlCapabilities,
    ExposureBounds,
    GainBounds,
    SensorMode,
)

_PUBLIC_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}")


def project_sensor_capabilities(
    profile: CameraProfile | None,
    sensor: ArgusCapabilities | None,
    inspected: ArgusCapabilities | None,
) -> CameraCapabilities:
    """Project only declarations and discovery bound to the selected sensor.

    Args:
        profile: Curated profile explicitly selected by the camera owner.
        sensor: Sensor-specific declarations, such as those used by the
            application's controller. Generic controller defaults are not evidence.
        inspected: Cached read-only Argus inspection with optional sensor
            metadata. Missing properties are authoritative for this installation.

    Returns:
        An immutable projection with unknown sensor facts left unset.
    """
    if sensor == DEFAULT_CAPABILITIES:
        sensor = None

    model: str | None
    if inspected is not None and inspected.model is not None:
        model = inspected.model

        if (
            sensor is not None
            and sensor.model is not None
            and sensor.model.casefold() != model.casefold()
        ):
            sensor = None

    elif sensor is not None and sensor.model is not None:
        model = sensor.model

    else:
        model = profile.sensor if profile is not None else None

    if model is not None and _PUBLIC_MODEL.fullmatch(model) is None:
        model = None

    modes = inspected.sensor_modes if inspected is not None else ()

    if not modes and sensor is not None:
        modes = sensor.sensor_modes

    exposure = _control_support("exposuretimerange", sensor, inspected)
    gain = _control_support("gainrange", sensor, inspected)
    controls = (
        ControlCapabilities(
            ExposureBounds() if exposure else exposure,
            GainBounds() if gain else gain,
        )
        if exposure is not None or gain is not None
        else None
    )
    return CameraCapabilities(
        sensor_model=model,
        sensor_modes=(
            tuple(
                SensorMode(
                    index=mode.index,
                    width=(
                        mode.width
                        if mode.width is not None and mode.width <= MAX_SAFE_INTEGER
                        else None
                    ),
                    height=(
                        mode.height
                        if mode.height is not None and mode.height <= MAX_SAFE_INTEGER
                        else None
                    ),
                    max_fps=mode.max_fps,
                    hdr=mode.hdr,
                )
                for mode in modes
            )
            if modes
            else None
        ),
        controls=controls,
    )


def _control_support(
    property_name: str,
    sensor: ArgusCapabilities | None,
    inspected: ArgusCapabilities | None,
) -> bool | None:
    """Use inspection to veto support and declarations to confirm it."""
    if inspected is not None and not inspected.supports(property_name):
        return False

    return sensor.supports(property_name) if sensor is not None else None
