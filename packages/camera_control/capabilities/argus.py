"""Argus capability declarations and local discovery."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Sequence
from typing import Final

from ..models import CameraCapabilities, SensorMode

ARGUS_CONTROL_PROPERTIES: Final[frozenset[str]] = frozenset(
    {
        "aelock",
        "awblock",
        "exposuretimerange",
        "gainrange",
        "ispdigitalgainrange",
        "sensor-mode",
        "tnr-mode",
        "tnr-strength",
        "wbmode",
    }
)

DEFAULT_CAPABILITIES: Final[CameraCapabilities] = CameraCapabilities(
    source_properties=ARGUS_CONTROL_PROPERTIES
)


def discover_argus_capabilities(
    command: Sequence[str] = ("gst-inspect-1.0", "nvarguscamerasrc"),
    *,
    timeout: float = 3.0,
    sensor_modes: Iterable[SensorMode] = (),
    model: str | None = None,
) -> CameraCapabilities:
    """Inspect the locally installed Argus source without opening a sensor.

    Args:
        command: Command used to inspect the GStreamer element.
        timeout: Maximum execution time in seconds.
        sensor_modes: Optional sensor-mode metadata supplied by the integrator.
        model: Optional sensor model associated with the supplied modes.

    Returns:
        Discovered source property names and supplied sensor metadata.

    Raises:
        RuntimeError: If GStreamer inspection is unavailable or unsuccessful.
        ValueError: If the timeout is invalid.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive number")

    if timeout <= 0:
        raise ValueError("timeout must be a positive number")

    if not command:
        raise ValueError("inspection command must not be empty")

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Could not inspect nvarguscamerasrc") from error

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not inspect nvarguscamerasrc: {message}")

    property_pattern = r"^  ([a-z][a-z0-9-]*)\s+:"
    properties = frozenset(
        match.group(1)
        for match in re.finditer(property_pattern, result.stdout, re.MULTILINE)
    )

    if not properties:
        raise RuntimeError("nvarguscamerasrc did not report any properties")

    return CameraCapabilities(
        source_properties=properties,
        sensor_modes=tuple(sensor_modes),
        model=model,
    )
