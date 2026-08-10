"""Runtime NVIDIA Argus property parsing and application."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

ARGUS_RUNTIME_DEFAULTS = {
    "aelock": False,
    "awblock": False,
    "exposuretimerange": None,
    "gainrange": None,
    "ispdigitalgainrange": None,
    "sensor-mode": -1,
    "tnr-mode": 1,
    "tnr-strength": -1.0,
    "wbmode": 1,
}

ARGUS_ENUM_VALUES = {
    "tnr-mode": {"0": 0, "1": 1, "2": 2},
    "wbmode": {
        "off": 0,
        "auto": 1,
        "incandescent": 2,
        "fluorescent": 3,
        "warm-fluorescent": 4,
        "daylight": 5,
        "cloudy-daylight": 6,
        "twilight": 7,
        "shade": 8,
        "manual": 9,
    },
}

MANUAL_CONTROL_PROPERTY_NAMES = frozenset(
    {"aelock", "exposuretimerange", "gainrange", "ispdigitalgainrange"}
)


def parse_argus_property(property_value: str) -> tuple[str, Any]:
    """Convert one validated pipeline assignment to a GObject property value."""
    name, raw_value = property_value.split("=", maxsplit=1)
    raw_value = raw_value.strip('"')

    if name in {"aelock", "awblock"}:
        return name, raw_value == "true"

    if name in ARGUS_ENUM_VALUES:
        return name, ARGUS_ENUM_VALUES[name][raw_value]

    if name == "tnr-strength":
        return name, float(raw_value)

    if name == "sensor-mode":
        return name, int(raw_value)

    return name, raw_value


def manual_control_properties(properties: Sequence[str]) -> tuple[str, ...]:
    """Return controls that use the V4L2 sensor interface at runtime."""
    return tuple(
        property_value
        for property_value in properties
        if property_value.split("=", maxsplit=1)[0] in MANUAL_CONTROL_PROPERTY_NAMES
    )


def non_manual_control_properties(properties: Sequence[str]) -> tuple[str, ...]:
    """Return controls that remain safe to set through GObject."""
    return tuple(
        property_value
        for property_value in properties
        if property_value.split("=", maxsplit=1)[0] not in MANUAL_CONTROL_PROPERTY_NAMES
    )


def apply_live_properties(
    source: Any,
    current_properties: Sequence[str],
    properties: Sequence[str],
) -> None:
    """Apply changed non-manual Argus properties to a live source.

    Args:
        source: Active ``nvarguscamerasrc`` GObject.
        current_properties: Existing non-manual assignments.
        properties: Requested non-manual assignments.
    """
    current = dict(parse_argus_property(value) for value in current_properties)
    requested = dict(parse_argus_property(value) for value in properties)

    changed_values = {
        property_name: ARGUS_RUNTIME_DEFAULTS[property_name]
        for property_name in current.keys() - requested.keys()
        if property_name in ARGUS_RUNTIME_DEFAULTS
    }

    changed_values.update(
        {
            property_name: value
            for property_name, value in requested.items()
            if current.get(property_name) != value
        }
    )

    for property_name, value in changed_values.items():
        source.set_property(property_name, value)
