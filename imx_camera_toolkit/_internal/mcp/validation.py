"""Validate decoded tool arguments before accessing their sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType

from .contracts import HARD_SNAPSHOT_LIMITS, SnapshotLimits
from .errors import MCPError

EMPTY_ARGUMENTS: Mapping[str, object] = MappingProxyType({})
_CAMERA_KEYS = frozenset({"camera_id"})
_SNAPSHOT_KEYS = _CAMERA_KEYS | {
    "max_width",
    "max_height",
    "jpeg_quality",
    "max_age_ms",
    "wait_ms",
}


@dataclass(frozen=True, slots=True)
class SnapshotRequest:
    """Validated request and effective constraints for the fixed JPEG source."""

    max_width: int
    max_height: int
    jpeg_quality: int | None
    max_jpeg_quality: int
    max_age_ms: int
    wait_ms: int


def _camera_arguments(
    arguments: object, allowed_keys: frozenset[str]
) -> Mapping[str, object]:
    """Require an object with only declared keys and the registered camera ID."""
    if not isinstance(arguments, Mapping) or any(
        type(key) is not str or key not in allowed_keys for key in arguments
    ):
        raise MCPError("invalid_argument")

    camera_id = arguments.get("camera_id", "primary")

    if type(camera_id) is not str or camera_id != "primary":
        raise MCPError("invalid_argument")

    return arguments


def validate_camera_input(arguments: object) -> None:
    """Validate the input shared by camera status and capabilities."""
    _camera_arguments(arguments, _CAMERA_KEYS)


def validate_inference_status_input(arguments: object) -> None:
    """Require an empty object for the one bound inference source."""
    if not isinstance(arguments, Mapping) or arguments:
        raise MCPError("invalid_argument")


def _integer(value: object, minimum: int, maximum: int) -> int:
    """Validate a bounded JSON integer without coercion."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise MCPError("invalid_argument")

    return value


def parse_snapshot_input(arguments: object, limits: SnapshotLimits) -> SnapshotRequest:
    """Apply snapshot defaults and validate all fields against server limits."""
    options = _camera_arguments(arguments, _SNAPSHOT_KEYS)
    return SnapshotRequest(
        max_width=_integer(
            options.get("max_width", limits.max_width), 1, limits.max_width
        ),
        max_height=_integer(
            options.get("max_height", limits.max_height), 1, limits.max_height
        ),
        jpeg_quality=(
            _integer(options["jpeg_quality"], 1, limits.max_jpeg_quality)
            if "jpeg_quality" in options
            else None
        ),
        max_jpeg_quality=limits.max_jpeg_quality,
        max_age_ms=_integer(
            options.get("max_age_ms", min(2_000, limits.max_age_ms)),
            1,
            limits.max_age_ms,
        ),
        wait_ms=_integer(options.get("wait_ms", 0), 0, limits.max_wait_ms),
    )


def validate_snapshot_limits(limits: SnapshotLimits) -> None:
    """Reject invalid server configuration before binding a source."""
    if not isinstance(limits, SnapshotLimits):
        raise ValueError("invalid server snapshot limits")

    for name, maximum in asdict(HARD_SNAPSHOT_LIMITS).items():
        value = getattr(limits, name)
        minimum = 0 if name == "max_wait_ms" else 1

        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("invalid server snapshot limits")
