"""Typed camera results for the versioned MCP tool contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CameraBackend = Literal["cpu", "gpu"]
CameraState = Literal["unavailable", "starting", "ready", "recovering", "error"]


@dataclass(frozen=True, slots=True)
class CameraStatus:
    """Versioned status with UTC observation time and allowlisted metrics."""

    state: CameraState
    backend: CameraBackend | None
    observed_at_utc: str
    toolkit_version: str
    capture_fps: float | None = None
    capture_dropped_frames: int | None = None
    subscriber_dropped_frames: dict[str, int] | None = None
    schema_version: Literal[1] = field(default=1, init=False)
    camera_id: Literal["primary"] = field(default="primary", init=False)


@dataclass(frozen=True, slots=True)
class SensorMode:
    """Known sensor mode with pixel dimensions and maximum frames per second."""

    index: int
    width: int | None = None
    height: int | None = None
    max_fps: float | None = None
    hdr: bool | None = None


@dataclass(frozen=True, slots=True)
class ExposureBounds:
    """Fixed-exposure bounds in microseconds, with optional endpoints."""

    min: int | None = None
    max: int | None = None


@dataclass(frozen=True, slots=True)
class GainBounds:
    """Linear gain bounds with optional endpoints."""

    min: float | None = None
    max: float | None = None


@dataclass(frozen=True, slots=True)
class ControlCapabilities:
    """Control support reported as bounds, unsupported, or unknown."""

    exposure_us: ExposureBounds | Literal[False] | None = None
    gain: GainBounds | Literal[False] | None = None


@dataclass(frozen=True, slots=True)
class SnapshotLimits:
    """Effective snapshot limits in pixels, milliseconds, and bytes."""

    max_width: int
    max_height: int
    max_jpeg_quality: int
    max_age_ms: int
    max_wait_ms: int
    max_jpeg_bytes: int
    max_response_bytes: int


@dataclass(frozen=True, slots=True)
class SnapshotCapabilities:
    """Snapshot source support, MCP configuration, and effective limits."""

    source_supported: bool | None
    configured: bool
    limits: SnapshotLimits


@dataclass(frozen=True, slots=True)
class CameraCapabilities:
    """Versioned public capability values, with ``None`` for unknown fields."""

    sensor_model: str | None = None
    sensor_modes: tuple[SensorMode, ...] | None = None
    frame_format: Literal["BGR_CPU", "NV12_NVMM"] | None = None
    controls: ControlCapabilities | None = None
    snapshot: SnapshotCapabilities | None = None
    schema_version: Literal[1] = field(default=1, init=False)
    camera_id: Literal["primary"] = field(default="primary", init=False)
