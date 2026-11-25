"""Typed results for the versioned MCP tool contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CameraBackend = Literal["cpu", "gpu"]
CameraFrameFormat = Literal["BGR_CPU", "NV12_NVMM"]
CameraState = Literal["unavailable", "starting", "ready", "recovering", "error"]
InferenceState = Literal["unknown", "stopped", "initializing", "running", "degraded"]
MAX_SAFE_INTEGER = (1 << 53) - 1


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


HARD_SNAPSHOT_LIMITS = SnapshotLimits(1920, 1080, 95, 10_000, 1_000, 524_288, 786_432)


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
    frame_format: CameraFrameFormat | None = None
    controls: ControlCapabilities | None = None
    snapshot: SnapshotCapabilities | None = None
    schema_version: Literal[1] = field(default=1, init=False)
    camera_id: Literal["primary"] = field(default="primary", init=False)


@dataclass(frozen=True, slots=True)
class SnapshotMetadata:
    """Metadata for one JPEG, including its age on the server monotonic clock."""

    width: int
    height: int
    jpeg_quality: int | None
    jpeg_bytes: int
    frame_sequence: int
    age_ms: float
    schema_version: Literal[1] = field(default=1, init=False)
    camera_id: Literal["primary"] = field(default="primary", init=False)
    mime_type: Literal["image/jpeg"] = field(default="image/jpeg", init=False)
    sequence_domain: Literal["jpeg_publication"] = field(
        default="jpeg_publication", init=False
    )
    age_clock: Literal["server_monotonic"] = field(
        default="server_monotonic", init=False
    )


@dataclass(frozen=True, slots=True)
class CameraSnapshot:
    """One detached image and the metadata for that exact publication."""

    metadata: SnapshotMetadata
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class InferenceStatus:
    """Cached worker diagnostics with optional timings in milliseconds."""

    state: InferenceState
    processed_frames: int | None = None
    failed_frames: int | None = None
    dropped_frames: int | None = None
    consecutive_failures: int | None = None
    error: Literal["worker_failure"] | None = None
    inference_time_ms: float | None = None
    age_ms: float | None = None
    schema_version: Literal[1] = field(default=1, init=False)
