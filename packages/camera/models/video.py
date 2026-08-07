"""Contracts for optional hardware-encoded production preview frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol, runtime_checkable

from .formats import MemoryType
from .gpu_frame import GpuFrame


class VideoCodec(str, Enum):
    """Hardware video codecs supported by the Jetson V4L2 encoder."""

    H264 = "H264"
    H265 = "H265"


@dataclass(frozen=True, slots=True)
class HardwareVideoConfig:
    """Validated NVMM-to-hardware-encoder settings."""

    codec: VideoCodec = VideoCodec.H264
    bitrate_bps: int = 4_000_000
    keyframe_interval: int = 30
    max_performance: bool = False
    measure_latency: bool = True

    def __post_init__(self) -> None:
        """Reject settings unsupported by the production pipeline contract."""
        if not isinstance(self.codec, VideoCodec):
            raise ValueError("codec must be a VideoCodec")

        if (
            isinstance(self.bitrate_bps, bool)
            or not isinstance(self.bitrate_bps, int)
            or not 64_000 <= self.bitrate_bps <= 120_000_000
        ):
            raise ValueError("bitrate_bps must be between 64000 and 120000000")

        if (
            isinstance(self.keyframe_interval, bool)
            or not isinstance(self.keyframe_interval, int)
            or self.keyframe_interval <= 0
        ):
            raise ValueError("keyframe_interval must be a positive integer")

        for name in ("max_performance", "measure_latency"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class EncodedVideoFrame:
    """One encoded access unit copied from the hardware encoder output."""

    sequence: int
    timestamp_ns: int
    codec: VideoCodec
    data: bytes
    keyframe: bool
    pts_ns: int | None = None
    duration_ns: int | None = None

    def __post_init__(self) -> None:
        """Validate portable encoded-frame metadata."""
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise ValueError("sequence must be a positive integer")

        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be non-negative")

        if not isinstance(self.codec, VideoCodec):
            raise ValueError("codec must be a VideoCodec")

        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("data must be non-empty bytes")

        if not isinstance(self.keyframe, bool):
            raise ValueError("keyframe must be a boolean")

        for name in ("pts_ns", "duration_ns"):
            value = getattr(self, name)

            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be non-negative or None")


@dataclass(frozen=True, slots=True)
class VideoEncodeStats:
    """Fixed-size hardware encoder throughput snapshot."""

    encoded_frames: int = 0
    encoded_bytes: int = 0
    encode_fps: float = 0.0
    bitrate_bps: float = 0.0
    last_frame_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        """Validate non-negative cumulative and recent-rate metrics."""
        for name in ("encoded_frames", "encoded_bytes"):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        for name in ("encode_fps", "bitrate_bps"):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")


@runtime_checkable
class VideoOverlayRenderer(Protocol):
    """In-place overlay hook executed on an isolated encoder-branch frame."""

    @property
    def memory_type(self) -> MemoryType:
        """Memory domain required by this renderer."""
        ...

    def render(self, frame: GpuFrame) -> None:
        """Finish drawing before returning the frame to the encoder."""
        ...

    def close(self) -> None:
        """Release renderer-owned GPU resources."""
        ...


__all__ = [
    "EncodedVideoFrame",
    "HardwareVideoConfig",
    "VideoCodec",
    "VideoEncodeStats",
    "VideoOverlayRenderer",
]
