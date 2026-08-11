"""Contracts for optional encoded production preview frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol, runtime_checkable

from .formats import MemoryType
from .gpu_frame import GpuFrame


class VideoCodec(str, Enum):
    """Video codecs supported by production preview encoders."""

    H264 = "H264"
    H265 = "H265"


class VideoEncoderBackend(str, Enum):
    """Encoder selection policy for the isolated production video branch."""

    AUTO = "auto"
    NVENC = "nvenc"
    X264 = "x264"


@dataclass(frozen=True, slots=True)
class VideoEncoderConfig:
    """Validated settings for an NVMM-fed production video encoder.

    ``AUTO`` prefers Jetson NVENC when its GStreamer element exists and falls
    back to x264 for H.264.  Only the x264 encoder branch leaves NVMM.
    """

    codec: VideoCodec = VideoCodec.H264
    bitrate_bps: int = 4_000_000
    keyframe_interval: int = 30
    max_performance: bool = False
    measure_latency: bool = True
    backend: VideoEncoderBackend = VideoEncoderBackend.AUTO
    software_preset: str = "ultrafast"

    def __post_init__(self) -> None:
        """Reject settings unsupported by the production pipeline contract."""
        if not isinstance(self.codec, VideoCodec):
            raise ValueError("codec must be a VideoCodec")

        if not isinstance(self.backend, VideoEncoderBackend):
            raise ValueError("backend must be a VideoEncoderBackend")

        if (
            self.backend is VideoEncoderBackend.X264
            and self.codec is not VideoCodec.H264
        ):
            raise ValueError("x264 backend supports only H.264")

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

        if self.software_preset not in {
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
        }:
            raise ValueError("software_preset is not supported by x264enc")


HardwareVideoConfig = VideoEncoderConfig


@dataclass(frozen=True, slots=True)
class EncodedStreamDescription:
    """Negotiated, immutable description of encoded access units."""

    codec: VideoCodec
    stream_format: str = "byte-stream"
    alignment: str = "au"
    profile: str | None = None
    level: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    codec_data: bytes | None = None
    sps: bytes | None = None
    pps: bytes | None = None

    def __post_init__(self) -> None:
        """Validate portable caps metadata without importing GStreamer."""
        if not isinstance(self.codec, VideoCodec):
            raise ValueError("codec must be a VideoCodec")

        if not isinstance(self.stream_format, str) or not self.stream_format:
            raise ValueError("stream_format must be a non-empty string")

        if not isinstance(self.alignment, str) or not self.alignment:
            raise ValueError("alignment must be a non-empty string")

        for name in ("profile", "level"):
            value = getattr(self, name)

            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")

        for name in ("width", "height", "fps"):
            value = getattr(self, name)

            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")

        for name in ("codec_data", "sps", "pps"):
            value = getattr(self, name)

            if value is not None and (not isinstance(value, bytes) or not value):
                raise ValueError(f"{name} must be non-empty bytes or None")

    @property
    def profile_level_id(self) -> str | None:
        """Return RFC 6184 profile-level-id derived from the real SPS."""
        if self.codec is not VideoCodec.H264 or self.sps is None:
            return None

        sps = self.sps
        if len(sps) >= 4 and sps[0] & 0x1F == 7:
            return sps[1:4].hex()

        return None


@dataclass(frozen=True, slots=True)
class VideoEncoderPipeline:
    """Public result returned by a custom encoder pipeline factory."""

    pipeline: str
    backend: str
    description: EncodedStreamDescription
    required_elements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject incomplete custom pipeline definitions early."""
        if not isinstance(self.pipeline, str) or not self.pipeline.strip():
            raise ValueError("pipeline must be a non-empty GStreamer segment")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be a non-empty string")
        if not isinstance(self.description, EncodedStreamDescription):
            raise ValueError("description must be an EncodedStreamDescription")
        if any(
            not isinstance(item, str) or not item for item in self.required_elements
        ):
            raise ValueError("required_elements must contain non-empty strings")


class VideoEncoderPipelineFactory(Protocol):
    """Build a public encoder segment without touching ``GpuCamera`` internals."""

    def __call__(
        self,
        config: VideoEncoderConfig,
        width: int,
        height: int,
        framerate: int,
    ) -> VideoEncoderPipeline:
        """Return a complete encoder/parser/appsink segment."""
        ...


@dataclass(frozen=True, slots=True)
class EncodedVideoFrame:
    """One encoded access unit copied from the selected encoder output."""

    sequence: int
    timestamp_ns: int
    codec: VideoCodec
    data: bytes
    keyframe: bool
    pts_ns: int | None = None
    duration_ns: int | None = None
    dts_ns: int | None = None
    stream_description: EncodedStreamDescription | None = None

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

        for name in ("pts_ns", "dts_ns", "duration_ns"):
            value = getattr(self, name)

            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be non-negative or None")

        if self.stream_description is not None and not isinstance(
            self.stream_description, EncodedStreamDescription
        ):
            raise ValueError(
                "stream_description must be an EncodedStreamDescription or None"
            )


@dataclass(frozen=True, slots=True)
class VideoEncodeStats:
    """Fixed-size video encoder throughput snapshot."""

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
    "EncodedStreamDescription",
    "EncodedVideoFrame",
    "HardwareVideoConfig",
    "VideoCodec",
    "VideoEncoderBackend",
    "VideoEncoderConfig",
    "VideoEncoderPipeline",
    "VideoEncoderPipelineFactory",
    "VideoEncodeStats",
    "VideoOverlayRenderer",
]
