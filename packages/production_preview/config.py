"""Validated WebRTC and HLS production preview configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path

from packages.camera.models import VideoCodec

from .errors import ProductionPreviewConfigurationError


class PreviewTransport(str, Enum):
    """Browser-oriented production video transports."""

    WEBRTC = "webrtc"
    HLS = "hls"


@dataclass(frozen=True, slots=True)
class ProductionPreviewConfig:
    """Transport settings applied after the shared video encoder."""

    transport: PreviewTransport = PreviewTransport.WEBRTC
    max_clients: int = 4
    client_timeout_seconds: float = 30.0
    webrtc_latency_ms: int = 50
    stun_server: str | None = None
    turn_server: str | None = None
    hls_directory: Path | None = None
    hls_target_duration: int = 1
    hls_playlist_length: int = 3
    hls_max_files: int = 5
    stream_description_timeout_seconds: float = 2.0
    max_sdp_bytes: int = 64 * 1024
    max_ice_candidate_bytes: int = 4 * 1024
    max_ice_candidates_per_session: int = 64
    max_new_sessions_per_second: int = 4

    def __post_init__(self) -> None:
        """Validate transport limits without importing optional runtimes."""
        if not isinstance(self.transport, PreviewTransport):
            raise ProductionPreviewConfigurationError(
                "transport must be a PreviewTransport"
            )

        integer_limits = {
            "max_clients": 1024,
            "webrtc_latency_ms": 60_000,
            "hls_target_duration": 3_600,
            "hls_playlist_length": 10_000,
            "hls_max_files": 10_000,
            "max_sdp_bytes": 1024 * 1024,
            "max_ice_candidate_bytes": 64 * 1024,
            "max_ice_candidates_per_session": 4096,
            "max_new_sessions_per_second": 1000,
        }

        for name, maximum in integer_limits.items():
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 < value <= maximum
            ):
                raise ProductionPreviewConfigurationError(
                    f"{name} must be a positive integer no greater than {maximum}"
                )

        for name in (
            "client_timeout_seconds",
            "stream_description_timeout_seconds",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 < value <= 3_600
            ):
                raise ProductionPreviewConfigurationError(
                    f"{name} must be finite and positive"
                )

        if self.hls_playlist_length < 3:
            raise ProductionPreviewConfigurationError(
                "hls_playlist_length must be at least 3"
            )

        if self.hls_max_files < self.hls_playlist_length:
            raise ProductionPreviewConfigurationError(
                "hls_max_files must be at least hls_playlist_length"
            )

        for name in ("stun_server", "turn_server"):
            value = getattr(self, name)

            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ProductionPreviewConfigurationError(
                    f"{name} must be a non-empty string or None"
                )

        if self.hls_directory is not None:
            object.__setattr__(self, "hls_directory", Path(self.hls_directory))

        if self.transport is PreviewTransport.HLS and self.hls_directory is None:
            raise ProductionPreviewConfigurationError(
                "hls_directory is required for HLS transport"
            )

    def validate_codec(self, codec: VideoCodec) -> None:
        """Reject codec/transport pairs unsupported by common browsers."""
        if not isinstance(codec, VideoCodec):
            raise ProductionPreviewConfigurationError("codec must be a VideoCodec")

        if self.transport is PreviewTransport.WEBRTC and codec is VideoCodec.H265:
            raise ProductionPreviewConfigurationError(
                "WebRTC preview requires H.264 for broad browser compatibility; "
                "use HLS for H.265"
            )


__all__ = ["PreviewTransport", "ProductionPreviewConfig"]
