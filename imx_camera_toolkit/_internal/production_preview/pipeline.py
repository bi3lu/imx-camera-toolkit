"""GStreamer transport pipeline builders for encoded Jetson video."""

from __future__ import annotations

from pathlib import Path

from imx_camera_toolkit._internal.camera.models import (
    EncodedStreamDescription,
    VideoCodec,
)

from .config import ProductionPreviewConfig


def _description(
    stream: VideoCodec | EncodedStreamDescription,
) -> EncodedStreamDescription:
    """Accept the legacy codec-only builder input without guessing fmtp."""
    if isinstance(stream, EncodedStreamDescription):
        return stream

    return EncodedStreamDescription(codec=stream)


def _encoded_caps(stream: VideoCodec | EncodedStreamDescription) -> str:
    """Return access-unit caps derived from the actual encoder description."""
    description = _description(stream)
    media_type = (
        "video/x-h264" if description.codec is VideoCodec.H264 else "video/x-h265"
    )
    fields = [
        f"stream-format=(string){description.stream_format}",
        f"alignment=(string){description.alignment}",
    ]

    if description.profile is not None:
        fields.append(f"profile=(string){description.profile}")

    if description.level is not None:
        fields.append(f"level=(string){description.level}")

    return f"{media_type}, " + ", ".join(fields)


def _parser(codec: VideoCodec) -> str:
    """Return the parser matching an encoded codec."""
    return "h264parse" if codec is VideoCodec.H264 else "h265parse"


def _quote_path(path: Path) -> str:
    """Escape a filesystem path for a quoted gst-launch property."""
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def build_hls_transport_pipeline(
    codec: VideoCodec | EncodedStreamDescription,
    config: ProductionPreviewConfig,
) -> str:
    """Build a parser/muxer pipeline; video encoding remains upstream."""
    description = _description(codec)
    config.validate_codec(description.codec)

    if config.hls_directory is None:
        raise ValueError("HLS transport requires hls_directory")

    playlist = _quote_path(config.hls_directory / "playlist.m3u8")
    segments = _quote_path(config.hls_directory / "segment%05d.ts")
    return (
        "appsrc name=encoded_source is-live=true format=time block=false "
        f'caps="{_encoded_caps(description)}" ! '
        "queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 "
        "leaky=downstream ! "
        f"{_parser(description.codec)} config-interval=-1 ! "
        f'hlssink2 name=hls_sink playlist-location="{playlist}" '
        f'location="{segments}" '
        f"target-duration={config.hls_target_duration} "
        f"playlist-length={config.hls_playlist_length} "
        f"max-files={config.hls_max_files} send-keyframe-requests=true"
    )


def build_webrtc_peer_pipeline(
    codec: VideoCodec | EncodedStreamDescription,
    config: ProductionPreviewConfig,
) -> str:
    """Build one H.264 RTP/WebRTC peer fed by the shared encoder output."""
    description = _description(codec)
    config.validate_codec(description.codec)
    rtp_caps = (
        "application/x-rtp,media=(string)video,encoding-name=(string)H264,"
        "clock-rate=(int)90000,payload=(int)96,packetization-mode=(string)1"
    )
    if description.profile_level_id is not None:
        rtp_caps += ",profile-level-id=(string)" + description.profile_level_id
    return (
        "webrtcbin name=webrtc bundle-policy=max-bundle "
        f"latency={config.webrtc_latency_ms} "
        "appsrc name=encoded_source is-live=true format=time block=false "
        f'caps="{_encoded_caps(description)}" ! '
        "queue name=peer_queue max-size-buffers=1 max-size-bytes=0 "
        "max-size-time=0 "
        "leaky=downstream ! "
        "h264parse name=peer_parser config-interval=-1 ! "
        "rtph264pay name=peer_payloader config-interval=-1 pt=96 ! "
        f"{rtp_caps} ! webrtc."
    )


__all__ = ["build_hls_transport_pipeline", "build_webrtc_peer_pipeline"]
