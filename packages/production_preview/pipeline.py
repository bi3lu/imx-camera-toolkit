"""GStreamer transport pipeline builders for encoded Jetson video."""

from __future__ import annotations

from pathlib import Path

from packages.camera.models import VideoCodec

from .config import ProductionPreviewConfig


def _encoded_caps(codec: VideoCodec) -> str:
    """Return byte-stream access-unit caps for one hardware codec."""
    media_type = "video/x-h264" if codec is VideoCodec.H264 else "video/x-h265"
    return (
        f"{media_type}, stream-format=(string)byte-stream, "
        "alignment=(string)au"
    )


def _parser(codec: VideoCodec) -> str:
    """Return the parser matching an encoded codec."""
    return "h264parse" if codec is VideoCodec.H264 else "h265parse"


def _quote_path(path: Path) -> str:
    """Escape a filesystem path for a quoted gst-launch property."""
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def build_hls_transport_pipeline(
    codec: VideoCodec,
    config: ProductionPreviewConfig,
) -> str:
    """Build a parser/muxer pipeline; hardware encoding remains upstream."""
    config.validate_codec(codec)

    if config.hls_directory is None:
        raise ValueError("HLS transport requires hls_directory")

    playlist = _quote_path(config.hls_directory / "playlist.m3u8")
    segments = _quote_path(config.hls_directory / "segment%05d.ts")
    return (
        "appsrc name=encoded_source is-live=true format=time block=false "
        f'caps="{_encoded_caps(codec)}" ! '
        "queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 "
        "leaky=downstream ! "
        f"{_parser(codec)} config-interval=-1 ! "
        f'hlssink2 name=hls_sink playlist-location="{playlist}" '
        f'location="{segments}" '
        f"target-duration={config.hls_target_duration} "
        f"playlist-length={config.hls_playlist_length} "
        f"max-files={config.hls_max_files} send-keyframe-requests=true"
    )


def build_webrtc_peer_pipeline(
    codec: VideoCodec,
    config: ProductionPreviewConfig,
) -> str:
    """Build one H.264 RTP/WebRTC peer fed by the shared encoder output."""
    config.validate_codec(codec)
    return (
        "webrtcbin name=webrtc bundle-policy=max-bundle "
        f"latency={config.webrtc_latency_ms} "
        "appsrc name=encoded_source is-live=true format=time block=false "
        f'caps="{_encoded_caps(codec)}" ! '
        "queue name=peer_queue max-size-buffers=1 max-size-bytes=0 "
        "max-size-time=0 "
        "leaky=downstream ! "
        "h264parse config-interval=-1 ! "
        "rtph264pay config-interval=-1 pt=96 ! "
        "application/x-rtp,media=(string)video,encoding-name=(string)H264,"
        "clock-rate=(int)90000,payload=(int)96 ! webrtc."
    )


__all__ = ["build_hls_transport_pipeline", "build_webrtc_peer_pipeline"]
