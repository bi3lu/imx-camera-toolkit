"""Lazy PyGObject loading for optional production video transports."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .errors import ProductionPreviewDependencyError


@dataclass(frozen=True, slots=True)
class GStreamerRuntime:
    """GStreamer namespaces required by HLS and optional WebRTC signaling."""

    Gst: Any
    GstSdp: Any | None = None
    GstWebRTC: Any | None = None


def load_gstreamer_runtime(
    *,
    webrtc: bool = False,
    required_elements: tuple[str, ...] = (),
) -> GStreamerRuntime:
    """Import system GStreamer namespaces only when production preview starts."""
    try:
        gi_module = importlib.import_module("gi")
        gi_module.require_version("Gst", "1.0")

        if webrtc:
            gi_module.require_version("GstSdp", "1.0")
            gi_module.require_version("GstWebRTC", "1.0")

        gst = importlib.import_module("gi.repository.Gst")
        gst.init(None)
        required = (
            (
                "appsrc",
                "queue",
                "capsfilter",
                "webrtcbin",
                "h264parse",
                "rtph264pay",
                "nicesrc",
                "nicesink",
            )
            if webrtc
            else ("appsrc", "queue", "hlssink2")
        ) + required_elements
        missing = [
            name for name in required if gst.ElementFactory.find(name) is None
        ]

        if missing:
            formatted = ", ".join(missing)
            raise ProductionPreviewDependencyError(
                f"required GStreamer element(s) unavailable: {formatted}"
            )

        gst_sdp = (
            importlib.import_module("gi.repository.GstSdp") if webrtc else None
        )

        gst_webrtc = (
            importlib.import_module("gi.repository.GstWebRTC") if webrtc else None
        )
        return GStreamerRuntime(gst, gst_sdp, gst_webrtc)

    except ProductionPreviewDependencyError:
        raise

    except (ImportError, ValueError) as error:
        runtime_name = "GStreamer WebRTC" if webrtc else "GStreamer HLS"
        raise ProductionPreviewDependencyError(
            f"{runtime_name} PyGObject runtime is unavailable"
        ) from error


__all__ = ["GStreamerRuntime", "load_gstreamer_runtime"]
