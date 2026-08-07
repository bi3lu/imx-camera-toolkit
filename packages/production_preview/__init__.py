"""Optional shared-encoder WebRTC and HLS production preview."""

from .config import PreviewTransport, ProductionPreviewConfig
from .errors import (
    ProductionPreviewConfigurationError,
    ProductionPreviewDependencyError,
    ProductionPreviewError,
)
from .metrics import PreviewClientStats, ProductionPreviewStats
from .overlay import CudaOverlayRenderer, OverlayRectangle, RectangleMapper
from .pipeline import build_hls_transport_pipeline, build_webrtc_peer_pipeline
from .transport import EncodedVideoSource, ProductionPreviewServer

__all__ = [
    "EncodedVideoSource",
    "CudaOverlayRenderer",
    "OverlayRectangle",
    "PreviewClientStats",
    "PreviewTransport",
    "ProductionPreviewConfig",
    "ProductionPreviewConfigurationError",
    "ProductionPreviewDependencyError",
    "ProductionPreviewError",
    "ProductionPreviewServer",
    "ProductionPreviewStats",
    "RectangleMapper",
    "build_hls_transport_pipeline",
    "build_webrtc_peer_pipeline",
]
