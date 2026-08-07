"""Public optional WebRTC/HLS production preview API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from packages.production_preview import (
    CudaOverlayRenderer,
    DescribedEncodedVideoSource,
    EncodedVideoSource,
    OverlayRectangle,
    PreviewClientStats,
    PreviewTransport,
    ProductionPreviewConfig,
    ProductionPreviewConfigurationError,
    ProductionPreviewDependencyError,
    ProductionPreviewError,
    ProductionPreviewServer,
    ProductionPreviewStats,
    RectangleMapper,
    build_hls_transport_pipeline,
    build_webrtc_peer_pipeline,
)

if TYPE_CHECKING:
    from packages.production_preview.api import (
        create_production_preview_app as create_production_preview_app,
    )

__all__ = [
    "EncodedVideoSource",
    "DescribedEncodedVideoSource",
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


def __getattr__(name: str) -> Any:
    """Load FastAPI only when the production browser application is requested."""
    if name != "create_production_preview_app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from packages.production_preview.api import create_production_preview_app
    except ImportError as error:
        raise ImportError(
            "Production preview HTTP support is optional. Install it with "
            '`uv add "imx-camera-toolkit[production-preview]"`.'
        ) from error
    globals()[name] = create_production_preview_app
    __all__.append(name)
    return create_production_preview_app
