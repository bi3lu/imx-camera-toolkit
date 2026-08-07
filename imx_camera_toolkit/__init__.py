"""Public namespace for the IMX Camera Toolkit library.

Applications should import stable library APIs from this namespace instead of
the repository-internal ``packages`` package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .camera import (
    Camera,
    CameraConfig,
    CameraConfigurationError,
    CameraDependencyError,
    CameraError,
    CameraFrame,
    CameraOpenError,
    CameraProfile,
    CameraProfileStatus,
    CameraReadError,
    CameraRecoveryError,
    CameraStats,
    CameraTimeoutError,
    EncodedVideoFrame,
    Frame,
    FrameFormat,
    GpuBufferHandle,
    GpuCamera,
    GpuFrame,
    GpuFrameExpiredError,
    HardwareVideoConfig,
    MemoryType,
    MetricsRecorder,
    PipelineMetrics,
    PipelineStage,
    StageMetrics,
    VideoCodec,
    VideoEncodeStats,
    VideoOverlayRenderer,
    build_gpu_gstreamer_pipeline,
    get_camera_profile,
    list_camera_profiles,
)
from .consumers import (
    FrameConsumer,
    InferenceConsumer,
    InferencePreviewSource,
    InferenceResultSource,
    LatestFrameSubscription,
    OverlayRenderer,
    PreviewOverlayContext,
)
from .inference import (
    FrameSpec,
    InferenceResult,
    InferenceRunner,
    ShapeProfile,
    TensorOutput,
    TensorRTRunner,
)

if TYPE_CHECKING:
    from .preview import CameraPreview as CameraPreview
    from .preview import PreviewServer as PreviewServer
    from .preview import PreviewSource as PreviewSource
    from .preview import create_preview_app as create_preview_app
    from .preview import preview as preview
    from .preview import serve as serve

__version__ = "0.5.0"

__all__ = [
    "Camera",
    "CameraConfig",
    "CameraConfigurationError",
    "CameraProfile",
    "CameraProfileStatus",
    "CameraDependencyError",
    "CameraError",
    "CameraFrame",
    "CameraOpenError",
    "CameraReadError",
    "CameraRecoveryError",
    "CameraStats",
    "CameraTimeoutError",
    "EncodedVideoFrame",
    "Frame",
    "FrameConsumer",
    "FrameFormat",
    "FrameSpec",
    "GpuBufferHandle",
    "GpuCamera",
    "GpuFrame",
    "GpuFrameExpiredError",
    "HardwareVideoConfig",
    "InferenceResult",
    "InferenceConsumer",
    "InferencePreviewSource",
    "InferenceResultSource",
    "InferenceRunner",
    "LatestFrameSubscription",
    "MemoryType",
    "MetricsRecorder",
    "PipelineMetrics",
    "PipelineStage",
    "OverlayRenderer",
    "PreviewOverlayContext",
    "StageMetrics",
    "ShapeProfile",
    "TensorOutput",
    "TensorRTRunner",
    "VideoCodec",
    "VideoEncodeStats",
    "VideoOverlayRenderer",
    "build_gpu_gstreamer_pipeline",
    "get_camera_profile",
    "list_camera_profiles",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Lazily import browser-preview helpers and their optional dependencies.

    Raises:
        AttributeError: If ``name`` is not a public namespace member.
        ImportError: If preview helpers are requested without the ``preview``
            optional dependency group.
    """
    if name not in {
        "CameraPreview",
        "PreviewServer",
        "PreviewSource",
        "create_preview_app",
        "preview",
        "serve",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from .preview import (
            CameraPreview,
            PreviewServer,
            PreviewSource,
            create_preview_app,
            preview,
            serve,
        )

    except ImportError as error:
        raise ImportError(
            "Browser preview support is optional. Install it with "
            '`uv add "imx-camera-toolkit[preview]"`. '
        ) from error

    globals().update(
        {
            "CameraPreview": CameraPreview,
            "PreviewServer": PreviewServer,
            "PreviewSource": PreviewSource,
            "create_preview_app": create_preview_app,
            "preview": preview,
            "serve": serve,
        }
    )
    __all__.extend(
        (
            "CameraPreview",
            "PreviewServer",
            "PreviewSource",
            "create_preview_app",
            "preview",
            "serve",
        )
    )
    return globals()[name]
