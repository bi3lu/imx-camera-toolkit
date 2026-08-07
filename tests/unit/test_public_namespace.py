"""Tests for the stable external ``imx_camera_toolkit`` namespace."""

from __future__ import annotations

from imx_camera_toolkit import Camera as RootCamera
from imx_camera_toolkit import (
    CameraConfig,
    CameraConfigurationError,
    CameraDependencyError,
    CameraError,
    CameraFrame,
    CameraOpenError,
    CameraPreview,
    CameraProfile,
    CameraProfileStatus,
    CameraReadError,
    CameraRecoveryError,
    CameraStats,
    CameraTimeoutError,
    EncodedVideoFrame,
    Frame,
    FrameConsumer,
    FrameFormat,
    FrameSpec,
    GpuBufferHandle,
    GpuCamera,
    GpuFrame,
    GpuFrameExpiredError,
    HardwareVideoConfig,
    InferenceConsumer,
    InferencePreviewSource,
    InferenceResult,
    InferenceResultSource,
    InferenceRunner,
    LatestFrameSubscription,
    MemoryType,
    MetricsRecorder,
    OverlayRenderer,
    PipelineMetrics,
    PipelineStage,
    PreviewOverlayContext,
    PreviewServer,
    PreviewSource,
    ShapeProfile,
    StageMetrics,
    TensorOutput,
    TensorRTRunner,
    VideoCodec,
    VideoEncodeStats,
    VideoOverlayRenderer,
    __version__,
    build_gpu_gstreamer_pipeline,
    create_preview_app,
    get_camera_profile,
    list_camera_profiles,
    preview,
    serve,
)
from imx_camera_toolkit.api import create_app
from imx_camera_toolkit.camera import Camera
from imx_camera_toolkit.camera_control import CameraController
from imx_camera_toolkit.controls import CameraControls, ExposureConfig
from imx_camera_toolkit.frames import (
    CameraFrameSource,
    CaptureFrameSource,
    FrameSource,
    GpuFrameSource,
)
from imx_camera_toolkit.stream import MJPEGStream
from imx_camera_toolkit.testing import MockCamera, MockFrameSource
from packages.api.api import create_app as InternalCreateApp
from packages.camera.camera import Camera as InternalCamera
from packages.camera.config import CameraConfig as InternalCameraConfig
from packages.camera.errors import (
    CameraConfigurationError as InternalConfigurationError,
)
from packages.camera.errors import CameraDependencyError as InternalDependencyError
from packages.camera.errors import CameraError as InternalCameraError
from packages.camera.errors import CameraOpenError as InternalOpenError
from packages.camera.errors import CameraReadError as InternalReadError
from packages.camera.errors import CameraRecoveryError as InternalRecoveryError
from packages.camera.errors import CameraTimeoutError as InternalTimeoutError
from packages.camera.gpu_camera import GpuCamera as InternalGpuCamera
from packages.camera.models import CameraFrame as InternalCameraFrame
from packages.camera.models import CameraStats as InternalCameraStats
from packages.camera.models import EncodedVideoFrame as InternalEncodedVideoFrame
from packages.camera.models import Frame as InternalFrame
from packages.camera.models import FrameFormat as InternalFrameFormat
from packages.camera.models import GpuBufferHandle as InternalGpuBufferHandle
from packages.camera.models import GpuFrame as InternalGpuFrame
from packages.camera.models import GpuFrameExpiredError as InternalGpuFrameExpiredError
from packages.camera.models import HardwareVideoConfig as InternalHardwareVideoConfig
from packages.camera.models import MemoryType as InternalMemoryType
from packages.camera.models import MetricsRecorder as InternalMetricsRecorder
from packages.camera.models import PipelineMetrics as InternalPipelineMetrics
from packages.camera.models import PipelineStage as InternalPipelineStage
from packages.camera.models import StageMetrics as InternalStageMetrics
from packages.camera.models import VideoCodec as InternalVideoCodec
from packages.camera.models import VideoEncodeStats as InternalVideoEncodeStats
from packages.camera.models import VideoOverlayRenderer as InternalVideoOverlayRenderer
from packages.camera.pipeline import (
    build_gpu_gstreamer_pipeline as internal_build_gpu_gstreamer_pipeline,
)
from packages.camera.profiles import (
    CameraProfile as InternalCameraProfile,
)
from packages.camera.profiles import (
    CameraProfileStatus as InternalCameraProfileStatus,
)
from packages.camera.profiles import (
    get_camera_profile as internal_get_camera_profile,
)
from packages.camera.profiles import (
    list_camera_profiles as internal_list_camera_profiles,
)
from packages.camera_control.camera_control import (
    CameraController as InternalController,
)
from packages.camera_control.camera_control import (
    CameraSettings as InternalCameraSettings,
)
from packages.consumers import FrameConsumer as InternalFrameConsumer
from packages.consumers import InferenceConsumer as InternalInferenceConsumer
from packages.consumers import (
    InferencePreviewSource as InternalInferencePreviewSource,
)
from packages.consumers import InferenceResultSource as InternalInferenceResultSource
from packages.consumers import (
    LatestFrameSubscription as InternalLatestFrameSubscription,
)
from packages.consumers import OverlayRenderer as InternalOverlayRenderer
from packages.consumers import PreviewOverlayContext as InternalPreviewOverlayContext
from packages.frames import CameraFrameSource as InternalCameraFrameSource
from packages.frames import CaptureFrameSource as InternalCaptureFrameSource
from packages.frames import FrameSource as InternalFrameSource
from packages.frames import GpuFrameSource as InternalGpuFrameSource
from packages.inference import FrameSpec as InternalFrameSpec
from packages.inference import InferenceResult as InternalInferenceResult
from packages.inference import InferenceRunner as InternalInferenceRunner
from packages.inference import ShapeProfile as InternalShapeProfile
from packages.inference import TensorOutput as InternalTensorOutput
from packages.inference import TensorRTRunner as InternalTensorRTRunner
from packages.stream.stream import MJPEGStream as InternalMJPEGStream
from packages.testing import MockCamera as InternalMockCamera
from packages.testing import MockFrameSource as InternalMockFrameSource


def test_public_namespace_reexports_stable_library_types() -> None:
    """External imports must resolve to the existing implementation classes."""
    assert __version__ == "0.5.1"
    assert CameraPreview.__module__ == "imx_camera_toolkit.preview"
    assert PreviewServer.__module__ == "packages.preview.server"
    assert PreviewSource.__module__ == "packages.preview.server"
    assert create_preview_app.__module__ == "imx_camera_toolkit.preview"
    assert preview.__module__ == "imx_camera_toolkit.preview"
    assert serve.__module__ == "imx_camera_toolkit.preview"
    assert create_app is InternalCreateApp
    assert RootCamera is InternalCamera
    assert Camera is InternalCamera
    assert CameraConfig is InternalCameraConfig
    assert CameraProfile is InternalCameraProfile
    assert CameraProfileStatus is InternalCameraProfileStatus
    assert CameraStats is InternalCameraStats
    assert CameraDependencyError is InternalDependencyError
    assert CameraError is InternalCameraError
    assert CameraOpenError is InternalOpenError
    assert CameraReadError is InternalReadError
    assert CameraTimeoutError is InternalTimeoutError
    assert EncodedVideoFrame is InternalEncodedVideoFrame
    assert CameraConfigurationError is InternalConfigurationError
    assert CameraRecoveryError is InternalRecoveryError
    assert CameraFrame is InternalCameraFrame
    assert Frame is InternalFrame
    assert FrameConsumer is InternalFrameConsumer
    assert FrameFormat is InternalFrameFormat
    assert FrameSpec is InternalFrameSpec
    assert GpuBufferHandle is InternalGpuBufferHandle
    assert GpuCamera is InternalGpuCamera
    assert GpuFrame is InternalGpuFrame
    assert GpuFrameExpiredError is InternalGpuFrameExpiredError
    assert HardwareVideoConfig is InternalHardwareVideoConfig
    assert InferenceResult is InternalInferenceResult
    assert InferenceConsumer is InternalInferenceConsumer
    assert InferencePreviewSource is InternalInferencePreviewSource
    assert InferenceResultSource is InternalInferenceResultSource
    assert InferenceRunner is InternalInferenceRunner
    assert LatestFrameSubscription is InternalLatestFrameSubscription
    assert MemoryType is InternalMemoryType
    assert MetricsRecorder is InternalMetricsRecorder
    assert PipelineMetrics is InternalPipelineMetrics
    assert PipelineStage is InternalPipelineStage
    assert OverlayRenderer is InternalOverlayRenderer
    assert PreviewOverlayContext is InternalPreviewOverlayContext
    assert StageMetrics is InternalStageMetrics
    assert ShapeProfile is InternalShapeProfile
    assert TensorOutput is InternalTensorOutput
    assert TensorRTRunner is InternalTensorRTRunner
    assert VideoCodec is InternalVideoCodec
    assert VideoEncodeStats is InternalVideoEncodeStats
    assert VideoOverlayRenderer is InternalVideoOverlayRenderer
    assert build_gpu_gstreamer_pipeline is internal_build_gpu_gstreamer_pipeline
    assert get_camera_profile is internal_get_camera_profile
    assert list_camera_profiles is internal_list_camera_profiles
    assert CameraController is InternalController
    assert CameraFrameSource is InternalCameraFrameSource
    assert FrameSource is InternalFrameSource
    assert CaptureFrameSource is InternalCaptureFrameSource
    assert GpuFrameSource is InternalGpuFrameSource
    assert MJPEGStream is InternalMJPEGStream
    assert CameraControls is InternalController
    assert ExposureConfig is InternalCameraSettings
    assert MockCamera is InternalMockCamera
    assert MockFrameSource is InternalMockFrameSource
