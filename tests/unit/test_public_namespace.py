"""Tests for the stable external ``imx_camera_toolkit`` namespace."""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import pytest

import imx_camera_toolkit
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
    PlatformCapabilities,
    PreviewBackend,
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
from imx_camera_toolkit._internal.api.api import create_app as InternalCreateApp
from imx_camera_toolkit._internal.camera.camera import Camera as InternalCamera
from imx_camera_toolkit._internal.camera.config import (
    CameraConfig as InternalCameraConfig,
)
from imx_camera_toolkit._internal.camera.errors import (
    CameraConfigurationError as InternalConfigurationError,
)
from imx_camera_toolkit._internal.camera.errors import (
    CameraDependencyError as InternalDependencyError,
)
from imx_camera_toolkit._internal.camera.errors import (
    CameraError as InternalCameraError,
)
from imx_camera_toolkit._internal.camera.errors import (
    CameraOpenError as InternalOpenError,
)
from imx_camera_toolkit._internal.camera.errors import (
    CameraReadError as InternalReadError,
)
from imx_camera_toolkit._internal.camera.errors import (
    CameraRecoveryError as InternalRecoveryError,
)
from imx_camera_toolkit._internal.camera.errors import (
    CameraTimeoutError as InternalTimeoutError,
)
from imx_camera_toolkit._internal.camera.gpu_camera import (
    GpuCamera as InternalGpuCamera,
)
from imx_camera_toolkit._internal.camera.models import (
    CameraFrame as InternalCameraFrame,
)
from imx_camera_toolkit._internal.camera.models import (
    CameraStats as InternalCameraStats,
)
from imx_camera_toolkit._internal.camera.models import (
    EncodedVideoFrame as InternalEncodedVideoFrame,
)
from imx_camera_toolkit._internal.camera.models import Frame as InternalFrame
from imx_camera_toolkit._internal.camera.models import (
    FrameFormat as InternalFrameFormat,
)
from imx_camera_toolkit._internal.camera.models import (
    GpuBufferHandle as InternalGpuBufferHandle,
)
from imx_camera_toolkit._internal.camera.models import GpuFrame as InternalGpuFrame
from imx_camera_toolkit._internal.camera.models import (
    GpuFrameExpiredError as InternalGpuFrameExpiredError,
)
from imx_camera_toolkit._internal.camera.models import (
    HardwareVideoConfig as InternalHardwareVideoConfig,
)
from imx_camera_toolkit._internal.camera.models import MemoryType as InternalMemoryType
from imx_camera_toolkit._internal.camera.models import (
    MetricsRecorder as InternalMetricsRecorder,
)
from imx_camera_toolkit._internal.camera.models import (
    PipelineMetrics as InternalPipelineMetrics,
)
from imx_camera_toolkit._internal.camera.models import (
    PipelineStage as InternalPipelineStage,
)
from imx_camera_toolkit._internal.camera.models import (
    StageMetrics as InternalStageMetrics,
)
from imx_camera_toolkit._internal.camera.models import VideoCodec as InternalVideoCodec
from imx_camera_toolkit._internal.camera.models import (
    VideoEncodeStats as InternalVideoEncodeStats,
)
from imx_camera_toolkit._internal.camera.models import (
    VideoOverlayRenderer as InternalVideoOverlayRenderer,
)
from imx_camera_toolkit._internal.camera.pipeline import (
    build_gpu_gstreamer_pipeline as internal_build_gpu_gstreamer_pipeline,
)
from imx_camera_toolkit._internal.camera.platform_capabilities import (
    PlatformCapabilities as InternalPlatformCapabilities,
)
from imx_camera_toolkit._internal.camera.profiles import (
    CameraProfile as InternalCameraProfile,
)
from imx_camera_toolkit._internal.camera.profiles import (
    CameraProfileStatus as InternalCameraProfileStatus,
)
from imx_camera_toolkit._internal.camera.profiles import (
    get_camera_profile as internal_get_camera_profile,
)
from imx_camera_toolkit._internal.camera.profiles import (
    list_camera_profiles as internal_list_camera_profiles,
)
from imx_camera_toolkit._internal.camera_control.camera_control import (
    CameraController as InternalController,
)
from imx_camera_toolkit._internal.camera_control.camera_control import (
    CameraSettings as InternalCameraSettings,
)
from imx_camera_toolkit._internal.consumers import (
    FrameConsumer as InternalFrameConsumer,
)
from imx_camera_toolkit._internal.consumers import (
    InferenceConsumer as InternalInferenceConsumer,
)
from imx_camera_toolkit._internal.consumers import (
    InferencePreviewSource as InternalInferencePreviewSource,
)
from imx_camera_toolkit._internal.consumers import (
    InferenceResultSource as InternalInferenceResultSource,
)
from imx_camera_toolkit._internal.consumers import (
    LatestFrameSubscription as InternalLatestFrameSubscription,
)
from imx_camera_toolkit._internal.consumers import (
    OverlayRenderer as InternalOverlayRenderer,
)
from imx_camera_toolkit._internal.consumers import (
    PreviewOverlayContext as InternalPreviewOverlayContext,
)
from imx_camera_toolkit._internal.frames import (
    CameraFrameSource as InternalCameraFrameSource,
)
from imx_camera_toolkit._internal.frames import (
    CaptureFrameSource as InternalCaptureFrameSource,
)
from imx_camera_toolkit._internal.frames import FrameSource as InternalFrameSource
from imx_camera_toolkit._internal.frames import GpuFrameSource as InternalGpuFrameSource
from imx_camera_toolkit._internal.inference import FrameSpec as InternalFrameSpec
from imx_camera_toolkit._internal.inference import (
    InferenceResult as InternalInferenceResult,
)
from imx_camera_toolkit._internal.inference import (
    InferenceRunner as InternalInferenceRunner,
)
from imx_camera_toolkit._internal.inference import ShapeProfile as InternalShapeProfile
from imx_camera_toolkit._internal.inference import TensorOutput as InternalTensorOutput
from imx_camera_toolkit._internal.inference import (
    TensorRTRunner as InternalTensorRTRunner,
)
from imx_camera_toolkit._internal.stream.stream import (
    MJPEGStream as InternalMJPEGStream,
)
from imx_camera_toolkit._internal.testing import MockCamera as InternalMockCamera
from imx_camera_toolkit._internal.testing import (
    MockFrameSource as InternalMockFrameSource,
)
from imx_camera_toolkit.api import APIConfig, create_app, load_api_config
from imx_camera_toolkit.camera import Camera
from imx_camera_toolkit.camera_control import CameraController
from imx_camera_toolkit.controls import CameraControls, ExposureConfig
from imx_camera_toolkit.frames import (
    CameraFrameSource,
    CaptureFrameSource,
    FrameSource,
    GpuFrameSource,
)
from imx_camera_toolkit.preview import PreviewBackend as ModulePreviewBackend
from imx_camera_toolkit.stream import MJPEGStream
from imx_camera_toolkit.testing import MockCamera, MockFrameSource


def test_public_namespace_reexports_stable_library_types() -> None:
    """External imports must resolve to the existing implementation classes."""
    assert __version__
    assert __version__ == importlib.metadata.version("imx-camera-toolkit")
    assert CameraPreview.__module__ == "imx_camera_toolkit.preview"
    assert PreviewBackend is ModulePreviewBackend
    assert PreviewServer.__module__ == "imx_camera_toolkit._internal.preview.server"
    assert PreviewSource.__module__ == "imx_camera_toolkit._internal.preview.server"
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
    assert PlatformCapabilities is InternalPlatformCapabilities
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


def test_public_version_uses_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime version must follow metadata rather than a duplicated literal."""

    def version(distribution_name: str) -> str:
        """Return a distinct version for the expected distribution only."""
        assert distribution_name == "imx-camera-toolkit"
        return "1.2.3.dev4"

    monkeypatch.setattr(importlib.metadata, "version", version)
    spec = importlib.util.spec_from_file_location(
        "imx_camera_toolkit", imx_camera_toolkit.__file__
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.__version__ == "1.2.3.dev4"


def test_api_version_defaults_to_metadata_and_allows_override(tmp_path: Path) -> None:
    """Default API metadata follows the package; explicit versions still work."""
    assert APIConfig().version == __version__
    assert load_api_config().version == __version__
    config_path = tmp_path / "api.yml"
    config_path.write_text('api_config:\n  version: "custom-api-version"\n')
    assert load_api_config(config_path, strict=True).version == "custom-api-version"
