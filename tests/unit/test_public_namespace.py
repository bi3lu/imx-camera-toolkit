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
    Frame,
    PreviewServer,
    PreviewSource,
    __version__,
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
from imx_camera_toolkit.frames import CameraFrameSource, FrameSource
from imx_camera_toolkit.stream import MJPEGStream
from imx_camera_toolkit.testing import MockCamera
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
from packages.camera.models import CameraFrame as InternalCameraFrame
from packages.camera.models import CameraStats as InternalCameraStats
from packages.camera.models import Frame as InternalFrame
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
from packages.frames import CameraFrameSource as InternalCameraFrameSource
from packages.frames import FrameSource as InternalFrameSource
from packages.stream.stream import MJPEGStream as InternalMJPEGStream
from packages.testing import MockCamera as InternalMockCamera


def test_public_namespace_reexports_stable_library_types() -> None:
    """External imports must resolve to the existing implementation classes."""
    assert __version__ == "0.4.0"
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
    assert CameraConfigurationError is InternalConfigurationError
    assert CameraRecoveryError is InternalRecoveryError
    assert CameraFrame is InternalCameraFrame
    assert Frame is InternalFrame
    assert get_camera_profile is internal_get_camera_profile
    assert list_camera_profiles is internal_list_camera_profiles
    assert CameraController is InternalController
    assert CameraFrameSource is InternalCameraFrameSource
    assert FrameSource is InternalFrameSource
    assert MJPEGStream is InternalMJPEGStream
    assert CameraControls is InternalController
    assert ExposureConfig is InternalCameraSettings
    assert MockCamera is InternalMockCamera
