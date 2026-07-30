"""Tests for the stable external ``imx_camera_toolkit`` namespace."""

from __future__ import annotations

from imx_camera_toolkit import Camera as RootCamera
from imx_camera_toolkit import (
    CameraConfig,
    CameraDependencyError,
    CameraFrame,
    CameraPreview,
    CameraProfile,
    CameraProfileStatus,
    CameraStats,
    Frame,
    PreviewServer,
    PreviewSource,
    __version__,
    create_preview_app,
    get_camera_profile,
    list_camera_profiles,
    preview,
)
from imx_camera_toolkit.api import create_app
from imx_camera_toolkit.camera import Camera
from imx_camera_toolkit.camera_control import CameraController
from imx_camera_toolkit.frames import CameraFrameSource, FrameSource
from imx_camera_toolkit.stream import MJPEGStream
from packages.api.api import create_app as InternalCreateApp
from packages.camera.camera import Camera as InternalCamera
from packages.camera.config import CameraConfig as InternalCameraConfig
from packages.camera.errors import CameraDependencyError as InternalDependencyError
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
from packages.frames import CameraFrameSource as InternalCameraFrameSource
from packages.frames import FrameSource as InternalFrameSource
from packages.stream.stream import MJPEGStream as InternalMJPEGStream


def test_public_namespace_reexports_stable_library_types() -> None:
    """External imports must resolve to the existing implementation classes."""
    assert __version__ == "0.3.1"
    assert CameraPreview.__module__ == "imx_camera_toolkit.preview"
    assert PreviewServer.__module__ == "packages.preview.server"
    assert PreviewSource.__module__ == "packages.preview.server"
    assert create_preview_app.__module__ == "imx_camera_toolkit.preview"
    assert preview.__module__ == "imx_camera_toolkit.preview"
    assert create_app is InternalCreateApp
    assert RootCamera is InternalCamera
    assert Camera is InternalCamera
    assert CameraConfig is InternalCameraConfig
    assert CameraProfile is InternalCameraProfile
    assert CameraProfileStatus is InternalCameraProfileStatus
    assert CameraStats is InternalCameraStats
    assert CameraDependencyError is InternalDependencyError
    assert CameraFrame is InternalCameraFrame
    assert Frame is InternalFrame
    assert get_camera_profile is internal_get_camera_profile
    assert list_camera_profiles is internal_list_camera_profiles
    assert CameraController is InternalController
    assert CameraFrameSource is InternalCameraFrameSource
    assert FrameSource is InternalFrameSource
    assert MJPEGStream is InternalMJPEGStream
