"""Tests for the stable external ``imx_camera_toolkit`` namespace."""

from __future__ import annotations

from imx_camera_toolkit import Camera as RootCamera
from imx_camera_toolkit import (
    CameraDependencyError,
    CameraFrame,
    CameraPreview,
    Frame,
    __version__,
    preview,
)
from imx_camera_toolkit.api import create_app
from imx_camera_toolkit.camera import Camera
from imx_camera_toolkit.camera_control import CameraController
from imx_camera_toolkit.frames import CameraFrameSource, FrameSource
from imx_camera_toolkit.stream import MJPEGStream
from packages.api.api import create_app as InternalCreateApp
from packages.camera.camera import Camera as InternalCamera
from packages.camera.errors import CameraDependencyError as InternalDependencyError
from packages.camera.models import CameraFrame as InternalCameraFrame
from packages.camera.models import Frame as InternalFrame
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
    assert preview.__module__ == "imx_camera_toolkit.preview"
    assert create_app is InternalCreateApp
    assert RootCamera is InternalCamera
    assert Camera is InternalCamera
    assert CameraDependencyError is InternalDependencyError
    assert CameraFrame is InternalCameraFrame
    assert Frame is InternalFrame
    assert CameraController is InternalController
    assert CameraFrameSource is InternalCameraFrameSource
    assert FrameSource is InternalFrameSource
    assert MJPEGStream is InternalMJPEGStream
