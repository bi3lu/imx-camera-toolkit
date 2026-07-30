"""Tests for the stable external ``imx_camera_toolkit`` namespace."""

from __future__ import annotations

from imx_camera_toolkit import __version__
from imx_camera_toolkit.api import create_app
from imx_camera_toolkit.camera import Camera
from imx_camera_toolkit.camera_control import CameraController
from imx_camera_toolkit.stream import MJPEGStream
from imx_camera_toolkit.vision import SyntheticFrameSource, VisionPipeline
from packages.api.api import create_app as InternalCreateApp
from packages.camera.camera import Camera as InternalCamera
from packages.camera_control.camera_control import (
    CameraController as InternalController,
)
from packages.stream.stream import MJPEGStream as InternalMJPEGStream
from packages.vision import (
    SyntheticFrameSource as InternalSyntheticFrameSource,
)
from packages.vision import VisionPipeline as InternalVisionPipeline


def test_public_namespace_reexports_stable_library_types() -> None:
    """External imports must resolve to the existing implementation classes."""
    assert __version__ == "0.3.1"
    assert create_app is InternalCreateApp
    assert Camera is InternalCamera
    assert CameraController is InternalController
    assert MJPEGStream is InternalMJPEGStream
    assert SyntheticFrameSource is InternalSyntheticFrameSource
    assert VisionPipeline is InternalVisionPipeline
