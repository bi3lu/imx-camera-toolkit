"""Frame-source protocols and bundled deterministic source implementations."""

from .base import FrameSource, RawFrameCamera
from .camera import CameraFrameSource
from .file import FileFrameSource, PlaybackMode
from .synthetic import SyntheticFrameSource

__all__ = [
    "FileFrameSource",
    "CameraFrameSource",
    "FrameSource",
    "PlaybackMode",
    "RawFrameCamera",
    "SyntheticFrameSource",
]
