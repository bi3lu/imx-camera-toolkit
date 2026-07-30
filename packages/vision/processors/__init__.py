"""Frame-processing contracts and optional result-rendering components."""

from .base import FrameProcessor, ManagedFrameProcessor, NoopFrameProcessor
from .overlay import OpenCVOverlay, Overlay

__all__ = [
    "FrameProcessor",
    "ManagedFrameProcessor",
    "NoopFrameProcessor",
    "OpenCVOverlay",
    "Overlay",
]
