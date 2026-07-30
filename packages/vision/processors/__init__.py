"""Frame-processing contracts and optional result-rendering components."""

from .base import FrameProcessor, NoopFrameProcessor
from .overlay import OpenCVOverlay, Overlay

__all__ = ["FrameProcessor", "NoopFrameProcessor", "OpenCVOverlay", "Overlay"]
