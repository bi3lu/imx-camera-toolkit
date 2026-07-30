"""Composable, latest-frame vision-pipeline primitives for AI Vision workloads."""

from .events import PipelineEvent, PipelineEventType
from .models import BoundingBox, Detection, Frame, InferenceResult, OverlayFrame
from .pipeline import PipelineState, PipelineStats, VisionPipeline
from .processors import FrameProcessor, NoopFrameProcessor, OpenCVOverlay, Overlay
from .sources import FileFrameSource, FrameSource, SyntheticFrameSource

__all__ = [
    "BoundingBox",
    "Detection",
    "FileFrameSource",
    "Frame",
    "FrameProcessor",
    "FrameSource",
    "InferenceResult",
    "NoopFrameProcessor",
    "OpenCVOverlay",
    "Overlay",
    "OverlayFrame",
    "PipelineEvent",
    "PipelineEventType",
    "PipelineState",
    "PipelineStats",
    "SyntheticFrameSource",
    "VisionPipeline",
]
