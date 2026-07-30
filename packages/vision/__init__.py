"""Composable, latest-frame vision-pipeline primitives for AI Vision workloads."""

from .events import EventBus, EventDispatchMode, PipelineEvent, PipelineEventType
from .models import BoundingBox, Detection, Frame, InferenceResult, OverlayFrame
from .pipeline import PipelineState, PipelineStats, VisionPipeline
from .processors import (
    FrameProcessor,
    ManagedFrameProcessor,
    NoopFrameProcessor,
    OpenCVOverlay,
    Overlay,
)
from .sources import (
    CameraFrameSource,
    FileFrameSource,
    FrameSource,
    PlaybackMode,
    RawFrameCamera,
    SyntheticFrameSource,
)

__all__ = [
    "BoundingBox",
    "CameraFrameSource",
    "Detection",
    "EventBus",
    "EventDispatchMode",
    "FileFrameSource",
    "Frame",
    "FrameProcessor",
    "FrameSource",
    "InferenceResult",
    "ManagedFrameProcessor",
    "NoopFrameProcessor",
    "OpenCVOverlay",
    "Overlay",
    "OverlayFrame",
    "PlaybackMode",
    "PipelineEvent",
    "PipelineEventType",
    "PipelineState",
    "PipelineStats",
    "SyntheticFrameSource",
    "RawFrameCamera",
    "VisionPipeline",
]
