"""Minimal frame-source contracts for external processing pipelines."""

from .source import (
    CameraFrameSource,
    CaptureFrame,
    CaptureFrameSource,
    FrameSource,
    GpuFrameSource,
)

__all__ = [
    "CameraFrameSource",
    "CaptureFrame",
    "CaptureFrameSource",
    "FrameSource",
    "GpuFrameSource",
]
