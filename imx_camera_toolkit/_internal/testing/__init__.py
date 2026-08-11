"""Deterministic test doubles for toolkit integrations."""

from .mock_camera import MockCamera
from .mock_frames import MockFrameSource, mock_cpu_frame, mock_gpu_frame

__all__ = ["MockCamera", "MockFrameSource", "mock_cpu_frame", "mock_gpu_frame"]
