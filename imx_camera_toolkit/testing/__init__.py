"""Stable deterministic test doubles for external integrations."""

from imx_camera_toolkit._internal.testing import (
    MockCamera,
    MockFrameSource,
    mock_cpu_frame,
    mock_gpu_frame,
)

__all__ = ["MockCamera", "MockFrameSource", "mock_cpu_frame", "mock_gpu_frame"]
