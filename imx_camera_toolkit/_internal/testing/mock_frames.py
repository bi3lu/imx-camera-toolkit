"""Deterministic CPU and GPU frame sources for consumer contract tests."""

from __future__ import annotations

import threading
import time
from math import isfinite

from imx_camera_toolkit._internal.camera.models import (
    Frame,
    FrameFormat,
    GpuBufferHandle,
    GpuFrame,
    MemoryType,
)
from imx_camera_toolkit._internal.frames import CaptureFrame


def mock_cpu_frame(
    image: object,
    *,
    sequence: int = 1,
    width: int = 1,
    height: int = 1,
    timestamp_ns: int | None = None,
) -> Frame:
    """Create a legacy-compatible synthetic BGR CPU frame."""
    resolved_timestamp_ns = (
        time.monotonic_ns() if timestamp_ns is None else timestamp_ns
    )
    return Frame(
        image=image,
        sequence=sequence,
        timestamp_ns=resolved_timestamp_ns,
        capture_timestamp_ns=resolved_timestamp_ns,
        width=width,
        height=height,
        format="BGR",
    )


def mock_gpu_frame(
    buffer: object,
    *,
    sequence: int = 1,
    width: int = 1,
    height: int = 1,
    timestamp_ns: int | None = None,
) -> GpuFrame:
    """Create a synthetic borrowed NV12/NVMM frame around an opaque handle."""
    resolved_timestamp_ns = (
        time.monotonic_ns() if timestamp_ns is None else timestamp_ns
    )
    return GpuFrame(
        sequence=sequence,
        timestamp_ns=resolved_timestamp_ns,
        capture_timestamp_ns=resolved_timestamp_ns,
        width=width,
        height=height,
        format=FrameFormat.NV12_NVMM,
        memory_type=MemoryType.NVMM,
        buffer=GpuBufferHandle(buffer),
    )


class MockFrameSource:
    """Thread-safe latest-frame source for CPU and GPU contract tests.

    Publication replaces one retained slot. Replacing a :class:`GpuFrame`
    invalidates its lease before the successor becomes visible.
    """

    def __init__(self, *, auto_start: bool = True) -> None:
        """Initialize an empty source with optional automatic startup."""
        self._condition = threading.Condition()
        self._running = auto_start
        self._frame: CaptureFrame | None = None

    @property
    def running(self) -> bool:
        """Whether publication and blocking reads are active."""
        with self._condition:
            return self._running

    def start(self) -> None:
        """Start the synthetic source."""
        with self._condition:
            self._running = True
            self._condition.notify_all()

    def stop(self) -> None:
        """Stop the source and invalidate any retained GPU lease."""
        with self._condition:
            if isinstance(self._frame, GpuFrame):
                self._frame.invalidate()
            self._frame = None
            self._running = False
            self._condition.notify_all()

    def publish(self, frame: CaptureFrame) -> None:
        """Replace the newest frame without creating a queue."""
        if not isinstance(frame, (Frame, GpuFrame)):
            raise TypeError("frame must be a Frame or GpuFrame")

        with self._condition:
            if not self._running:
                raise RuntimeError("mock frame source is not running")
            if isinstance(self._frame, GpuFrame):
                self._frame.invalidate()
            self._frame = frame
            self._condition.notify_all()

    def read(self, timeout: float | None = None) -> CaptureFrame | None:
        """Return the one newest frame or ``None`` when unavailable."""
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be a finite non-negative number or None")

        with self._condition:
            self._condition.wait_for(
                lambda: self._frame is not None or not self._running,
                timeout=timeout,
            )
            return self._frame
