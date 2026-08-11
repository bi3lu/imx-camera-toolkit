"""Frame-source protocol and adapter for the toolkit camera."""

from __future__ import annotations

from math import isfinite
from typing import Protocol, TypeAlias, runtime_checkable

from imx_camera_toolkit._internal.camera.models import Frame, GpuFrame

CaptureFrame: TypeAlias = Frame | GpuFrame


@runtime_checkable
class FrameSource(Protocol):
    """Minimal provider of the newest available camera frame.

    A source neither owns inference nor dictates batching, threading, tracking,
    transport, or lifecycle policies. External applications decide how each
    returned :class:`Frame` is processed.
    """

    def read(self, timeout: float | None = None) -> Frame | None:
        """Return the newest frame, or ``None`` when none is available.

        Args:
            timeout: Maximum wait in seconds. ``None`` uses the source default.

        Returns:
            Newest available frame, or ``None`` after the timeout.
        """
        ...


@runtime_checkable
class GpuFrameSource(Protocol):
    """Provider of one newest borrowed GPU frame at a time.

    Calling :meth:`read` may invalidate the frame returned by the preceding
    call. Consumers must therefore finish GPU work that uses a frame before
    requesting its successor.
    """

    def read(self, timeout: float | None = None) -> GpuFrame | None:
        """Return the newest borrowed GPU frame, or ``None`` on timeout."""
        ...


@runtime_checkable
class CaptureFrameSource(Protocol):
    """Model-agnostic source that may return CPU or GPU frames."""

    def read(self, timeout: float | None = None) -> CaptureFrame | None:
        """Return the newest frame in the source's declared memory domain."""
        ...


class _CameraReader(Protocol):
    """Subset of the camera contract needed by :class:`CameraFrameSource`."""

    def read(self, timeout: float = 2.0, copy: bool = True) -> Frame | None:
        """Return the newest camera frame with caller-selected ownership."""
        ...


class CameraFrameSource:
    """Expose an existing camera as a minimal :class:`FrameSource`.

    The adapter does not start or stop the camera. Applications retain camera
    lifecycle ownership, typically through ``with Camera() as camera``. By
    default it forwards the shared raw image with ``copy=False``; consumers
    must therefore treat ``Frame.image`` as read-only.

    Args:
        camera: Started camera exposing the raw ``read`` contract.
        copy: Whether each source read should copy the camera image payload.
    """

    def __init__(self, camera: _CameraReader, *, copy: bool = False) -> None:
        """Store an existing camera without taking ownership of its lifecycle."""
        if not isinstance(copy, bool):
            raise ValueError("copy must be a boolean")

        self._camera = camera
        self._copy = copy

    def read(self, timeout: float | None = None) -> Frame | None:
        """Return one newest raw camera frame without JPEG decoding.

        Args:
            timeout: Maximum wait in seconds. ``None`` uses the camera default.

        Returns:
            Newest available raw frame, or ``None`` when unavailable.

        Raises:
            ValueError: If ``timeout`` is not ``None`` or a non-negative number.
        """
        if timeout is None:
            return self._camera.read(copy=self._copy)

        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a non-negative number or None")

        if not isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be a non-negative number or None")

        return self._camera.read(timeout=timeout, copy=self._copy)
