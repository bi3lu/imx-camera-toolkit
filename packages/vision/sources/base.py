"""Contracts for frame acquisition independent from a camera backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FrameSource(Protocol):
    """Lifecycle-aware provider of image payloads for :class:`VisionPipeline`.

    A source must return ``None`` only when it has no frame available. The
    pipeline checks :attr:`exhausted` to distinguish a finite source completing
    from a temporarily empty live source.
    """

    @property
    def exhausted(self) -> bool:
        """bool: Whether this source cannot provide additional frames."""
        ...

    def open(self) -> None:
        """Allocate source resources and prepare the first frame."""
        ...

    def read(self) -> object | None:
        """Return one source image payload, or ``None`` when unavailable."""
        ...

    def close(self) -> None:
        """Release resources and unblock a pending read when possible."""
        ...


@runtime_checkable
class RawFrameCamera(Protocol):
    """Camera contract required by :class:`CameraFrameSource`.

    The protocol deliberately exposes raw BGR frames instead of JPEG bytes.
    This lets a vision pipeline share camera capture with preview encoding
    without a JPEG decode round trip.
    """

    @property
    def running(self) -> bool:
        """bool: Whether camera capture remains active."""
        ...

    @property
    def raw_frame_number(self) -> int:
        """int: Identifier of the newest raw frame."""
        ...

    def start(self) -> None:
        """Start camera capture when it is not already active."""
        ...

    def stop(self) -> None:
        """Stop camera capture and wake raw-frame consumers."""
        ...

    def wait_for_raw_frame(
        self,
        previous_frame_number: int,
        timeout: float = 2.0,
    ) -> tuple[int, object | None]:
        """Wait for a newer raw frame without copying its payload."""
        ...
