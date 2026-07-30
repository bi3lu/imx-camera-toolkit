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
