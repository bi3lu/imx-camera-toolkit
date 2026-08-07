"""Single-slot publication for borrowed NVMM frames."""

from __future__ import annotations

import threading
from collections.abc import Callable

from ..models import GpuFrame


class GpuFramePublisher:
    """Retain only the newest GPU lease and invalidate its predecessor."""

    def __init__(self) -> None:
        """Initialize an empty latest-frame slot."""
        self._condition = threading.Condition()
        self._frame: GpuFrame | None = None

    @property
    def latest_frame(self) -> GpuFrame | None:
        """Newest valid borrowed GPU frame, when available."""
        with self._condition:
            return self._frame

    @property
    def frame_number(self) -> int:
        """Sequence of the newest frame, or zero before publication."""
        with self._condition:
            return self._frame.sequence if self._frame is not None else 0

    def publish(self, frame: GpuFrame) -> None:
        """Replace the latest frame after invalidating the previous lease."""
        if not isinstance(frame, GpuFrame):
            raise TypeError("frame must be a GpuFrame")

        with self._condition:
            previous = self._frame
            if previous is not None:
                previous.invalidate()

            self._frame = frame
            self._condition.notify_all()

    def wait_for_frame(
        self,
        previous_frame_number: int,
        timeout: float,
        is_running: Callable[[], bool],
    ) -> GpuFrame | None:
        """Wait for a sequence newer than ``previous_frame_number``."""
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._frame is not None
                    and self._frame.sequence != previous_frame_number
                )
                or not is_running(),
                timeout=timeout,
            )
            return self._frame

    def notify_waiters(self) -> None:
        """Wake readers during shutdown or terminal recovery failure."""
        with self._condition:
            self._condition.notify_all()

    def clear(self) -> None:
        """Invalidate and discard the retained GPU lease."""
        with self._condition:
            if self._frame is not None:
                self._frame.invalidate()

            self._frame = None
            self._condition.notify_all()
