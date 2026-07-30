"""Latest raw-frame publication for local AI Vision consumers."""

from __future__ import annotations

import threading
from collections.abc import Callable


class RawFramePublisher:
    """Retain one processed BGR frame without copying or encoding it.

    The publisher owns only the reference to the newest camera frame. It does
    not mutate or copy the payload, so consumers must treat returned frames as
    read-only. This avoids a BGR-to-JPEG-to-BGR round trip when AI Vision and an
    MJPEG preview use the same camera capture loop.
    """

    def __init__(self) -> None:
        """Initialize an empty raw-frame slot."""
        self._condition = threading.Condition()
        self._frame: object | None = None
        self._frame_number = 0

    @property
    def frame(self) -> object | None:
        """object | None: Newest processed BGR frame without a copy."""
        with self._condition:
            return self._frame

    @property
    def frame_number(self) -> int:
        """int: Monotonically increasing identifier of the newest raw frame."""
        with self._condition:
            return self._frame_number

    def publish(self, frame: object) -> int:
        """Publish a processed BGR frame and wake waiting consumers.

        Args:
            frame: Frame payload retained by reference without mutation or copy.

        Returns:
            Identifier assigned to the published raw frame.
        """
        with self._condition:
            self._frame = frame
            self._frame_number += 1
            self._condition.notify_all()
            return self._frame_number

    def wait_for_frame(
        self,
        previous_frame_number: int,
        timeout: float,
        is_running: Callable[[], bool],
    ) -> tuple[int, object | None]:
        """Wait for a raw frame newer than a known identifier.

        Args:
            previous_frame_number: Identifier already consumed by the caller.
            timeout: Maximum wait time in seconds.
            is_running: Callable reporting whether camera capture remains active.

        Returns:
            Newest frame identifier and frame payload, when available.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: self._frame_number != previous_frame_number
                or not is_running(),
                timeout=timeout,
            )
            return self._frame_number, self._frame

    def notify_waiters(self) -> None:
        """Wake consumers after camera shutdown or a capture failure."""
        with self._condition:
            self._condition.notify_all()

    def clear(self) -> None:
        """Discard the retained raw frame while keeping its sequence monotonic."""
        with self._condition:
            self._frame = None
