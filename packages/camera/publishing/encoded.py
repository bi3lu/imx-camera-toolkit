"""Latest-frame publication for JPEG bytes encoded inside GStreamer."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class EncodedJPEGPublisher:
    """Rate-limit publication and retain one newest pre-encoded JPEG."""

    def __init__(self, max_fps: float) -> None:
        """Initialize an empty publication slot."""
        self._interval = 1.0 / max_fps
        self._last_publish_time = 0.0
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_number = 0
        self.frames_encoded = 0
        self.last_frame_time: float | None = None

    @property
    def frame_available(self) -> bool:
        """Whether at least one JPEG has been published."""
        with self._condition:
            return self._jpeg is not None

    @property
    def frame_number(self) -> int:
        """Identifier of the latest encoded preview frame."""
        with self._condition:
            return self._frame_number

    @property
    def jpeg(self) -> bytes | None:
        """Newest JPEG bytes, when available."""
        with self._condition:
            return self._jpeg

    def publish(self, jpeg: bytes) -> bool:
        """Publish non-empty encoded bytes when the rate limit permits."""
        if not isinstance(jpeg, bytes) or not jpeg:
            raise ValueError("jpeg must be non-empty bytes")

        now = time.monotonic()

        if now - self._last_publish_time < self._interval:
            return False

        self._last_publish_time = now
        with self._condition:
            self._jpeg = jpeg
            self._frame_number += 1
            self.frames_encoded += 1
            self.last_frame_time = time.time()
            self._condition.notify_all()
            return True

    def wait_for_jpeg(
        self,
        previous_frame_number: int,
        timeout: float,
        is_running: Callable[[], bool],
    ) -> tuple[int, bytes | None]:
        """Wait for preview bytes newer than a known identifier."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._frame_number != previous_frame_number or not is_running(),
                timeout=timeout,
            )
            return self._frame_number, self._jpeg

    def notify_waiters(self) -> None:
        """Wake preview consumers during shutdown."""
        with self._condition:
            self._condition.notify_all()

    def clear(self) -> None:
        """Discard retained preview bytes."""
        with self._condition:
            self._jpeg = None
