"""Latest-frame publication for JPEG bytes encoded inside GStreamer."""

from __future__ import annotations

import time

from .jpeg_slot import LatestJPEGSlot


class EncodedJPEGPublisher(LatestJPEGSlot):
    """Rate-limit publication and retain one newest pre-encoded JPEG."""

    def __init__(self, max_fps: float) -> None:
        """Initialize an empty publication slot."""
        super().__init__()
        self._interval = 1.0 / max_fps
        self._last_publish_time = 0.0

    def publish(
        self,
        jpeg: bytes,
        *,
        width: int | None = None,
        height: int | None = None,
        quality: int | None = None,
    ) -> bool:
        """Publish detached encoded bytes when the rate limit permits."""
        if not isinstance(jpeg, bytes) or not jpeg:
            raise ValueError("jpeg must be non-empty bytes")

        now = time.monotonic()

        if now - self._last_publish_time < self._interval:
            return False

        self._last_publish_time = now
        self._store(jpeg, width=width, height=height, quality=quality)
        return True
