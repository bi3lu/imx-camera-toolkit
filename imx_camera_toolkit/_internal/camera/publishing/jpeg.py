"""Latest-frame JPEG publication and synchronization."""

from __future__ import annotations

import importlib
import time
from typing import Any

from ..errors import CameraDependencyError
from .jpeg_slot import LatestJPEGSlot

try:
    cv2_module: Any | None = importlib.import_module("cv2")

except ImportError:
    cv2_module = None


def opencv_available() -> bool:
    """Return whether JPEG encoding support is available."""
    return cv2_module is not None


class JPEGPublisher(LatestJPEGSlot):
    """Rate-limit JPEG encoding and retain only the newest encoded frame."""

    def __init__(self, quality: int, max_fps: float) -> None:
        """Initialize an empty publisher without encoding a frame."""
        super().__init__()
        self._quality = quality
        self._interval = 1.0 / max_fps
        self._last_encode_time = 0.0

    def publish(
        self,
        frame: Any,
        *,
        width: int | None = None,
        height: int | None = None,
        acquired_at_ns: int | None = None,
    ) -> bool:
        """Encode a BGR frame with its original acquisition provenance.

        Returns:
            Whether a JPEG frame was encoded and published.
        """
        if cv2_module is None:
            raise CameraDependencyError(
                "System OpenCV with GStreamer support is required."
            )

        now = time.monotonic()

        if now - self._last_encode_time < self._interval:
            return False

        self._last_encode_time = now

        success, encoded = cv2_module.imencode(
            ".jpg",
            frame,
            [cv2_module.IMWRITE_JPEG_QUALITY, self._quality],
        )
        if not success:
            return False

        self._store(
            encoded.tobytes(),
            width=width,
            height=height,
            quality=self._quality,
            acquired_at_ns=acquired_at_ns,
        )
        return True
