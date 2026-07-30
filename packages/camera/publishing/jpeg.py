"""Latest-frame JPEG publication and synchronization."""

from __future__ import annotations

import importlib
import threading
import time
from typing import Any

from ..errors import CameraDependencyError

try:
    cv2_module: Any | None = importlib.import_module("cv2")

except ImportError:
    cv2_module = None


def opencv_available() -> bool:
    """Return whether JPEG encoding support is available."""
    return cv2_module is not None


class JPEGPublisher:
    """Rate-limit JPEG encoding and retain only the newest encoded frame."""

    def __init__(self, quality: int, max_fps: float) -> None:
        """Initialize an empty publisher without encoding a frame."""
        self._quality = quality
        self._interval = 1.0 / max_fps
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_number = 0
        self._last_encode_time = 0.0
        self.frames_encoded = 0
        self.last_frame_time: float | None = None

    @property
    def frame_available(self) -> bool:
        """bool: Whether a JPEG frame has been encoded."""
        with self._condition:
            return self._jpeg is not None

    @property
    def frame_number(self) -> int:
        """int: Identifier of the newest JPEG frame."""
        with self._condition:
            return self._frame_number

    @property
    def jpeg(self) -> bytes | None:
        """bytes | None: Latest JPEG frame, or ``None`` when unavailable."""
        with self._condition:
            return self._jpeg

    def publish(self, frame: Any) -> bool:
        """Encode and publish a BGR frame when the rate limit permits it.

        Returns:
            ``True`` when a JPEG frame was encoded and published.
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

        with self._condition:
            self._jpeg = encoded.tobytes()
            self._frame_number += 1
            self._condition.notify_all()

        self.frames_encoded += 1
        self.last_frame_time = time.time()
        return True

    def wait_for_jpeg(
        self,
        previous_frame_number: int,
        timeout: float,
        is_running: Any,
    ) -> tuple[int, bytes | None]:
        """Wait for a JPEG newer than a known frame number.

        Args:
            previous_frame_number: Identifier already consumed by the caller.
            timeout: Maximum time to wait, in seconds.
            is_running: Callable that returns whether capture remains active.

        Returns:
            The newest frame number and its JPEG bytes, when available.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: self._frame_number != previous_frame_number or not is_running(),
                timeout=timeout,
            )
            return self._frame_number, self._jpeg

    def notify_waiters(self) -> None:
        """Wake blocked consumers after camera shutdown or a capture failure."""
        with self._condition:
            self._condition.notify_all()

    def clear(self) -> None:
        """Discard the retained JPEG frame."""
        with self._condition:
            self._jpeg = None
