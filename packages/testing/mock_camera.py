"""Thread-safe in-memory camera used by tests, diagnostics, and benchmarks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from packages.camera.models import CameraStats


@dataclass
class MockCamera:
    """In-memory JPEG camera implementing the stream and API camera contract.

    Args:
        start_error: Optional exception raised when ``start()`` is called.
        auto_start: Whether the mock should be started during initialization.
    """

    start_error: Exception | None = None
    auto_start: bool = True
    frames_captured: int = 0
    dropped_frames: int = 0
    frames_encoded: int = 0
    last_frame_time: float | None = None
    last_error: Exception | None = None
    recovery_attempts: int = 0
    recoveries: int = 0
    consecutive_failures: int = 0
    last_recovery_error: Exception | None = None
    _running: bool = field(default=False, init=False, repr=False)
    _jpeg: bytes | None = field(default=None, init=False, repr=False)
    _frame_number: int = field(default=0, init=False, repr=False)
    _last_frame_timestamp_ns: int | None = field(default=None, init=False, repr=False)
    _condition: threading.Condition = field(
        default_factory=threading.Condition,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Optionally start the in-memory camera."""
        if self.auto_start:
            self.start()

    @property
    def running(self) -> bool:
        """bool: Whether the mock camera is active."""
        with self._condition:
            return self._running

    @property
    def frame_available(self) -> bool:
        """bool: Whether a JPEG frame has been published."""
        with self._condition:
            return self._jpeg is not None

    @property
    def frame_number(self) -> int:
        """int: Identifier of the latest published JPEG frame."""
        with self._condition:
            return self._frame_number

    @property
    def jpeg(self) -> bytes | None:
        """bytes | None: Latest JPEG bytes, or ``None`` when unavailable."""
        with self._condition:
            return self._jpeg

    def stats(self) -> CameraStats:
        """Return stable diagnostics compatible with the production camera."""
        with self._condition:
            return CameraStats(
                captured_frames=self.frames_captured,
                dropped_frames=self.dropped_frames,
                capture_fps=0.0,
                last_frame_timestamp_ns=self._last_frame_timestamp_ns,
                recovery_count=self.recoveries,
                consecutive_failures=self.consecutive_failures,
                running=self._running,
            )

    def start(self) -> None:
        """Start the mock camera or raise its configured startup error."""
        if self.start_error is not None:
            self.last_error = self.start_error
            raise self.start_error

        with self._condition:
            self._running = True
            self.last_error = None
            self._condition.notify_all()

    def stop(self) -> None:
        """Stop the mock camera and wake blocked consumers."""
        with self._condition:
            self._running = False
            self._condition.notify_all()

    def publish_jpeg(self, jpeg: bytes) -> int:
        """Publish one deterministic JPEG payload.

        Args:
            jpeg: Non-empty bytes accepted as a JPEG payload by stream tests.

        Returns:
            Identifier assigned to the published frame.

        Raises:
            RuntimeError: If the mock is stopped.
            ValueError: If ``jpeg`` is empty.
        """
        if not jpeg:
            raise ValueError("jpeg must not be empty")

        with self._condition:
            if not self._running:
                raise RuntimeError("mock camera is not running")

            self._jpeg = jpeg
            self._frame_number += 1
            self.frames_captured += 1
            self._last_frame_timestamp_ns = time.monotonic_ns()
            self.frames_encoded += 1
            self.last_frame_time = time.time()
            self._condition.notify_all()
            return self._frame_number

    def wait_for_jpeg(
        self,
        previous_frame_number: int,
        timeout: float = 2.0,
    ) -> tuple[int, bytes | None]:
        """Wait for a frame newer than ``previous_frame_number``.

        Args:
            previous_frame_number: Frame identifier already consumed.
            timeout: Maximum time to wait, in seconds.

        Returns:
            Latest frame number and JPEG payload, if one has been published.
        """
        if timeout < 0:
            raise ValueError("timeout must be greater than or equal to zero")

        with self._condition:
            self._condition.wait_for(
                lambda: self._frame_number != previous_frame_number
                or not self._running,
                timeout=timeout,
            )
            return self._frame_number, self._jpeg
