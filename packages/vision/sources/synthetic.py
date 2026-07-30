"""Synthetic frame source for deterministic tests and AI-pipeline development."""

from __future__ import annotations

import time
from collections.abc import Callable

FrameFactory = Callable[[int], object]


def _default_frame_factory(index: int) -> object:
    """Create a lightweight default synthetic payload."""
    return {"synthetic_frame": index}


class SyntheticFrameSource:
    """Generate deterministic frames without a sensor, codec, or file.

    Args:
        frame_factory: Function called with a zero-based frame index.
        max_frames: Number of frames to generate. ``None`` generates until the
            source is closed.
        interval: Delay between generated frames, in seconds.
    """

    def __init__(
        self,
        frame_factory: FrameFactory = _default_frame_factory,
        *,
        max_frames: int | None = None,
        interval: float = 0.0,
    ) -> None:
        """Configure a source without starting generation."""
        if max_frames is not None and (
            isinstance(max_frames, bool)
            or not isinstance(max_frames, int)
            or max_frames < 0
        ):
            raise ValueError("max_frames must be a non-negative integer or None")

        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or interval < 0
        ):
            raise ValueError("interval must be a non-negative number")

        self._frame_factory = frame_factory
        self._max_frames = max_frames
        self._interval = interval
        self._index = 0
        self._opened = False
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """bool: Whether the configured finite sequence has completed."""
        return self._exhausted

    def open(self) -> None:
        """Reset the generated sequence for a new pipeline lifecycle."""
        self._index = 0
        self._opened = True
        self._exhausted = self._max_frames == 0

    def read(self) -> object | None:
        """Generate one payload or signal finite-source exhaustion."""
        if not self._opened:
            raise RuntimeError("synthetic frame source is not open")

        if self._exhausted:
            return None

        if self._interval:
            time.sleep(self._interval)

        index = self._index
        self._index += 1
        payload = self._frame_factory(index)

        if self._max_frames is not None and self._index >= self._max_frames:
            self._exhausted = True

        return payload

    def close(self) -> None:
        """Stop further frame generation."""
        self._opened = False
        self._exhausted = True
