"""Coherent, detached JPEG publications shared by preview consumers."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class JPEGPublication:
    """One immutable JPEG with camera-local sequence and optional provenance."""

    data: bytes
    sequence: int
    width: int | None
    height: int | None
    quality: int | None
    acquired_at_ns: int | None


@dataclass(frozen=True, slots=True)
class JPEGPreview:
    """Point-in-time sample of a fixed preview source and its latest JPEG."""

    state: str
    enabled: bool
    acquisition_timing: bool
    width: int
    height: int
    quality: int | None
    publication: JPEGPublication | None


class LatestJPEGSlot:
    """Store one detached JPEG, with legacy preview properties and waiting."""

    def __init__(self) -> None:
        """Initialize an empty slot with a camera-lifetime sequence."""
        self._condition = threading.Condition()
        self._publication: JPEGPublication | None = None
        self._frame_number = 0
        self.frames_encoded = 0
        self.last_frame_time: float | None = None

    @property
    def publication(self) -> JPEGPublication | None:
        """Return the latest immutable record without waiting for capture."""
        with self._condition:
            return self._publication

    @property
    def frame_available(self) -> bool:
        """Return whether a JPEG is retained."""
        return self.publication is not None

    @property
    def frame_number(self) -> int:
        """Return the most recent sequence, including after a clear."""
        with self._condition:
            return self._frame_number

    @property
    def jpeg(self) -> bytes | None:
        """Return the latest encoded bytes, when available."""
        publication = self.publication
        return publication.data if publication is not None else None

    def _store(
        self,
        jpeg: bytes,
        *,
        width: int | None = None,
        height: int | None = None,
        quality: int | None = None,
        acquired_at_ns: int | None = None,
    ) -> None:
        """Replace the slot and wake legacy preview waiters."""
        with self._condition:
            self._frame_number += 1
            self._publication = JPEGPublication(
                jpeg, self._frame_number, width, height, quality, acquired_at_ns
            )
            self.frames_encoded += 1
            self.last_frame_time = time.time()
            self._condition.notify_all()

    def wait_for_jpeg(
        self,
        previous_frame_number: int,
        timeout: float,
        is_running: Callable[[], bool],
    ) -> tuple[int, bytes | None]:
        """Wait for preview bytes newer than a known identifier or for shutdown."""
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._publication is not None
                    and self._frame_number != previous_frame_number
                )
                or not is_running(),
                timeout=timeout,
            )
            publication = self._publication
            return self._frame_number, publication.data if publication else None

    def notify_waiters(self) -> None:
        """Wake legacy preview consumers during shutdown."""
        with self._condition:
            self._condition.notify_all()

    def clear(self) -> None:
        """Discard the retained record without resetting the sequence."""
        with self._condition:
            self._publication = None
