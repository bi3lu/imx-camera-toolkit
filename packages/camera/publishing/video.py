"""Latest encoded access unit and throughput publication."""

from __future__ import annotations

import threading
import time
from collections import deque

from ..models.video import EncodedVideoFrame, VideoEncodeStats


class EncodedVideoPublisher:
    """Retain one access unit and recent encode FPS/bitrate aggregates."""

    WINDOW_NS = 1_000_000_000

    def __init__(self) -> None:
        """Initialize an empty encoded-frame slot."""
        self._condition = threading.Condition()
        self._frame: EncodedVideoFrame | None = None
        self._samples: deque[tuple[int, int]] = deque()
        self._encoded_frames = 0
        self._encoded_bytes = 0

    @property
    def latest_frame(self) -> EncodedVideoFrame | None:
        """Newest encoded access unit, if available."""
        with self._condition:
            return self._frame

    def publish(self, frame: EncodedVideoFrame) -> None:
        """Replace the current access unit and record its encoded size."""
        if not isinstance(frame, EncodedVideoFrame):
            raise TypeError("frame must be an EncodedVideoFrame")

        with self._condition:
            self._frame = frame
            self._encoded_frames += 1
            self._encoded_bytes += len(frame.data)
            self._samples.append((frame.timestamp_ns, len(frame.data)))
            self._prune(frame.timestamp_ns)
            self._condition.notify_all()

    def stats(self, now_ns: int | None = None) -> VideoEncodeStats:
        """Return recent one-second encode rates and cumulative counters."""
        with self._condition:
            if now_ns is None:
                now_ns = time.monotonic_ns()

            self._prune(now_ns)
            fps = 0.0
            bitrate = 0.0

            if len(self._samples) >= 2:
                elapsed_ns = self._samples[-1][0] - self._samples[0][0]

                if elapsed_ns > 0:
                    fps = (len(self._samples) - 1) * 1_000_000_000 / elapsed_ns
                    bitrate = (
                        (sum(size for _, size in self._samples) - self._samples[0][1])
                        * 8
                        * 1_000_000_000
                        / elapsed_ns
                    )

            return VideoEncodeStats(
                encoded_frames=self._encoded_frames,
                encoded_bytes=self._encoded_bytes,
                encode_fps=fps,
                bitrate_bps=bitrate,
                last_frame_timestamp_ns=(
                    None if self._frame is None else self._frame.timestamp_ns
                ),
            )

    def clear(self) -> None:
        """Discard the retained access unit and recent-rate window."""
        with self._condition:
            self._frame = None
            self._samples.clear()
            self._condition.notify_all()

    def _prune(self, now_ns: int) -> None:
        """Drop rate samples older than one second."""
        oldest = now_ns - self.WINDOW_NS

        while self._samples and self._samples[0][0] < oldest:
            self._samples.popleft()


__all__ = ["EncodedVideoPublisher"]
