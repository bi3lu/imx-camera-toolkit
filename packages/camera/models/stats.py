"""Immutable diagnostics returned by the camera capture API."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from .metrics import PipelineMetrics


@dataclass(frozen=True, slots=True)
class CameraStats:
    """Point-in-time capture diagnostics without telemetry dependencies.

    Attributes:
        captured_frames: Number of successful source reads since construction.
        dropped_frames: Frames or read attempts that were not published as raw
            frames, including unsuccessful source reads and processor drops.
        capture_fps: Recent successful source-read rate in frames per second.
        last_frame_timestamp_ns: Monotonic timestamp of the latest successful
            source read, or ``None`` before the first frame.
        recovery_count: Number of successful backend recovery operations.
        consecutive_failures: Current uninterrupted source-read failure count.
        running: Whether the capture worker is currently active.
    """

    captured_frames: int
    dropped_frames: int
    capture_fps: float
    last_frame_timestamp_ns: int | None
    recovery_count: int
    consecutive_failures: int
    running: bool
    pipeline: PipelineMetrics = field(default_factory=PipelineMetrics)
    consumer_dropped_frames: tuple[tuple[str, int], ...] = ()
    last_capture_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        """Validate scalar diagnostics without inspecting camera resources."""
        integer_fields = (
            "captured_frames",
            "dropped_frames",
            "recovery_count",
            "consecutive_failures",
        )

        for field_name in integer_fields:
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

        if (
            isinstance(self.capture_fps, bool)
            or not isinstance(self.capture_fps, (int, float))
            or not isfinite(self.capture_fps)
            or self.capture_fps < 0
        ):
            raise ValueError("capture_fps must be a finite non-negative number")

        if self.last_frame_timestamp_ns is not None and (
            isinstance(self.last_frame_timestamp_ns, bool)
            or not isinstance(self.last_frame_timestamp_ns, int)
            or self.last_frame_timestamp_ns < 0
        ):
            raise ValueError(
                "last_frame_timestamp_ns must be a non-negative integer or None"
            )

        if self.last_capture_timestamp_ns is not None and (
            isinstance(self.last_capture_timestamp_ns, bool)
            or not isinstance(self.last_capture_timestamp_ns, int)
            or self.last_capture_timestamp_ns < 0
        ):
            raise ValueError(
                "last_capture_timestamp_ns must be a non-negative integer or None"
            )

        if not isinstance(self.running, bool):
            raise ValueError("running must be a boolean")

        if not isinstance(self.pipeline, PipelineMetrics):
            raise ValueError("pipeline must be a PipelineMetrics snapshot")

        consumer_names: set[str] = set()

        for item in self.consumer_dropped_frames:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("consumer_dropped_frames must contain pairs")

            consumer, count = item

            if not isinstance(consumer, str) or not consumer.strip():
                raise ValueError("consumer names must be non-empty strings")

            if consumer in consumer_names:
                raise ValueError("consumer names must be unique")

            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("consumer drop counts must be non-negative integers")

            consumer_names.add(consumer)
