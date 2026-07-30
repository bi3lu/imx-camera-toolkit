"""Raw camera-frame model for external image-processing consumers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One processed BGR camera frame without JPEG encoding or inference.

    The image payload is intentionally opaque so callers can pass it directly
    to OpenCV, TensorRT, DeepStream, CUDA, or another image-processing system.
    The model is shallowly immutable: ``image`` is retained by reference when
    ``Camera.read(copy=False)`` is used.

    Args:
        sequence: Monotonically increasing identifier assigned during capture.
        image: Processed BGR image payload.
        captured_at: Monotonic timestamp recorded when the source frame was
            acquired.
    """

    sequence: int
    image: object
    captured_at: float

    def __post_init__(self) -> None:
        """Validate the frame identity and timestamp."""
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise ValueError("sequence must be a positive integer")

        if (
            isinstance(self.captured_at, bool)
            or not isinstance(self.captured_at, (int, float))
            or not math.isfinite(self.captured_at)
            or self.captured_at < 0
        ):
            raise ValueError("captured_at must be a finite non-negative number")
