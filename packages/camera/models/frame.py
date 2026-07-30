"""Raw camera-frame model for external image-processing consumers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Frame:
    """One processed camera frame without JPEG encoding or inference.

    The image payload is intentionally opaque so callers can pass it directly
    to OpenCV, TensorRT, DeepStream, CUDA, or another image-processing system.
    The model is shallowly immutable: ``image`` is retained by reference when
    ``Camera.read(copy=False)`` is used.

    Args:
        image: Processed BGR image payload.
        sequence: Monotonically increasing identifier assigned during capture.
        timestamp_ns: Monotonic timestamp recorded when the frame is acquired.
        capture_timestamp_ns: Optional hardware-provided capture timestamp.
        width: Image width in pixels.
        height: Image height in pixels.
        format: Pixel format name, such as ``"BGR"``.
    """

    image: object
    sequence: int
    timestamp_ns: int
    capture_timestamp_ns: int | None
    width: int
    height: int
    format: str

    def __post_init__(self) -> None:
        """Validate frame metadata without inspecting the image payload."""
        integer_fields = ("sequence", "timestamp_ns", "width", "height")
        for field_name in integer_fields:
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")

        if self.sequence <= 0:
            raise ValueError("sequence must be a positive integer")

        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

        if not isinstance(self.format, str) or not self.format.strip():
            raise ValueError("format must be a non-empty string")

        if self.capture_timestamp_ns is not None and (
            isinstance(self.capture_timestamp_ns, bool)
            or not isinstance(self.capture_timestamp_ns, int)
            or self.capture_timestamp_ns < 0
        ):
            raise ValueError("capture_timestamp_ns must be a non-negative integer")


CameraFrame = Frame
