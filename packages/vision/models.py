"""Immutable data models exchanged by the vision pipeline."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    """Copy a mapping into an immutable representation."""
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class Frame:
    """One acquired image and its acquisition metadata.

    The image payload remains intentionally opaque. This keeps the pipeline
    compatible with NumPy/OpenCV images, CUDA-backed images, tensors, or custom
    image containers without imposing an AI framework dependency.

    Args:
        sequence: Monotonically increasing identifier assigned by the pipeline.
        image: Source image payload.
        captured_at: Monotonic timestamp recorded when the frame is acquired.
        metadata: Optional source-specific, JSON-compatible metadata.
    """

    sequence: int
    image: object
    captured_at: float = field(default_factory=time.monotonic)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identifiers and isolate metadata from caller mutation."""
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer")

        if not isinstance(self.captured_at, (int, float)) or self.captured_at < 0:
            raise ValueError("captured_at must be non-negative")

        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-aligned detection rectangle.

    Args:
        x: Left coordinate in pixels.
        y: Top coordinate in pixels.
        width: Rectangle width in pixels.
        height: Rectangle height in pixels.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate positive box dimensions."""
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("bounding-box width and height must be positive")


@dataclass(frozen=True, slots=True)
class Detection:
    """One model detection independent from a source image.

    Args:
        label: Human-readable predicted class.
        confidence: Normalized confidence from 0.0 to 1.0.
        box: Bounding box in pixels.
        attributes: Additional model-specific values.
    """

    label: str
    confidence: float
    box: BoundingBox
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate model output and isolate attributes from mutation."""
        if not self.label:
            raise ValueError("detection label must not be empty")

        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("detection confidence must be between 0.0 and 1.0")

        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Model output associated with a frame, without retaining its image.

    Keeping inference output separate from the image lets downstream consumers
    publish structured detections, events, telemetry, or storage records
    without duplicating potentially large frame buffers.

    Args:
        frame_sequence: Identifier of the frame processed by the model.
        detections: Model detections for that frame.
        values: Additional model-specific values, such as classifications or
            segmentation summary statistics.
        completed_at: Monotonic timestamp recorded after processing completes.
    """

    frame_sequence: int
    detections: tuple[Detection, ...] = ()
    values: Mapping[str, object] = field(default_factory=dict)
    completed_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        """Validate the result identity and isolate its metadata."""
        if (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be a non-negative integer")

        if not isinstance(self.completed_at, (int, float)) or self.completed_at < 0:
            raise ValueError("completed_at must be non-negative")

        object.__setattr__(self, "values", _freeze_mapping(self.values))


@dataclass(frozen=True, slots=True)
class OverlayFrame:
    """Optional rendered image derived from one frame and inference result.

    Args:
        frame_sequence: Identifier of the source frame.
        image: Rendered image payload.
        rendered_at: Monotonic timestamp recorded after rendering completes.
    """

    frame_sequence: int
    image: object
    rendered_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        """Validate the referenced frame identifier."""
        if (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be a non-negative integer")
