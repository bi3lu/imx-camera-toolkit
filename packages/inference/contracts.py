"""Model-agnostic contracts for GPU inference consumers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from packages.camera.models import FrameFormat, GpuFrame, MemoryType


@dataclass(frozen=True, slots=True)
class FrameSpec:
    """Static pixel and memory layout prepared by an inference runner."""

    width: int
    height: int
    format: FrameFormat
    memory_type: MemoryType

    def __post_init__(self) -> None:
        """Validate a complete positive frame layout."""
        for field_name in ("width", "height"):
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        if not isinstance(self.format, FrameFormat):
            raise ValueError("format must be a FrameFormat")

        if not isinstance(self.memory_type, MemoryType):
            raise ValueError("memory_type must be a MemoryType")

    @classmethod
    def from_gpu_frame(cls, frame: GpuFrame) -> FrameSpec:
        """Create a preparation spec from public GPU frame metadata."""
        if not isinstance(frame, GpuFrame):
            raise TypeError("frame must be a GpuFrame")

        return cls(
            width=frame.width,
            height=frame.height,
            format=frame.format,
            memory_type=frame.memory_type,
        )


@dataclass(frozen=True, slots=True)
class ShapeProfile:
    """TensorRT dynamic input bounds in ``min``/``opt``/``max`` order."""

    minimum: tuple[int, ...]
    optimum: tuple[int, ...]
    maximum: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate ranks, positive dimensions, and monotonic bounds."""
        ranks = {len(self.minimum), len(self.optimum), len(self.maximum)}

        if len(ranks) != 1 or not self.minimum:
            raise ValueError("shape profile bounds must have one non-zero rank")

        for dimensions in (self.minimum, self.optimum, self.maximum):
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in dimensions
            ):
                raise ValueError("shape profile dimensions must be positive integers")

        if any(
            minimum > optimum or optimum > maximum
            for minimum, optimum, maximum in zip(
                self.minimum,
                self.optimum,
                self.maximum,
                strict=True,
            )
        ):
            raise ValueError("shape profile must satisfy min <= opt <= max")

    def contains(self, shape: tuple[int, ...]) -> bool:
        """Whether ``shape`` has the configured rank and bounded dimensions."""
        return len(shape) == len(self.minimum) and all(
            minimum <= value <= maximum
            for minimum, value, maximum in zip(
                self.minimum,
                shape,
                self.maximum,
                strict=True,
            )
        )

    def as_dict(self) -> dict[str, list[int]]:
        """Return canonical JSON-compatible cache metadata."""
        return {
            "min": list(self.minimum),
            "opt": list(self.optimum),
            "max": list(self.maximum),
        }


@dataclass(frozen=True, slots=True)
class TensorOutput:
    """One named output tensor copied back after GPU execution."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    data: object = field(repr=False)

    def __post_init__(self) -> None:
        """Validate portable tensor metadata without prescribing an array type."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tensor output name must be non-empty")

        if not self.shape or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.shape
        ):
            raise ValueError("tensor output shape must contain non-negative integers")

        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("tensor output dtype must be non-empty")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Model-neutral outputs and timing associated with one input frame."""

    frame_sequence: int
    frame_timestamp_ns: int
    inference_time_ns: int
    outputs: tuple[TensorOutput, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)
    overlays: tuple[object, ...] = ()
    capture_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        """Validate timing and detach mutable result metadata."""
        if (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence <= 0
        ):
            raise ValueError("frame_sequence must be a positive integer")

        if (
            isinstance(self.inference_time_ns, bool)
            or not isinstance(self.inference_time_ns, int)
            or self.inference_time_ns < 0
        ):
            raise ValueError("inference_time_ns must be non-negative")

        if (
            isinstance(self.frame_timestamp_ns, bool)
            or not isinstance(self.frame_timestamp_ns, int)
            or self.frame_timestamp_ns < 0
        ):
            raise ValueError("frame_timestamp_ns must be non-negative")

        if not isinstance(self.outputs, tuple) or not all(
            isinstance(output, TensorOutput) for output in self.outputs
        ):
            raise ValueError("outputs must be a tuple of TensorOutput values")

        if self.capture_timestamp_ns is not None and (
            isinstance(self.capture_timestamp_ns, bool)
            or not isinstance(self.capture_timestamp_ns, int)
            or self.capture_timestamp_ns < 0
        ):
            raise ValueError("capture_timestamp_ns must be non-negative or None")

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "overlays", tuple(self.overlays))


@runtime_checkable
class InferenceRunner(Protocol):
    """Lifecycle contract implemented by model-specific GPU consumers."""

    def prepare(self, frame_spec: FrameSpec) -> None:
        """Build or load resources compatible with ``frame_spec``."""
        ...

    def infer(self, frame: GpuFrame) -> InferenceResult:
        """Run one borrowed GPU frame before its lease expires."""
        ...

    def close(self) -> None:
        """Synchronize and release runner-owned GPU resources."""
        ...
