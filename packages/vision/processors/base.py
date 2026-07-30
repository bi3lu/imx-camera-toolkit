"""Contracts for AI inference and other frame-processing stages."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Frame, InferenceResult


@runtime_checkable
class FrameProcessor(Protocol):
    """Transform one frame into structured output without owning image storage."""

    def process(self, frame: Frame) -> InferenceResult:
        """Run processing for one frame and return its independent result."""
        ...


@runtime_checkable
class ManagedFrameProcessor(FrameProcessor, Protocol):
    """Optional resource lifecycle for model-backed frame processors.

    Implement this protocol only when a processor needs to load an engine,
    create a CUDA context, allocate buffers, or release external resources.
    Plain :class:`FrameProcessor` implementations remain valid and require no
    no-op lifecycle methods.
    """

    def open(self) -> None:
        """Allocate processor resources before pipeline capture starts."""
        ...

    def close(self) -> None:
        """Release processor resources after pipeline workers stop."""
        ...


class NoopFrameProcessor:
    """Minimal processor useful for pipeline wiring, testing, and benchmarking."""

    def process(self, frame: Frame) -> InferenceResult:
        """Return an empty result linked to the processed frame.

        Args:
            frame: Frame consumed by this processor.

        Returns:
            Empty inference output with the matching frame sequence.
        """
        return InferenceResult(frame_sequence=frame.sequence)
