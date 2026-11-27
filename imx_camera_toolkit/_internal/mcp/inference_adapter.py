"""Read cached inference state from an application-owned source."""

from __future__ import annotations

import time
from typing import Protocol

from ..consumers.inference import InferenceResultSource
from ..inference.contracts import InferenceResult
from .contracts import MAX_SAFE_INTEGER, InferenceState, InferenceStatus
from .errors import MCPError

_NS_PER_MS = 1_000_000


class InferenceWorkerSource(InferenceResultSource, Protocol):
    """Cached result and counters exposed by an existing inference worker."""

    @property
    def running(self) -> bool:
        """Whether the worker is accepting frames."""
        ...

    @property
    def processed_frames(self) -> int:
        """Number of successful inference calls."""
        ...

    @property
    def failed_frames(self) -> int:
        """Number of failed preparations or inference calls."""
        ...

    @property
    def dropped_frames(self) -> int:
        """Number of replaced input frames."""
        ...

    @property
    def consecutive_failures(self) -> int:
        """Number of failures since the worker's last success or start."""
        ...


def _counter(value: int) -> int | None:
    """Keep counters in the JSON-safe integer range."""
    return value if type(value) is int and 0 <= value <= MAX_SAFE_INTEGER else None


def _milliseconds(nanoseconds: int) -> float | None:
    """Convert a bounded nonnegative duration to finite milliseconds."""
    if 0 <= nanoseconds <= MAX_SAFE_INTEGER * _NS_PER_MS:
        return nanoseconds / _NS_PER_MS
    return None


def _frame_age_ms(result: InferenceResult) -> float | None:
    """Measure age from a known timestamp in the host's monotonic domain."""
    if result.frame_timestamp_ns > 0:
        return _milliseconds(time.monotonic_ns() - result.frame_timestamp_ns)

    return None


def require_fresh_inference(result: InferenceResult, max_age_ms: int) -> float:
    """Return the current age or reject an expired or unmeasurable result.

    Raises:
        MCPError: If the result's age is unknown or exceeds the limit.
    """
    age_ms = _frame_age_ms(result)

    if age_ms is None or age_ms > max_age_ms:
        raise MCPError("stale_inference_result", age_ms=age_ms, max_age_ms=max_age_ms)

    return age_ms


def inference_duration_ms(result: InferenceResult) -> float | None:
    """Return the runner duration in bounded milliseconds."""
    return _milliseconds(result.inference_time_ns)


class InferenceReadAdapter:
    """Borrow a latest-result slot without taking worker lifecycle ownership.

    The source must expose a cached read and use the server monotonic clock for
    known frame timestamps. Bind a worker with `from_worker` to include health
    counters. The stock `InferenceConsumer` satisfies that read interface.
    Enable `raw_capture_sequence` when the source preserves the input capture
    counter. That counter may repeat across capture epochs.
    """

    def __init__(
        self, source: InferenceResultSource, *, raw_capture_sequence: bool = False
    ) -> None:
        """Bind the existing result slot without sampling it."""
        self._source = source
        self._raw_capture_sequence = raw_capture_sequence
        self._worker: InferenceWorkerSource | None = None

    @classmethod
    def from_worker(
        cls, worker: InferenceWorkerSource, *, raw_capture_sequence: bool = False
    ) -> InferenceReadAdapter:
        """Borrow both the result slot and cached counters from the same worker."""
        adapter = cls(worker, raw_capture_sequence=raw_capture_sequence)
        adapter._worker = worker
        return adapter

    def frame_sequence(self, result: InferenceResult) -> int | None:
        """Project a declared raw capture sequence as a diagnostic counter."""
        return _counter(result.frame_sequence) if self._raw_capture_sequence else None

    def status(self) -> InferenceStatus:
        """Read the slot once and project only safe timings and counters."""
        result = self._source.latest_result
        inference_time_ms = None if result is None else inference_duration_ms(result)
        age_ms = None if result is None else _frame_age_ms(result)

        worker = self._worker

        if worker is None:
            return InferenceStatus(
                state="unknown", inference_time_ms=inference_time_ms, age_ms=age_ms
            )

        running = worker.running
        failures = worker.consecutive_failures
        state: InferenceState

        if not running:
            state = "stopped"

        elif failures > 0:
            state = "degraded"

        elif result is None:
            state = "initializing"

        else:
            state = "running"

        return InferenceStatus(
            state=state,
            processed_frames=_counter(worker.processed_frames),
            failed_frames=_counter(worker.failed_frames),
            dropped_frames=_counter(worker.dropped_frames),
            consecutive_failures=_counter(failures),
            error="worker_failure" if failures > 0 else None,
            inference_time_ms=inference_time_ms,
            age_ms=age_ms,
        )

    def latest_result(self) -> InferenceResult:
        """Return one cached result for internal processing outside worker locks.

        Raises:
            MCPError: If the source has no result.
        """
        result = self._source.latest_result

        if result is None:
            raise MCPError("no_inference_result")

        return result
