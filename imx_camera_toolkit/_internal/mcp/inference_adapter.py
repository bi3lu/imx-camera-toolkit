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


class InferenceReadAdapter:
    """Borrow a latest-result slot without taking worker lifecycle ownership.

    The source must expose a cached read and use the server monotonic clock for
    known frame timestamps. Bind a worker with `from_worker` to include health
    counters. The stock `InferenceConsumer` satisfies that read interface.
    """

    def __init__(self, source: InferenceResultSource) -> None:
        """Bind the existing result slot without sampling it."""
        self._source = source
        self._worker: InferenceWorkerSource | None = None

    @classmethod
    def from_worker(cls, worker: InferenceWorkerSource) -> InferenceReadAdapter:
        """Borrow both the result slot and cached counters from the same worker."""
        adapter = cls(worker)
        adapter._worker = worker
        return adapter

    def status(self) -> InferenceStatus:
        """Read the slot once and project only safe timings and counters."""
        result = self._source.latest_result
        inference_time_ms = None
        age_ms = None

        if result is not None:
            inference_time_ms = _milliseconds(result.inference_time_ns)

            if result.frame_timestamp_ns > 0:
                age_ms = _milliseconds(time.monotonic_ns() - result.frame_timestamp_ns)

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
