"""Model-agnostic pipeline timing and drop-counter contracts."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite


class PipelineStage(str, Enum):
    """Stages whose latency can be reported without knowing the model."""

    TRANSFER = "transfer"
    INFERENCE = "inference"
    ENCODER = "encoder"
    END_TO_END = "end_to_end"


@dataclass(frozen=True, slots=True)
class StageMetrics:
    """Immutable latency aggregate for one pipeline stage, in nanoseconds."""

    samples: int = 0
    total_duration_ns: int = 0
    last_duration_ns: int | None = None
    max_duration_ns: int | None = None

    def __post_init__(self) -> None:
        """Validate a consistent non-negative aggregate."""
        for field_name in ("samples", "total_duration_ns"):
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

        for field_name in ("last_duration_ns", "max_duration_ns"):
            value = getattr(self, field_name)

            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field_name} must be non-negative or None")

        if self.samples == 0 and (
            self.total_duration_ns != 0
            or self.last_duration_ns is not None
            or self.max_duration_ns is not None
        ):
            raise ValueError("empty stage metrics cannot contain durations")

        if self.samples > 0 and (
            self.last_duration_ns is None or self.max_duration_ns is None
        ):
            raise ValueError("non-empty stage metrics require last and max durations")

    @property
    def mean_duration_ns(self) -> float | None:
        """Arithmetic mean duration, or ``None`` without samples."""
        if self.samples == 0:
            return None

        return self.total_duration_ns / self.samples


@dataclass(frozen=True, slots=True)
class PipelineMetrics:
    """Immutable latency snapshot for every supported processing stage."""

    transfer: StageMetrics = field(default_factory=StageMetrics)
    inference: StageMetrics = field(default_factory=StageMetrics)
    encoder: StageMetrics = field(default_factory=StageMetrics)
    end_to_end: StageMetrics = field(default_factory=StageMetrics)

    def for_stage(self, stage: PipelineStage | str) -> StageMetrics:
        """Return metrics for ``stage`` with public string-enum validation."""
        normalized = _normalize_stage(stage)
        if normalized is PipelineStage.TRANSFER:
            return self.transfer
        if normalized is PipelineStage.INFERENCE:
            return self.inference
        if normalized is PipelineStage.ENCODER:
            return self.encoder
        return self.end_to_end


@dataclass(slots=True)
class _MutableStageMetrics:
    """Internal accumulator protected by :class:`MetricsRecorder`."""

    samples: int = 0
    total_duration_ns: int = 0
    last_duration_ns: int | None = None
    max_duration_ns: int | None = None

    def record(self, duration_ns: int) -> None:
        """Add one validated duration."""
        self.samples += 1
        self.total_duration_ns += duration_ns
        self.last_duration_ns = duration_ns
        self.max_duration_ns = max(self.max_duration_ns or 0, duration_ns)

    def snapshot(self) -> StageMetrics:
        """Return an immutable copy of this accumulator."""
        return StageMetrics(
            samples=self.samples,
            total_duration_ns=self.total_duration_ns,
            last_duration_ns=self.last_duration_ns,
            max_duration_ns=self.max_duration_ns,
        )


def _normalize_stage(stage: PipelineStage | str) -> PipelineStage:
    """Normalize a public stage identifier."""
    if isinstance(stage, PipelineStage):
        return stage

    if not isinstance(stage, str):
        raise ValueError("stage must be a PipelineStage or string")

    try:
        return PipelineStage(stage)

    except ValueError as error:
        supported = ", ".join(item.value for item in PipelineStage)
        raise ValueError(f"unknown pipeline stage; use one of: {supported}") from error


class MetricsRecorder:
    """Thread-safe recorder shared by capture and external consumers.

    The recorder stores aggregates only: it never retains frames or builds a
    per-frame timing queue. Consumers can report inference and end-to-end
    durations with :meth:`record_stage`, and report frames skipped by their
    own latest-frame loop with :meth:`record_consumer_drop`.
    """

    def __init__(self) -> None:
        """Initialize empty fixed-size timing aggregates and drop counters."""
        self._lock = threading.Lock()
        self._stages = {stage: _MutableStageMetrics() for stage in PipelineStage}
        self._consumer_drops: dict[str, int] = {}

    def record_stage(self, stage: PipelineStage | str, duration_ns: int) -> None:
        """Record one finite non-negative stage duration in nanoseconds."""
        normalized = _normalize_stage(stage)

        if (
            isinstance(duration_ns, bool)
            or not isinstance(duration_ns, (int, float))
            or not isfinite(duration_ns)
            or duration_ns < 0
        ):
            raise ValueError("duration_ns must be a finite non-negative number")

        resolved_duration_ns = int(duration_ns)

        with self._lock:
            self._stages[normalized].record(resolved_duration_ns)

    def record_consumer_drop(self, consumer: str, count: int = 1) -> None:
        """Increment the skipped-frame counter for a named consumer."""
        if not isinstance(consumer, str) or not consumer.strip():
            raise ValueError("consumer must be a non-empty string")

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")

        normalized = consumer.strip()

        with self._lock:
            self._consumer_drops[normalized] = (
                self._consumer_drops.get(normalized, 0) + count
            )

    def snapshot(self) -> PipelineMetrics:
        """Return immutable latency aggregates."""
        with self._lock:
            return PipelineMetrics(
                transfer=self._stages[PipelineStage.TRANSFER].snapshot(),
                inference=self._stages[PipelineStage.INFERENCE].snapshot(),
                encoder=self._stages[PipelineStage.ENCODER].snapshot(),
                end_to_end=self._stages[PipelineStage.END_TO_END].snapshot(),
            )

    def consumer_drops(self) -> tuple[tuple[str, int], ...]:
        """Return a stable, immutable snapshot of per-consumer drops."""
        with self._lock:
            return tuple(sorted(self._consumer_drops.items()))
