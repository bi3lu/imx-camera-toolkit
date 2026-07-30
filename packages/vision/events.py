"""Thread-safe event publication for the vision-pipeline lifecycle."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from .models import InferenceResult

logger = logging.getLogger(__name__)


class PipelineEventType(str, Enum):
    """Kinds of state changes emitted by :class:`VisionPipeline`."""

    STARTED = "started"
    FRAME_CAPTURED = "frame_captured"
    FRAME_DROPPED = "frame_dropped"
    RESULT_AVAILABLE = "result_available"
    OVERLAY_AVAILABLE = "overlay_available"
    SOURCE_EXHAUSTED = "source_exhausted"
    SOURCE_ERROR = "source_error"
    PROCESSING_ERROR = "processing_error"
    OVERLAY_ERROR = "overlay_error"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """A lifecycle, frame, result, or error notification.

    Events deliberately contain frame identifiers and inference results rather
    than image buffers. Event consumers can therefore log, serialize, or route
    model output without retaining source image memory.

    Args:
        type: Category of pipeline activity.
        occurred_at: Monotonic event timestamp.
        frame_sequence: Relevant frame identifier, when applicable.
        result: Inference output, when processing completed successfully.
        error: Source, processor, or overlay exception, when applicable.
        details: Additional event metadata.
    """

    type: PipelineEventType
    occurred_at: float = field(default_factory=time.monotonic)
    frame_sequence: int | None = None
    result: InferenceResult | None = None
    error: Exception | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate optional frame identity and isolate event details."""
        if self.frame_sequence is not None and (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be a non-negative integer")

        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


PipelineEventHandler = Callable[[PipelineEvent], None]


class EventBus:
    """Publish events to subscribers while isolating listener failures."""

    def __init__(self) -> None:
        """Initialize an empty, thread-safe subscription list."""
        self._handlers: list[PipelineEventHandler] = []
        self._lock = threading.RLock()

    def subscribe(self, handler: PipelineEventHandler) -> Callable[[], None]:
        """Register an event handler and return an unsubscribe callback.

        Args:
            handler: Callback invoked synchronously after pipeline state changes.

        Returns:
            Function that removes this handler. It is safe to call repeatedly.
        """
        with self._lock:
            self._handlers.append(handler)

        def unsubscribe() -> None:
            """Remove the registered callback if it is still present."""
            with self._lock:
                if handler in self._handlers:
                    self._handlers.remove(handler)

        return unsubscribe

    def emit(self, event: PipelineEvent) -> None:
        """Invoke subscribers while isolating listener failures.

        Args:
            event: Notification to deliver.
        """
        with self._lock:
            handlers = tuple(self._handlers)

        for handler in handlers:
            try:
                handler(event)

            except Exception:
                logger.exception("Vision-pipeline event handler failed")
