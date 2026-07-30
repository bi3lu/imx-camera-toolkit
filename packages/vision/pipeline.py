"""Lifecycle-managed latest-frame pipeline for AI Vision processing."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from .events import EventBus, PipelineEvent, PipelineEventHandler, PipelineEventType
from .models import Frame, InferenceResult, OverlayFrame
from .processors import FrameProcessor, Overlay
from .sources import FrameSource

logger = logging.getLogger(__name__)


class PipelineState(str, Enum):
    """Lifecycle states of :class:`VisionPipeline`."""

    STOPPED = "stopped"
    RUNNING = "running"
    STOPPING = "stopping"


@dataclass(frozen=True, slots=True)
class PipelineStats:
    """Immutable counters for one pipeline lifecycle.

    Attributes:
        frames_captured: Frames accepted from the source.
        frames_processed: Successful processor calls.
        frames_dropped: Pending frames replaced by newer frames.
        processing_errors: Processor failures that did not stop capture.
        overlay_errors: Overlay failures that did not invalidate inference.
        source_errors: Exceptions raised by the source.
    """

    frames_captured: int
    frames_processed: int
    frames_dropped: int
    processing_errors: int
    overlay_errors: int
    source_errors: int


class VisionPipeline:
    """Run a frame source and processor using a bounded latest-frame policy.

    Capture and processing run on separate threads. The capture side holds at
    most one pending frame. When source acquisition outpaces processing, the
    pending frame is atomically replaced, so inference receives the most recent
    available image rather than a growing queue of stale work.

    Inference results, source frames, and optional overlay frames are stored in
    separate properties. In particular, :attr:`latest_result` never contains a
    source image buffer.

    Args:
        source: Lifecycle-aware source of image payloads.
        processor: AI inference or other processor invoked for each selected
            frame.
        overlay: Optional renderer invoked after successful inference.
        idle_sleep: Delay used when a live source temporarily returns no frame.
    """

    def __init__(
        self,
        source: FrameSource,
        processor: FrameProcessor,
        *,
        overlay: Overlay | None = None,
        idle_sleep: float = 0.005,
    ) -> None:
        """Initialize a stopped pipeline without opening the source."""
        if isinstance(idle_sleep, bool) or idle_sleep <= 0:
            raise ValueError("idle_sleep must be a positive number")

        self._source = source
        self._processor = processor
        self._overlay = overlay
        self._idle_sleep = idle_sleep
        self._condition = threading.Condition(threading.RLock())
        self._events = EventBus()
        self._state = PipelineState.STOPPED
        self._stop_requested = threading.Event()
        self._capture_complete = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._processing_thread: threading.Thread | None = None
        self._pending_frame: Frame | None = None
        self._latest_frame: Frame | None = None
        self._latest_result: InferenceResult | None = None
        self._latest_overlay: OverlayFrame | None = None
        self._last_error: Exception | None = None
        self._next_sequence = 0
        self._frames_captured = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._processing_errors = 0
        self._overlay_errors = 0
        self._source_errors = 0

    @property
    def state(self) -> PipelineState:
        """PipelineState: Current lifecycle state."""
        with self._condition:
            return self._state

    @property
    def running(self) -> bool:
        """bool: Whether capture and processing are active."""
        return self.state is PipelineState.RUNNING

    @property
    def latest_frame(self) -> Frame | None:
        """Frame | None: Latest acquired source frame, independent of results."""
        with self._condition:
            return self._latest_frame

    @property
    def latest_result(self) -> InferenceResult | None:
        """InferenceResult | None: Latest successful model output without image data."""
        with self._condition:
            return self._latest_result

    @property
    def latest_overlay(self) -> OverlayFrame | None:
        """OverlayFrame | None: Latest optional rendered image."""
        with self._condition:
            return self._latest_overlay

    @property
    def last_error(self) -> Exception | None:
        """Exception | None: Most recent source, processor, or overlay failure."""
        with self._condition:
            return self._last_error

    @property
    def stats(self) -> PipelineStats:
        """PipelineStats: Snapshot of counters for the current lifecycle."""
        with self._condition:
            return PipelineStats(
                frames_captured=self._frames_captured,
                frames_processed=self._frames_processed,
                frames_dropped=self._frames_dropped,
                processing_errors=self._processing_errors,
                overlay_errors=self._overlay_errors,
                source_errors=self._source_errors,
            )

    def subscribe(self, handler: PipelineEventHandler) -> Callable[[], None]:
        """Subscribe to pipeline events.

        Args:
            handler: Callback invoked after lifecycle and processing events.

        Returns:
            A callback that unsubscribes ``handler``.
        """
        return self._events.subscribe(handler)

    def start(self) -> None:
        """Open the source and start capture and processing threads.

        Raises:
            RuntimeError: If the previous lifecycle is still stopping.
        """
        with self._condition:
            if self._state is PipelineState.RUNNING:
                return

            if self._state is PipelineState.STOPPING or self._threads_alive():
                raise RuntimeError("vision pipeline is still stopping")

            self._reset_for_start()
            source_event: PipelineEvent | None

            try:
                self._source.open()

            except Exception as error:
                self._last_error = error
                self._source_errors += 1
                source_event = PipelineEvent(
                    PipelineEventType.SOURCE_ERROR,
                    error=error,
                )

            else:
                source_event = None
                self._state = PipelineState.RUNNING

        if source_event is not None:
            self._events.emit(source_event)
            assert source_event.error is not None
            raise source_event.error

        self._events.emit(PipelineEvent(PipelineEventType.STARTED))

        with self._condition:
            if self._state is not PipelineState.RUNNING:
                return

            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="imx-vision-capture",
                daemon=True,
            )
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                name="imx-vision-process",
                daemon=True,
            )
            self._capture_thread.start()
            self._processing_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Request shutdown, release the source, and wait for worker threads.

        Args:
            timeout: Maximum total wait time for capture and processing workers.

        Raises:
            RuntimeError: If a worker does not stop before ``timeout``.
            ValueError: If ``timeout`` is negative.
        """
        if timeout < 0:
            raise ValueError("timeout must be non-negative")

        with self._condition:
            if self._state is PipelineState.STOPPED and not self._threads_alive():
                return
            self._state = PipelineState.STOPPING
            self._stop_requested.set()
            self._condition.notify_all()
            capture_thread = self._capture_thread
            processing_thread = self._processing_thread

        self._source.close()
        deadline = time.monotonic() + timeout
        self._join_thread(capture_thread, deadline)
        self._join_thread(processing_thread, deadline)

        if self._threads_alive():
            raise RuntimeError("vision pipeline workers did not stop before timeout")

        self._finish_stopped()

    def wait_until_stopped(self, timeout: float = 5.0) -> bool:
        """Wait until a finite source lifecycle completes.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            ``True`` when the pipeline stopped before the timeout.
        """
        if timeout < 0:
            raise ValueError("timeout must be non-negative")

        with self._condition:
            return self._condition.wait_for(
                lambda: self._state is PipelineState.STOPPED,
                timeout=timeout,
            )

    def _reset_for_start(self) -> None:
        """Reset lifecycle-specific state while holding ``_condition``."""
        self._stop_requested.clear()
        self._capture_complete.clear()
        self._pending_frame = None
        self._latest_frame = None
        self._latest_result = None
        self._latest_overlay = None
        self._last_error = None
        self._next_sequence = 0
        self._frames_captured = 0
        self._frames_processed = 0
        self._frames_dropped = 0
        self._processing_errors = 0
        self._overlay_errors = 0
        self._source_errors = 0

    def _threads_alive(self) -> bool:
        """Return whether either pipeline worker is still alive."""
        return any(
            thread is not None and thread.is_alive()
            for thread in (self._capture_thread, self._processing_thread)
        )

    def _capture_loop(self) -> None:
        """Acquire source frames and replace the single pending frame slot."""
        try:
            while not self._stop_requested.is_set():
                payload = self._source.read()

                if payload is None:
                    if self._source.exhausted:
                        self._events.emit(PipelineEvent(PipelineEventType.SOURCE_EXHAUSTED))
                        return

                    time.sleep(self._idle_sleep)
                    continue

                captured_event, dropped_event = self._publish_latest_frame(payload)
                self._events.emit(captured_event)

                if dropped_event is not None:
                    self._events.emit(dropped_event)

        except Exception as error:
            logger.exception("Vision frame source failed")

            with self._condition:
                self._source_errors += 1
                self._last_error = error

            self._events.emit(
                PipelineEvent(PipelineEventType.SOURCE_ERROR, error=error)
            )

        finally:
            self._capture_complete.set()

            with self._condition:
                self._condition.notify_all()

            self._source.close()

    def _publish_latest_frame(
        self,
        payload: object,
    ) -> tuple[PipelineEvent, PipelineEvent | None]:
        """Store an acquired frame and atomically discard stale pending work."""
        with self._condition:
            frame = Frame(sequence=self._next_sequence, image=payload)
            self._next_sequence += 1
            dropped_event: PipelineEvent | None = None

            if self._pending_frame is not None:
                dropped_sequence = self._pending_frame.sequence
                self._frames_dropped += 1
                dropped_event = PipelineEvent(
                    PipelineEventType.FRAME_DROPPED,
                    frame_sequence=dropped_sequence,
                    details={"replacement_sequence": frame.sequence},
                )

            self._pending_frame = frame
            self._latest_frame = frame
            self._frames_captured += 1
            self._condition.notify_all()
            captured_event = PipelineEvent(
                PipelineEventType.FRAME_CAPTURED,
                frame_sequence=frame.sequence,
            )
            return captured_event, dropped_event

    def _processing_loop(self) -> None:
        """Process selected newest frames until capture is complete or stopped."""
        try:
            while True:
                frame = self._next_pending_frame()
                if frame is None:
                    return
                self._process_frame(frame)

        finally:
            self._finish_stopped()

    def _next_pending_frame(self) -> Frame | None:
        """Wait for pending work and return its newest frame."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._pending_frame is not None
                or self._capture_complete.is_set()
            )
            frame = self._pending_frame
            self._pending_frame = None
            return frame

    def _process_frame(self, frame: Frame) -> None:
        """Run inference and optional rendering while preserving result isolation."""
        try:
            result = self._processor.process(frame)

            if result.frame_sequence != frame.sequence:
                raise ValueError("processor result does not match the source frame")

        except Exception as error:
            logger.exception("Vision frame processing failed")

            with self._condition:
                self._processing_errors += 1
                self._last_error = error

            self._events.emit(
                PipelineEvent(
                    PipelineEventType.PROCESSING_ERROR,
                    frame_sequence=frame.sequence,
                    error=error,
                )
            )
            return

        with self._condition:
            self._frames_processed += 1
            self._latest_result = result

        self._events.emit(
            PipelineEvent(
                PipelineEventType.RESULT_AVAILABLE,
                frame_sequence=frame.sequence,
                result=result,
            )
        )

        if self._overlay is not None:
            self._render_overlay(frame, result)

    def _render_overlay(self, frame: Frame, result: InferenceResult) -> None:
        """Render optional output without invalidating a successful result."""
        overlay = self._overlay
        assert overlay is not None

        try:
            overlay_frame = overlay.render(frame, result)

            if overlay_frame.frame_sequence != frame.sequence:
                raise ValueError("overlay frame does not match the source frame")

        except Exception as error:
            logger.exception("Vision overlay rendering failed")

            with self._condition:
                self._overlay_errors += 1
                self._last_error = error

            self._events.emit(
                PipelineEvent(
                    PipelineEventType.OVERLAY_ERROR,
                    frame_sequence=frame.sequence,
                    error=error,
                )
            )
            return

        with self._condition:
            self._latest_overlay = overlay_frame

        self._events.emit(
            PipelineEvent(
                PipelineEventType.OVERLAY_AVAILABLE,
                frame_sequence=frame.sequence,
            )
        )

    def _join_thread(self, thread: threading.Thread | None, deadline: float) -> None:
        """Join a worker for the remaining portion of a shutdown timeout."""
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def _finish_stopped(self) -> None:
        """Publish exactly one terminal state transition when workers are done."""
        emit_stopped = False

        with self._condition:
            if self._state is not PipelineState.STOPPED:
                self._state = PipelineState.STOPPED
                self._condition.notify_all()
                emit_stopped = True

        if emit_stopped:
            self._events.emit(PipelineEvent(PipelineEventType.STOPPED))
