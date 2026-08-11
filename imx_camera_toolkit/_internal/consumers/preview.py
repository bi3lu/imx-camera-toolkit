"""Model-neutral inference overlay adapter for encoded browser previews."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

from imx_camera_toolkit._internal.camera.publishing import EncodedJPEGPublisher
from imx_camera_toolkit._internal.inference.contracts import InferenceResult
from imx_camera_toolkit._internal.stream.stream import JPEGCamera

from .inference import InferenceResultSource


@dataclass(frozen=True, slots=True)
class PreviewOverlayContext:
    """Timing metadata passed to a model-specific preview renderer."""

    preview_frame_number: int
    result_frame_sequence: int | None
    result_frame_timestamp_ns: int | None
    detection_age_ns: int | None
    inference_time_ns: int | None


OverlayRenderer = Callable[
    [bytes, InferenceResult | None, PreviewOverlayContext],
    bytes,
]


class InferencePreviewSource:
    """Render the newest inference result onto each fresh preview JPEG.

    The adapter owns a dedicated worker and a single output slot. It reads the
    independently encoded preview branch at its natural rate and reuses the
    last completed inference result until a newer result arrives. Rendering is
    delegated to an application callback so boxes, masks, labels, and model
    result types remain outside the capture API.

    The preview source, inference source, and renderer lifecycle remain owned
    by the caller. Stop this adapter before stopping its preview source.
    """

    def __init__(
        self,
        source: JPEGCamera,
        inference: InferenceResultSource,
        renderer: OverlayRenderer,
        *,
        max_fps: float = 30.0,
        read_timeout: float = 0.25,
    ) -> None:
        """Configure an overlay worker without starting either source."""
        if not callable(renderer):
            raise TypeError("renderer must be callable")

        for name, value in (("max_fps", max_fps), ("read_timeout", read_timeout)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")

        self._source = source
        self._inference = inference
        self._renderer = renderer
        self._read_timeout = float(read_timeout)
        self._publisher = EncodedJPEGPublisher(float(max_fps))
        self._running = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._context_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._latest_context: PreviewOverlayContext | None = None
        self.processed_frames = 0
        self.skipped_preview_frames = 0
        self.failed_frames = 0
        self.last_error: Exception | None = None

    @property
    def running(self) -> bool:
        """Whether the independent overlay worker is active."""
        return self._running.is_set()

    @property
    def frame_number(self) -> int:
        """Identifier of the newest rendered JPEG."""
        return self._publisher.frame_number

    @property
    def jpeg(self) -> bytes | None:
        """Newest rendered JPEG bytes."""
        return self._publisher.jpeg

    @property
    def latest_context(self) -> PreviewOverlayContext | None:
        """Metadata used to render the newest output frame."""
        with self._context_lock:
            return self._latest_context

    @property
    def result_frame_timestamp_ns(self) -> int | None:
        """Monotonic timestamp of the frame used by the newest inference."""
        result = self._inference.latest_result
        return None if result is None else result.frame_timestamp_ns

    @property
    def detection_age_ns(self) -> int | None:
        """Current age of the newest inference result for UI telemetry."""
        timestamp_ns = self.result_frame_timestamp_ns
        if timestamp_ns is None:
            return None
        return max(time.monotonic_ns() - timestamp_ns, 0)

    def start(self) -> None:
        """Start overlay forwarding without taking ownership of either source."""
        with self._lifecycle_lock:
            if self.running:
                return

            if not self._source.running:
                raise RuntimeError("preview source must be running before adapter")

            if self._thread is not None:
                raise RuntimeError("inference preview source cannot be restarted")

            self.last_error = None
            self._running.set()
            self._thread = threading.Thread(
                target=self._run,
                name="imx-inference-preview",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float | None = 3.0) -> None:
        """Stop only this adapter and leave both input sources untouched."""
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be a finite non-negative number or None")

        with self._lifecycle_lock:
            self._running.clear()
            self._publisher.notify_waiters()
            thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=None if timeout is None else float(timeout))

            if thread.is_alive():
                raise RuntimeError("inference preview worker did not stop")

    def wait_for_jpeg(
        self,
        previous_frame_number: int,
        timeout: float = 2.0,
    ) -> tuple[int, bytes | None]:
        """Wait for a rendered JPEG compatible with the MJPEG stream API."""
        return self._publisher.wait_for_jpeg(
            previous_frame_number,
            timeout,
            lambda: self.running,
        )

    def _run(self) -> None:
        """Reuse the latest inference result for every newly observed preview."""
        previous_frame_number = -1
        try:
            while self.running:
                try:
                    frame_number, jpeg = self._source.wait_for_jpeg(
                        previous_frame_number,
                        self._read_timeout,
                    )
                except Exception as error:
                    self.failed_frames += 1
                    self.last_error = error
                    break

                if not self.running:
                    break

                if jpeg is None or frame_number == previous_frame_number:
                    if not self._source.running:
                        break

                    continue

                if previous_frame_number >= 0:
                    self.skipped_preview_frames += max(
                        frame_number - previous_frame_number - 1,
                        0,
                    )
                previous_frame_number = frame_number
                result = self._inference.latest_result
                timestamp_ns = None if result is None else result.frame_timestamp_ns
                context = PreviewOverlayContext(
                    preview_frame_number=frame_number,
                    result_frame_sequence=(
                        None if result is None else result.frame_sequence
                    ),
                    result_frame_timestamp_ns=timestamp_ns,
                    detection_age_ns=(
                        None
                        if timestamp_ns is None
                        else max(time.monotonic_ns() - timestamp_ns, 0)
                    ),
                    inference_time_ns=(
                        None if result is None else result.inference_time_ns
                    ),
                )

                try:
                    rendered = self._renderer(jpeg, result, context)
                    if not isinstance(rendered, bytes) or not rendered:
                        raise ValueError("renderer must return non-empty JPEG bytes")

                    published = self._publisher.publish(rendered)

                except Exception as error:
                    self.failed_frames += 1
                    self.last_error = error
                    continue

                if published:
                    with self._context_lock:
                        self._latest_context = context
                    self.processed_frames += 1

                else:
                    self.skipped_preview_frames += 1
        finally:
            self._running.clear()
            self._publisher.notify_waiters()

    def __enter__(self) -> InferencePreviewSource:
        """Start this adapter in a context-manager block."""
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop this adapter on context exit."""
        self.stop()


__all__ = ["InferencePreviewSource", "OverlayRenderer", "PreviewOverlayContext"]
