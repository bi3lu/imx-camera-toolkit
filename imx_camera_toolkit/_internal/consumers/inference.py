"""Dedicated latest-frame worker for model-agnostic GPU inference."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Protocol, runtime_checkable

from imx_camera_toolkit._internal.camera.models import GpuFrame, GpuFrameExpiredError
from imx_camera_toolkit._internal.inference.contracts import (
    FrameSpec,
    InferenceResult,
    InferenceRunner,
)

from .latest import FrameConsumer, LatestFrameHub, LatestFrameSubscription

logger = logging.getLogger(__name__)


@runtime_checkable
class InferenceResultSource(Protocol):
    """Small contract consumed by preview and telemetry adapters."""

    @property
    def latest_result(self) -> InferenceResult | None:
        """Newest completed inference result, if one exists."""
        ...


class InferenceConsumer:
    """Execute an inference runner on a dedicated latest-frame worker.

    One ``InferenceConsumer`` should own one runner instance. The reference
    ``TensorRTRunner`` owns one CUDA stream, so separate consumers naturally
    execute on separate worker threads and CUDA streams without blocking the
    camera capture loop.
    """

    def __init__(
        self,
        subscription: LatestFrameSubscription[GpuFrame],
        runner: InferenceRunner,
        *,
        prepared_spec: FrameSpec | None = None,
        close_runner: bool = True,
        wait_timeout: float = 0.25,
        on_error: Callable[[Exception], None] | None = None,
        error_log_interval: float = 5.0,
        initial_failure_backoff: float = 0.05,
        max_failure_backoff: float = 1.0,
    ) -> None:
        """Configure inference ownership and accept an optional prepared layout."""
        if not isinstance(runner, InferenceRunner):
            raise TypeError("runner must implement InferenceRunner")

        if not isinstance(close_runner, bool):
            raise TypeError("close_runner must be a boolean")

        runner_spec = getattr(runner, "prepared_frame_spec", None)

        if runner_spec is not None and not isinstance(runner_spec, FrameSpec):
            raise TypeError("runner.prepared_frame_spec must be a FrameSpec or None")

        if prepared_spec is not None and not isinstance(prepared_spec, FrameSpec):
            raise TypeError("prepared_spec must be a FrameSpec or None")

        if (
            prepared_spec is not None
            and runner_spec is not None
            and prepared_spec != runner_spec
        ):
            raise ValueError("prepared_spec differs from runner.prepared_frame_spec")

        self._runner = runner
        self._close_runner = close_runner
        self._result_lock = threading.Lock()
        self._latest_result: InferenceResult | None = None
        self._result_hub = LatestFrameHub[InferenceResult]()
        self._prepared_spec = prepared_spec or runner_spec
        self._runner_closed = False
        self._worker = FrameConsumer(
            subscription,
            self._infer,
            thread_name=f"imx-inference-{subscription.name}",
            wait_timeout=wait_timeout,
            on_error=on_error,
            error_log_interval=error_log_interval,
            initial_failure_backoff=initial_failure_backoff,
            max_failure_backoff=max_failure_backoff,
            drop_exceptions=(GpuFrameExpiredError,),
        )

    @property
    def running(self) -> bool:
        """Whether this consumer's inference worker is active."""
        return self._worker.running

    @property
    def latest_result(self) -> InferenceResult | None:
        """Newest completed result without waiting or consuming it."""
        with self._result_lock:
            return self._latest_result

    @property
    def prepared_frame_spec(self) -> FrameSpec | None:
        """Frame layout already prepared by this consumer's runner."""
        return self._prepared_spec

    @property
    def processed_frames(self) -> int:
        """Number of successful inference calls."""
        return self._worker.processed_frames

    @property
    def failed_frames(self) -> int:
        """Number of preparation or inference failures."""
        return self._worker.failed_frames

    @property
    def dropped_frames(self) -> int:
        """Number of input frames replaced while inference was busy."""
        return self._worker.dropped_frames

    @property
    def last_error(self) -> Exception | None:
        """Current worker exception, cleared after successful inference."""
        return self._worker.last_error

    @property
    def last_failure(self) -> Exception | None:
        """Most recent historical worker exception, including recovered ones."""
        return self._worker.last_failure

    @property
    def consecutive_failures(self) -> int:
        """Number of uninterrupted preparation or inference failures."""
        return self._worker.consecutive_failures

    @property
    def healthy(self) -> bool:
        """Whether the latest inference attempt completed successfully."""
        return self._worker.healthy

    def health(self) -> dict[str, object]:
        """Return model-neutral inference state suitable for diagnostics."""
        result = self.latest_result
        prepared = self._prepared_spec
        return {
            "running": self.running,
            "healthy": self.healthy,
            "prepared_frame_spec": (
                None
                if prepared is None
                else {
                    "width": prepared.width,
                    "height": prepared.height,
                    "format": prepared.format.value,
                    "memory_type": prepared.memory_type.value,
                }
            ),
            "processed_frames": self.processed_frames,
            "failed_frames": self.failed_frames,
            "dropped_frames": self.dropped_frames,
            "consecutive_failures": self.consecutive_failures,
            "last_error": None if self.last_error is None else str(self.last_error),
            "last_failure": (
                None if self.last_failure is None else str(self.last_failure)
            ),
            "latest_frame_sequence": (
                None if result is None else result.frame_sequence
            ),
            "latest_inference_time_ns": (
                None if result is None else result.inference_time_ns
            ),
            "output_shapes": (
                {}
                if result is None
                else {output.name: list(output.shape) for output in result.outputs}
            ),
        }

    @property
    def thread_ident(self) -> int | None:
        """Identifier of the dedicated inference worker."""
        return self._worker.thread_ident

    def subscribe_results(
        self,
        name: str,
    ) -> LatestFrameSubscription[InferenceResult]:
        """Create a separate latest-result slot for another adapter."""
        return self._result_hub.subscribe(name)

    def start(self) -> None:
        """Start inference on the consumer's dedicated worker."""
        if self._runner_closed:
            raise RuntimeError("inference consumer has already closed its runner")

        self._worker.start()

    def stop(self, timeout: float | None = 10.0) -> None:
        """Stop inference and close the runner after its worker exits.

        An active TensorRT engine build or inference call cannot be interrupted;
        prepare expensive engines before capture when shutdown must be bounded.
        """
        stopped = self._worker.stop(timeout)

        if not stopped:
            raise RuntimeError("inference worker did not stop before timeout")

        if self._close_runner and not self._runner_closed:
            self._runner.close()
            self._runner_closed = True

        self._result_hub.close()

    def _infer(self, frame: GpuFrame) -> None:
        """Prepare for layout changes, infer, and publish the newest result."""
        frame_spec = FrameSpec.from_gpu_frame(frame)
        if frame_spec != self._prepared_spec:
            self._runner.prepare(frame_spec)
            self._prepared_spec = frame_spec

        result = self._runner.infer(frame)
        result = replace(
            result,
            frame_sequence=frame.sequence,
            frame_timestamp_ns=frame.timestamp_ns,
            capture_timestamp_ns=frame.capture_timestamp_ns,
        )
        with self._result_lock:
            self._latest_result = result
        self._result_hub.publish(result)

    def __enter__(self) -> InferenceConsumer:
        """Start inference in a context-manager block."""
        self.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        """Stop inference without masking an exception from the managed body."""
        try:
            self.stop()

        except Exception:
            if exception_type is None:
                raise

            logger.exception("Inference cleanup failed after an earlier exception")


__all__ = ["InferenceConsumer", "InferenceResultSource"]
