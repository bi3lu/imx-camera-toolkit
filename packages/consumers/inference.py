"""Dedicated latest-frame worker for model-agnostic GPU inference."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Protocol, runtime_checkable

from packages.camera.models import GpuFrame
from packages.inference.contracts import FrameSpec, InferenceResult, InferenceRunner

from .latest import FrameConsumer, LatestFrameHub, LatestFrameSubscription


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
        close_runner: bool = True,
        wait_timeout: float = 0.25,
    ) -> None:
        """Configure inference ownership without preparing the model yet."""
        if not isinstance(runner, InferenceRunner):
            raise TypeError("runner must implement InferenceRunner")

        if not isinstance(close_runner, bool):
            raise TypeError("close_runner must be a boolean")

        self._runner = runner
        self._close_runner = close_runner
        self._result_lock = threading.Lock()
        self._latest_result: InferenceResult | None = None
        self._result_hub = LatestFrameHub[InferenceResult]()
        self._prepared_spec: FrameSpec | None = None
        self._runner_closed = False
        self._worker = FrameConsumer(
            subscription,
            self._infer,
            thread_name=f"imx-inference-{subscription.name}",
            wait_timeout=wait_timeout,
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
        """Newest worker exception, if inference has failed."""
        return self._worker.last_error

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
        """Stop inference and close the runner after its worker exits."""
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

    def __exit__(self, *_: object) -> None:
        """Stop inference and close its runner on context exit."""
        self.stop()


__all__ = ["InferenceConsumer", "InferenceResultSource"]
