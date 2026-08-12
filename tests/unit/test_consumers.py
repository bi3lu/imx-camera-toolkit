"""Tests for bounded asynchronous frame consumers and preview overlays."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from functools import partial

import pytest

from imx_camera_toolkit import Camera, GpuCamera
from imx_camera_toolkit._internal.camera.models import GpuFrame, GpuFrameExpiredError
from imx_camera_toolkit.consumers import (
    FrameConsumer,
    InferenceConsumer,
    InferencePreviewSource,
    LatestFrameHub,
    PreviewOverlayContext,
)
from imx_camera_toolkit.inference import FrameSpec, InferenceResult, TensorOutput
from imx_camera_toolkit.testing import mock_gpu_frame


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> None:
    """Wait until a zero-argument predicate becomes true."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        time.sleep(0.001)


def _preview_processed(
    preview: InferencePreviewSource,
    expected: int,
) -> bool:
    """Whether the adapter has rendered an expected number of previews."""
    return preview.processed_frames >= expected


def _consumer_failed(consumer: FrameConsumer[int], expected: int) -> bool:
    """Whether a deterministic integer consumer reached a failure count."""
    return consumer.failed_frames == expected


def test_each_subscription_has_one_independent_latest_frame_slot() -> None:
    """A slow subscriber must skip history without affecting a fast one."""
    drops: list[tuple[str, int]] = []
    hub = LatestFrameHub[int](lambda name, count: drops.append((name, count)))
    fast = hub.subscribe("preview")
    slow = hub.subscribe("inference")

    hub.publish(1)
    assert fast.receive(0) == 1
    hub.publish(2)
    assert fast.receive(0) == 2
    hub.publish(3)

    assert fast.receive(0) == 3
    assert slow.receive(0) == 3
    assert fast.dropped_frames == 0
    assert slow.dropped_frames == 2
    assert drops == [("inference", 1), ("inference", 1)]


def test_frame_consumer_runs_handler_off_capture_thread_and_keeps_latest() -> None:
    """An expensive callback must run on a worker and discard stale inputs."""
    hub = LatestFrameHub[int]()
    subscription = hub.subscribe("slow")
    entered = threading.Event()
    release = threading.Event()
    handled: list[tuple[int, int]] = []

    def handle(value: int) -> None:
        handled.append((value, threading.get_ident()))
        if value == 1:
            entered.set()
            release.wait(1.0)

    consumer = FrameConsumer(subscription, handle)
    consumer.start()
    hub.publish(1)
    assert entered.wait(1.0)
    for value in range(2, 11):
        hub.publish(value)
    release.set()
    _wait_for(lambda: len(handled) == 2)
    consumer.stop()

    assert [value for value, _ in handled] == [1, 10]
    assert all(ident != threading.get_ident() for _, ident in handled)
    assert consumer.dropped_frames == 8


def test_frame_consumer_reports_failure_and_clears_health_after_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Current health must recover while historical failure remains visible."""
    hub = LatestFrameHub[int]()
    failures: list[Exception] = []

    def handle(value: int) -> None:
        if value == 1:
            raise RuntimeError("invalid model configuration")

    consumer = FrameConsumer(
        hub.subscribe("recovering"),
        handle,
        on_error=failures.append,
        initial_failure_backoff=0,
    )
    with caplog.at_level(logging.ERROR):
        consumer.start()
        hub.publish(1)
        _wait_for(lambda: consumer.failed_frames == 1)
        assert consumer.healthy is False
        assert consumer.consecutive_failures == 1
        hub.publish(2)
        _wait_for(lambda: consumer.processed_frames == 1)
        consumer.stop()

    assert consumer.healthy is True
    assert consumer.consecutive_failures == 0
    assert consumer.last_error is None
    assert isinstance(consumer.last_failure, RuntimeError)
    assert failures == [consumer.last_failure]
    assert "invalid model configuration" in caplog.text


def test_frame_consumer_rate_limits_repeated_error_tracebacks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Repeated failures must remain counted without flooding the log."""
    hub = LatestFrameHub[int]()

    def fail(_: int) -> None:
        raise ValueError("permanent failure")

    consumer = FrameConsumer(
        hub.subscribe("broken"),
        fail,
        error_log_interval=60.0,
        initial_failure_backoff=0,
    )
    with caplog.at_level(logging.ERROR):
        consumer.start()
        for value in range(3):
            hub.publish(value)
            expected = value + 1
            _wait_for(partial(_consumer_failed, consumer, expected))
        consumer.stop()

    messages = [
        record.message
        for record in caplog.records
        if record.message.startswith("Frame consumer broken failed")
    ]
    assert len(messages) == 1
    assert consumer.failed_frames == 3
    assert consumer.consecutive_failures == 3


def test_cpu_camera_subscribe_latest_preserves_legacy_frame_contract() -> None:
    """Subscriptions must complement rather than change BGR read semantics."""
    camera = Camera(enable_preview=False)
    subscription = camera.subscribe_latest("processor")

    camera._publish_frame(bytearray(b"one"), timestamp_ns=1)
    camera._publish_frame(bytearray(b"two"), timestamp_ns=2)
    frame = subscription.receive(0)

    assert frame is not None
    assert frame.image == bytearray(b"two")
    assert frame.timestamp_ns == 2
    assert camera.raw_frame == bytearray(b"two")
    assert camera.consumer_dropped_frames["processor"] == 1


def test_gpu_camera_exposes_public_borrowed_latest_subscription() -> None:
    """GPU integrations can subscribe without accessing capture internals."""
    camera = GpuCamera()
    subscription = camera.subscribe_latest("inference")
    frame = mock_gpu_frame(object(), sequence=4)

    camera._frame_hub.publish(frame)
    received = subscription.receive(0)

    assert received is not None
    assert received is not frame
    assert received.payload() is frame.payload()
    subscription.release(received)


def test_gpu_subscriber_keeps_processing_lease_after_new_publication() -> None:
    """Publishing a successor must not expire a frame inside a worker."""
    camera = GpuCamera()
    entered = threading.Event()
    release = threading.Event()
    payloads: list[object] = []

    def handle(frame: GpuFrame) -> None:
        entered.set()
        assert release.wait(1.0)
        payloads.append(frame.payload())

    consumer = FrameConsumer(camera.subscribe_latest("inference"), handle)
    consumer.start()
    first_payload = object()
    first = mock_gpu_frame(first_payload, sequence=1)
    camera._gpu_publisher.publish(first)
    camera._frame_hub.publish(first)
    assert entered.wait(1.0)

    second = mock_gpu_frame(object(), sequence=2)
    camera._gpu_publisher.publish(second)
    camera._frame_hub.publish(second)
    assert first.valid is False
    release.set()
    _wait_for(lambda: consumer.processed_frames == 2)
    consumer.stop()

    assert payloads[0] is first_payload
    assert consumer.failed_frames == 0


class _SlowRunner:
    """Deterministic runner representing a ten-FPS inference workload."""

    def __init__(self, duration: float = 0.1) -> None:
        self.duration = duration
        self.prepared: list[FrameSpec] = []
        self.closed = False

    def prepare(self, frame_spec: FrameSpec) -> None:
        """Record preparation without loading a model."""
        self.prepared.append(frame_spec)

    @property
    def prepared_frame_spec(self) -> FrameSpec | None:
        """Expose the latest synchronously prepared layout."""
        return self.prepared[-1] if self.prepared else None

    def infer(self, frame: GpuFrame) -> InferenceResult:
        """Sleep for the configured cost and return a model-neutral result."""
        time.sleep(self.duration)
        sequence = frame.sequence
        timestamp_ns = frame.timestamp_ns
        return InferenceResult(
            frame_sequence=sequence,
            frame_timestamp_ns=timestamp_ns,
            inference_time_ns=int(self.duration * 1_000_000_000),
            outputs=(),
        )

    def close(self) -> None:
        """Record runner ownership release."""
        self.closed = True


class _ExpiredRunner(_SlowRunner):
    """Runner simulating expiry before CUDA imports the native buffer."""

    def infer(self, frame: GpuFrame) -> InferenceResult:
        """Reject the lease before starting inference work."""
        raise GpuFrameExpiredError("expired before import")


class _OutputRunner(_SlowRunner):
    """Runner exposing one named output shape for health diagnostics."""

    def infer(self, frame: GpuFrame) -> InferenceResult:
        """Return one portable tensor descriptor without model-specific data."""
        return InferenceResult(
            frame_sequence=frame.sequence,
            frame_timestamp_ns=frame.timestamp_ns,
            inference_time_ns=10,
            outputs=(TensorOutput("boxes", (1, 84, 8400), "float32", object()),),
        )


def test_inference_consumer_reuses_runner_prepared_before_camera_capture() -> None:
    """The first borrowed frame must not trigger a second expensive prepare."""
    hub = LatestFrameHub[GpuFrame]()
    runner = _SlowRunner(duration=0)
    frame = mock_gpu_frame(object(), width=1280, height=720)
    spec = FrameSpec.from_gpu_frame(frame)
    runner.prepare(spec)
    inference = InferenceConsumer(hub.subscribe("prepared"), runner)

    inference.start()
    hub.publish(frame)
    _wait_for(lambda: inference.processed_frames == 1)
    health = inference.health()
    inference.stop()

    assert runner.prepared == [spec]
    assert inference.prepared_frame_spec == spec
    assert health["running"] is True
    assert health["processed_frames"] == 1
    assert health["output_shapes"] == {}


def test_inference_health_reports_named_output_shapes() -> None:
    """Diagnostics must reveal decoder contract mismatches without tensor data."""
    hub = LatestFrameHub[GpuFrame]()
    inference = InferenceConsumer(hub.subscribe("outputs"), _OutputRunner())
    inference.start()
    hub.publish(mock_gpu_frame(object()))
    _wait_for(lambda: inference.processed_frames == 1)
    health = inference.health()
    inference.stop()

    assert health["output_shapes"] == {"boxes": [1, 84, 8400]}


def test_inference_context_cleanup_does_not_mask_primary_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A secondary stop timeout must preserve the application failure."""
    inference = InferenceConsumer(
        LatestFrameHub[GpuFrame]().subscribe("cleanup"),
        _SlowRunner(duration=0),
    )

    def fail_stop(timeout: float | None = 10.0) -> None:
        raise RuntimeError(f"cleanup timeout: {timeout}")

    monkeypatch.setattr(inference, "stop", fail_stop)

    with pytest.raises(ValueError, match="primary failure"), inference:
        raise ValueError("primary failure")


def test_inference_consumer_classifies_expired_input_as_drop() -> None:
    """A pre-inference lease expiry is a drop rather than runner failure."""
    hub = LatestFrameHub[GpuFrame]()
    inference = InferenceConsumer(
        hub.subscribe("expired"),
        _ExpiredRunner(duration=0),
        initial_failure_backoff=0,
    )
    inference.start()
    hub.publish(mock_gpu_frame(object()))
    _wait_for(lambda: inference.dropped_frames == 1)
    inference.stop()

    assert inference.failed_frames == 0
    assert inference.consecutive_failures == 0
    assert inference.last_error is None
    assert inference.last_failure is None
    assert inference.healthy is True


class _FakeJPEGSource:
    """Condition-backed latest JPEG source matching the browser stream API."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame_number = 0
        self._jpeg: bytes | None = None
        self._running = True

    @property
    def running(self) -> bool:
        """Whether publications remain active."""
        return self._running

    @property
    def frame_number(self) -> int:
        """Newest preview frame number."""
        with self._condition:
            return self._frame_number

    def publish(self, jpeg: bytes) -> None:
        """Replace the latest preview slot."""
        with self._condition:
            self._frame_number += 1
            self._jpeg = jpeg
            self._condition.notify_all()

    def wait_for_jpeg(
        self,
        previous_frame_number: int,
        timeout: float = 2.0,
    ) -> tuple[int, bytes | None]:
        """Wait for a preview newer than the caller's last observation."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._frame_number != previous_frame_number
                or not self._running,
                timeout,
            )
            return self._frame_number, self._jpeg


def test_slow_inference_does_not_limit_preview_or_accumulate_backlog() -> None:
    """Ten-FPS inference must coexist with fresh thirty-FPS preview frames."""
    frame_hub = LatestFrameHub[GpuFrame]()
    runner = _SlowRunner(duration=0.1)
    inference = InferenceConsumer(
        frame_hub.subscribe("inference"),
        runner,
    )
    preview_source = _FakeJPEGSource()
    contexts: list[PreviewOverlayContext] = []

    def render(
        jpeg: bytes,
        result: InferenceResult | None,
        context: PreviewOverlayContext,
    ) -> bytes:
        contexts.append(context)
        return jpeg + (b"-overlay" if result is not None else b"-pending")

    preview = InferencePreviewSource(
        preview_source,
        inference,
        render,
        max_fps=120.0,
        read_timeout=0.05,
    )
    inference.start()
    preview.start()

    started = time.monotonic()
    for sequence in range(1, 31):
        timestamp_ns = time.monotonic_ns()
        frame_hub.publish(
            mock_gpu_frame(
                object(),
                sequence=sequence,
                timestamp_ns=timestamp_ns,
            )
        )
        preview_source.publish(f"jpeg-{sequence}".encode())
        _wait_for(
            partial(_preview_processed, preview, sequence),
            timeout=0.2,
        )
        target = started + sequence / 30
        time.sleep(max(target - time.monotonic(), 0))

    _wait_for(
        lambda: inference.latest_result is not None
        and inference.latest_result.frame_sequence == 30,
        timeout=0.5,
    )
    preview.stop()
    inference.stop()

    assert preview.processed_frames == 30
    assert inference.processed_frames < preview.processed_frames
    assert inference.dropped_frames > 0
    assert inference.latest_result is not None
    assert inference.latest_result.frame_sequence == 30
    assert preview.frame_number == 30
    assert preview.result_frame_timestamp_ns == (
        inference.latest_result.frame_timestamp_ns
    )
    assert preview.detection_age_ns is not None
    assert contexts[-1].result_frame_sequence is not None
    assert runner.closed is True
