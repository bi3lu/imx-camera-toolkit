"""Verify cached inference reads and application ownership."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from imx_camera_toolkit import GpuFrame
from imx_camera_toolkit._internal.mcp import inference_adapter
from imx_camera_toolkit._internal.mcp.camera_adapter import CameraReadAdapter
from imx_camera_toolkit._internal.mcp.contracts import InferenceStatus
from imx_camera_toolkit._internal.mcp.errors import MCPError
from imx_camera_toolkit._internal.mcp.host import MCPHost
from imx_camera_toolkit._internal.mcp.inference_adapter import InferenceReadAdapter
from imx_camera_toolkit._internal.mcp.snapshot_adapter import CameraSnapshotAdapter
from imx_camera_toolkit.consumers import InferenceConsumer, LatestFrameHub
from imx_camera_toolkit.inference import FrameSpec, InferenceResult, TensorOutput
from imx_camera_toolkit.testing import mock_gpu_frame


class _Worker:
    """Expose counters and a slot that records each read."""

    running = False
    result: InferenceResult | None = None
    processed_frames = 0
    failed_frames = 0
    dropped_frames = 1
    consecutive_failures = 0
    reads = 0

    @property
    def latest_result(self) -> InferenceResult | None:
        """Record one poll of the latest slot."""
        self.reads += 1
        return self.result


def test_worker_status_projects_cached_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Project lifecycle and timings without reading worker or tensor details."""
    monkeypatch.setattr(
        inference_adapter, "time", SimpleNamespace(monotonic_ns=lambda: 25_000_000)
    )
    worker = _Worker()
    adapter = InferenceReadAdapter.from_worker(worker)
    assert worker.reads == 0
    assert adapter.status() == InferenceStatus("stopped", 0, 0, 1, 0)

    worker.running = True
    assert adapter.status() == InferenceStatus("initializing", 0, 0, 1, 0)
    with pytest.raises(MCPError) as empty:
        adapter.latest_result()
    assert empty.value.code == "no_inference_result"

    worker.failed_frames = worker.consecutive_failures = 1
    assert adapter.status() == InferenceStatus("degraded", 0, 1, 1, 1, "worker_failure")

    worker.consecutive_failures = 0
    worker.processed_frames = 2
    worker.result = InferenceResult(
        4,
        5_000_000,
        2_500_000,
        (TensorOutput("private output", (1,), "float32", object()),),
        metadata={"private": object()},
        overlays=(object(),),
        capture_timestamp_ns=3,
    )
    ready = InferenceStatus("running", 2, 1, 1, 0, inference_time_ms=2.5, age_ms=20.0)
    assert adapter.status() == ready
    assert adapter.latest_result() is worker.result
    worker.running = False
    assert adapter.status() == replace(ready, state="stopped")
    assert worker.reads == 7


@pytest.mark.parametrize("timestamp", [0, 26_000_000])
def test_result_only_source_has_optional_diagnostics(
    timestamp: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report unknown health, unavailable age and out-of-range duration as null."""
    monkeypatch.setattr(
        inference_adapter, "time", SimpleNamespace(monotonic_ns=lambda: 25_000_000)
    )
    source = _Worker()
    source.result = InferenceResult(1, timestamp, 10**400, ())
    assert InferenceReadAdapter(source).status() == InferenceStatus("unknown")
    assert source.reads == 1


class _Runner:
    """Gate one real consumer call and record runner activity and its thread."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.calls: list[tuple[str, int]] = []

    def prepare(self, frame_spec: FrameSpec) -> None:
        """Record preparation on the inference thread."""
        self.calls.append(("prepare", threading.get_ident()))

    def infer(self, frame: GpuFrame) -> InferenceResult:
        """Wait until status polling has sampled the busy worker."""
        self.calls.append(("infer", threading.get_ident()))
        self.entered.set()
        if not self.resume.wait(5):
            raise RuntimeError("test did not release inference")
        return InferenceResult(frame.sequence, frame.timestamp_ns, 1_000_000, ())

    def close(self) -> None:
        """Record cleanup by the application."""
        self.calls.append(("close", threading.get_ident()))


def test_parallel_reads_leave_worker_and_pending_response_independent() -> None:
    """Publish a result while a caller holds an earlier status response."""
    frames = LatestFrameHub[GpuFrame]()
    runner = _Runner()
    worker = InferenceConsumer(frames.subscribe("inference"), runner)
    results = worker.subscribe_results("test")
    adapter = InferenceReadAdapter.from_worker(worker)

    async def scenario() -> None:
        host = MCPHost(
            CameraReadAdapter(None),
            CameraSnapshotAdapter(None),
            inference_adapter=adapter,
        )
        pending = asyncio.Event()
        send = asyncio.Event()

        async def slow_response() -> InferenceStatus:
            """Hold the response after its cached source read."""
            status = host.inference_status()
            pending.set()
            await send.wait()
            return status

        async with host:
            response = asyncio.create_task(slow_response())
            try:
                await asyncio.wait_for(pending.wait(), 5)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    polls = [pool.submit(adapter.status) for _ in range(8)]
                    assert all(
                        poll.result(timeout=5).state == "initializing" for poll in polls
                    )

                runner.resume.set()
                result = await asyncio.to_thread(results.receive, 5)
                assert result is not None
                assert not response.done()
                assert host.inference_latest_result() is result
                assert host.inference_status().state == "running"
                send.set()
                assert (await asyncio.wait_for(response, 5)).state == "initializing"
            finally:
                response.cancel()
                await asyncio.gather(response, return_exceptions=True)
        assert worker.running

    with worker:
        try:
            frames.publish(mock_gpu_frame(object()))
            assert runner.entered.wait(5)
            thread_ident = worker.thread_ident
            assert thread_ident is not None
            asyncio.run(scenario())
            assert runner.calls == [("prepare", thread_ident), ("infer", thread_ident)]
        finally:
            runner.resume.set()
    assert runner.calls[-1] == ("close", threading.get_ident())


def test_unconfigured_host_and_lifespan_errors() -> None:
    """Reject unavailable bindings without opening or reading any worker."""

    async def scenario() -> None:
        host = MCPHost(CameraReadAdapter(None), CameraSnapshotAdapter(None))
        with pytest.raises(MCPError) as inactive:
            host.inference_latest_result()
        assert inactive.value.code == "camera_unavailable"

        async with host:
            for read in (host.inference_status, host.inference_latest_result):
                with pytest.raises(MCPError) as unconfigured:
                    read()
                assert unconfigured.value.code == "inference_not_configured"

    asyncio.run(scenario())
