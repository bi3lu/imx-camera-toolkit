"""Exercise bounded snapshots with deterministic sources and a synthetic JPEG."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from imx_camera_toolkit import Camera, GpuCamera
from imx_camera_toolkit._internal.camera.publishing import jpeg
from imx_camera_toolkit._internal.camera.publishing.jpeg_slot import (
    JPEGPreview,
    JPEGPublication,
)
from imx_camera_toolkit._internal.mcp import snapshot_adapter
from imx_camera_toolkit._internal.mcp.errors import MCPError
from imx_camera_toolkit._internal.mcp.snapshot_adapter import (
    HARD_SNAPSHOT_LIMITS,
    CameraSnapshotAdapter,
)

_NOW = 30_000_000_000


@pytest.fixture
def photo() -> bytes:
    """Load the one-pixel JPEG used by the stdio client probe."""
    return (
        Path(__file__).resolve().parents[2] / "examples/mcp_runtime_spike/synthetic.jpg"
    ).read_bytes()


@dataclass
class Preview:
    """Minimal mutable source that exposes only its fixed preview."""

    current: JPEGPreview
    reads: int = 0

    def _jpeg_preview(self) -> JPEGPreview:
        """Return a single sample and count source accesses."""
        self.reads += 1
        return self.current


@pytest.fixture
def source(photo: bytes) -> Preview:
    """Supply one complete JPEG with exactly known acquisition provenance."""
    return Preview(
        JPEGPreview(
            "running", True, True, 1, 1, 65, JPEGPublication(photo, 3, 1, 1, 65, _NOW)
        )
    )


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Freeze the adapter clock independently of asyncio's scheduler."""
    now = Mock(return_value=_NOW)
    monkeypatch.setattr(snapshot_adapter, "time", SimpleNamespace(monotonic_ns=now))
    return now


@pytest.mark.parametrize(
    "arguments",
    [
        {"unknown": 1},
        {"camera_id": "secondary"},
        {"max_width": 1921},
        {"max_height": 0},
        {"jpeg_quality": 96},
        {"jpeg_quality": None},
        {"max_age_ms": 1.0},
        {"max_age_ms": True},
        {"max_age_ms": float("nan")},
        {"wait_ms": -1},
        {"wait_ms": 1001},
    ],
)
def test_validates_before_access(source: Preview, arguments: dict[str, object]) -> None:
    """Reject invalid inputs before accessing the camera."""
    with pytest.raises(MCPError) as caught:
        asyncio.run(CameraSnapshotAdapter(source).snapshot(arguments))
    assert caught.value.code == "invalid_argument"
    assert source.reads == 0


@pytest.mark.parametrize(
    ("changes", "code", "reason"),
    [
        ({"state": "stopped"}, "camera_unavailable", None),
        ({"state": "failed"}, "camera_unavailable", None),
        ({"state": "recovering"}, "camera_starting", None),
        ({"state": "starting"}, "camera_starting", None),
        ({"enabled": False}, "snapshot_unavailable", "no_source"),
        ({"acquisition_timing": False}, "snapshot_unavailable", "timing_unavailable"),
        ({"width": 1921}, "snapshot_unavailable", "constraints_unsupported"),
        ({"quality": 100}, "snapshot_unavailable", "constraints_unsupported"),
        ({"publication": None}, "snapshot_unavailable", "no_frame"),
    ],
)
def test_source_errors(
    source: Preview, changes: dict[str, object], code: str, reason: str | None
) -> None:
    """Return stable source errors and the correct retry policy."""
    source.current = replace(source.current, **changes)  # type: ignore[arg-type]
    with pytest.raises(MCPError) as caught:
        asyncio.run(CameraSnapshotAdapter(source).snapshot())
    payload = caught.value.to_payload()
    assert payload["code"] == code
    assert payload["retryable"] == (reason == "no_frame" or code == "camera_starting")
    assert payload.get("details") == (
        {"reason": reason} if reason is not None else None
    )


def test_freshness_boundary(source: Preview, clock: Mock, photo: bytes) -> None:
    """Accept exactly the permitted age and preserve the original detached bytes."""
    clock.return_value = _NOW + 10_000_000
    adapter = CameraSnapshotAdapter(source)
    options = {"max_age_ms": 10, "max_width": 1, "max_height": 1, "jpeg_quality": 65}
    result = asyncio.run(adapter.snapshot(options))
    assert result.jpeg is photo
    assert result.metadata.age_ms == 10
    assert result.metadata.frame_sequence == 3
    assert result.metadata.jpeg_quality == 65
    assert (result.metadata.width, result.metadata.height) == (1, 1)

    clock.return_value += 1
    with pytest.raises(MCPError) as caught:
        asyncio.run(adapter.snapshot(options))
    assert caught.value.code == "snapshot_stale"
    assert caught.value.to_payload()["details"] == {
        "age_ms": 10.000001,
        "max_age_ms": 10,
    }


def test_fixed_source_rejects_explicit_quality(source: Preview) -> None:
    """Report fixed-source constraints even when a wait was requested."""
    with pytest.raises(MCPError) as caught:
        asyncio.run(
            CameraSnapshotAdapter(source).snapshot(
                {"jpeg_quality": 64, "wait_ms": 1000}
            )
        )
    assert caught.value.to_payload()["details"] == {"reason": "constraints_unsupported"}
    assert source.reads == 1


def test_exact_byte_budgets(source: Preview, clock: Mock, photo: bytes) -> None:
    """Enforce both independently inclusive limits with actual base64 expansion."""
    del clock
    publication = source.current.publication
    assert publication is not None
    remaining = HARD_SNAPSHOT_LIMITS.max_jpeg_bytes - len(photo)
    segments = []
    while remaining:
        length = min(remaining, 65537)
        segments.append(
            b"\xff\xef" + (length - 2).to_bytes(2, "big") + bytes(length - 4)
        )
        remaining -= length
    data = photo[:2] + b"".join(segments) + photo[2:]
    source.current = replace(
        source.current, publication=replace(publication, data=data)
    )
    result = asyncio.run(CameraSnapshotAdapter(source).snapshot())
    assert result.metadata.jpeg_bytes == 524_288
    wire = snapshot_adapter._estimated_wire_bytes(result.metadata)
    for limits in (
        replace(HARD_SNAPSHOT_LIMITS, max_jpeg_bytes=len(data) - 1),
        replace(HARD_SNAPSHOT_LIMITS, max_response_bytes=wire - 1),
    ):
        with pytest.raises(MCPError) as caught:
            asyncio.run(CameraSnapshotAdapter(source, limits=limits).snapshot())
        assert caught.value.code == "payload_too_large"

    exact = replace(HARD_SNAPSHOT_LIMITS, max_response_bytes=wire)
    assert asyncio.run(CameraSnapshotAdapter(source, limits=exact).snapshot()) == result


@pytest.mark.parametrize(
    "malform",
    [
        lambda data: data[:-2],
        lambda data: data + b"trailing",
        lambda data: data[:2] + b"\xff\xe1\x00\x08Exif\x00\x00" + data[2:],
        lambda data: b"\xff\xd8\xff\xd9",
    ],
)
def test_bad_jpeg_is_sanitized(
    source: Preview, clock: Mock, malform: Callable[[bytes], bytes]
) -> None:
    """Reject malformed, truncated or EXIF-bearing data with one safe error."""
    del clock
    publication = source.current.publication
    assert publication is not None
    source.current = replace(
        source.current, publication=replace(publication, data=malform(publication.data))
    )
    with pytest.raises(MCPError) as caught:
        asyncio.run(CameraSnapshotAdapter(source).snapshot())
    assert caught.value.code == "internal_error"
    assert isinstance(caught.value.__cause__, ValueError)


def test_concurrent_wait_overwrite_and_cancellation(
    source: Preview, clock: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two waiting readers get the latest publication and a third can cancel."""

    async def run() -> None:
        """Exercise the adapter on a dedicated event loop."""
        old = source.current.publication
        assert old is not None
        clock.return_value = _NOW + 2_000_000
        polling = asyncio.Queue[None]()
        resume = asyncio.Event()

        async def sleep(delay: float) -> None:
            """Hold request polling until the test publishes two replacements."""
            assert 0 <= delay <= 0.01
            polling.put_nowait(None)
            await resume.wait()

        monkeypatch.setattr(snapshot_adapter, "asyncio", SimpleNamespace(sleep=sleep))
        adapter = CameraSnapshotAdapter(source)
        callers = [
            asyncio.create_task(adapter.snapshot({"max_age_ms": 1, "wait_ms": 100}))
            for _ in range(3)
        ]
        try:
            for _ in callers:
                await asyncio.wait_for(polling.get(), 1)
            callers[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(callers[0], 1)
            source.current = replace(
                source.current, publication=replace(old, sequence=4)
            )
            source.current = replace(
                source.current,
                publication=replace(old, sequence=5, acquired_at_ns=clock.return_value),
            )
            resume.set()
            results = await asyncio.wait_for(asyncio.gather(*callers[1:]), 1)
            assert [result.metadata.frame_sequence for result in results] == [5, 5]
        finally:
            for caller in callers:
                caller.cancel()
            await asyncio.gather(*callers, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("stop", [False, True])
def test_wait_rechecks_staleness_deadline_and_shutdown(
    source: Preview, clock: Mock, monkeypatch: pytest.MonkeyPatch, stop: bool
) -> None:
    """Ignore stale arrivals and stop waiting at the outer deadline or shutdown."""

    async def run() -> None:
        """Exercise the adapter on a dedicated event loop."""
        previous = source.current.publication
        assert previous is not None
        source.current = replace(source.current, publication=None)
        deadline_ns = _NOW + 3_000_000
        sleeps = 0

        async def sleep(delay: float) -> None:
            """Advance the clock deterministically and inject a stale arrival."""
            nonlocal sleeps
            sleeps += 1
            if sleeps == 1:
                assert delay == 0.003
                clock.return_value = _NOW + 2_000_000
                source.current = replace(source.current, publication=previous)
            elif stop:
                source.current = replace(source.current, state="stopped")
            else:
                assert delay == 0.001
                clock.return_value = deadline_ns

        monkeypatch.setattr(snapshot_adapter, "asyncio", SimpleNamespace(sleep=sleep))
        with pytest.raises(MCPError) as caught:
            await CameraSnapshotAdapter(source).snapshot(
                {"max_age_ms": 1, "wait_ms": 1_000}, deadline_ns=deadline_ns
            )
        assert sleeps == 2
        assert caught.value.code == (
            "camera_unavailable" if stop else "request_timeout"
        )

    asyncio.run(run())


def test_cpu_record_and_legacy_preview_share_one_publication(
    monkeypatch: pytest.MonkeyPatch, photo: bytes
) -> None:
    """Attach the original CPU acquisition, preserving the existing HTTP slot."""
    encoder = Mock(return_value=(True, SimpleNamespace(tobytes=lambda: photo)))
    monkeypatch.setattr(
        jpeg, "cv2_module", SimpleNamespace(imencode=encoder, IMWRITE_JPEG_QUALITY=1)
    )
    camera = Camera(output_width=1, output_height=1, enable_preview=True)
    camera._running.set()
    camera._recovery.set_state("running")
    acquired_at_ns = time.monotonic_ns()
    try:
        camera._publish_frame(object(), acquired_at_ns)
        publication = camera._jpeg_preview().publication
        assert publication is not None
        assert publication.acquired_at_ns == acquired_at_ns
        assert camera.wait_for_jpeg(-1, 0) == (publication.sequence, photo)
        result = asyncio.run(CameraSnapshotAdapter(camera).snapshot())
        assert result.jpeg is publication.data
        assert result.metadata.frame_sequence == publication.sequence
        encoder.assert_called_once()
        camera._preview_started_ns = acquired_at_ns + 1
        assert camera._jpeg_preview().publication is None
        assert camera.latest_jpeg() == photo
    finally:
        camera.stop()
    assert camera._jpeg_preview().publication is None


def test_gpu_preview_never_acquires_a_raw_frame(
    monkeypatch: pytest.MonkeyPatch, photo: bytes
) -> None:
    """Keep detached GPU JPEGs available to HTTP and fail closed for MCP age."""
    camera = GpuCamera(enable_preview=True)
    raw_read = Mock(side_effect=AssertionError("unexpected raw frame acquisition"))
    monkeypatch.setattr(camera, "read", raw_read)
    camera._running.set()
    camera._recovery.set_state("running")
    try:
        assert camera._preview_publisher.publish(
            photo,
            width=camera.config.output_width,
            height=camera.config.output_height,
            quality=camera.config.quality,
        )
        assert camera.wait_for_jpeg(-1, 0) == (1, photo)
        with pytest.raises(MCPError) as caught:
            asyncio.run(CameraSnapshotAdapter(camera).snapshot({"wait_ms": 1_000}))
        assert caught.value.to_payload()["details"] == {"reason": "timing_unavailable"}
        raw_read.assert_not_called()
    finally:
        camera.stop()
    assert camera._jpeg_preview().publication is None
