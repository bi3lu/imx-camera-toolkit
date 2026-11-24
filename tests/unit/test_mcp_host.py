"""Verify one application binding across clients and lifecycle failures."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from time import monotonic_ns
from unittest.mock import Mock

import pytest

from imx_camera_toolkit import CameraStats
from imx_camera_toolkit._internal.camera.publishing.jpeg_slot import (
    JPEGPreview,
    JPEGPublication,
)
from imx_camera_toolkit._internal.mcp.camera_adapter import CameraReadAdapter
from imx_camera_toolkit._internal.mcp.contracts import (
    HARD_SNAPSHOT_LIMITS,
    CameraCapabilities,
    CameraSnapshot,
    SnapshotCapabilities,
)
from imx_camera_toolkit._internal.mcp.errors import MCPError
from imx_camera_toolkit._internal.mcp.host import MCPHost
from imx_camera_toolkit._internal.mcp.snapshot_adapter import CameraSnapshotAdapter


class FakeCamera:
    """Expose one replaceable publication and record lifecycle calls."""

    def __init__(self) -> None:
        """Create an unopened source with no publication."""
        self.running = False
        self.state = "stopped"
        self.publication: JPEGPublication | None = None
        self.events: list[str] = []
        self.preview_reads = 0
        self.two_reads = asyncio.Event()
        self.fail_start = False

    def start(self) -> None:
        """Enter the capture run before optionally simulating failure."""
        self.events.append("start")
        self.running = True
        self.state = "running"
        if self.fail_start:
            raise RuntimeError("partial camera startup")

    def stop(self) -> None:
        """Record shutdown and discard the latest publication."""
        self.events.append("stop")
        self.running = False
        self.state = "stopped"
        self.publication = None

    def stats(self) -> CameraStats:
        """Read current source diagnostics and publication readiness."""
        return CameraStats(
            0,
            0,
            0.0,
            None,
            0,
            0,
            self.running,
            state=self.state,
            has_published_frame=self.publication is not None,
        )

    def _jpeg_preview(self) -> JPEGPreview:
        """Signal when two calls have sampled the shared source."""
        self.preview_reads += 1
        if self.preview_reads == 2:
            self.two_reads.set()
        return JPEGPreview(self.state, True, True, 1, 1, 65, self.publication)


@pytest.fixture
def photo() -> bytes:
    """Load the complete one-pixel JPEG from the protocol fixtures."""
    return (
        Path(__file__).resolve().parents[2] / "examples/mcp_runtime_spike/synthetic.jpg"
    ).read_bytes()


def test_host_validates_decoded_arguments_before_reading() -> None:
    """Forward explicit null to validation and accept the registered ID."""

    async def scenario() -> None:
        """Exercise tool calls on a running borrowed host."""
        camera = FakeCamera()
        provider = Mock(return_value=CameraCapabilities())
        async with MCPHost(
            CameraReadAdapter(camera, capability_source=provider),
            CameraSnapshotAdapter(camera),
        ) as host:
            with pytest.raises(MCPError, match="Invalid tool arguments"):
                host.status(None)
            with pytest.raises(MCPError, match="Invalid tool arguments"):
                host.capabilities({"camera_id": "other"})
            with pytest.raises(MCPError, match="Invalid tool arguments"):
                await host.snapshot(None)
            provider.assert_not_called()
            assert camera.preview_reads == 0
            assert host.status({"camera_id": "primary"}).state == "unavailable"
            assert host.capabilities({"camera_id": "primary"}) == CameraCapabilities()

    asyncio.run(scenario())


def test_host_projects_its_bound_snapshot_source_and_limits(photo: bytes) -> None:
    """Report configured preview before publication and known timing afterward."""

    async def scenario() -> None:
        camera = FakeCamera()
        limits = replace(HARD_SNAPSHOT_LIMITS, max_width=1280)
        async with MCPHost(
            CameraReadAdapter(camera, backend="cpu"),
            CameraSnapshotAdapter(camera, limits=limits),
        ) as host:
            assert host.capabilities().snapshot == SnapshotCapabilities(
                None, True, limits
            )
            camera.start()
            camera.publication = JPEGPublication(photo, 1, 1, 1, 65, monotonic_ns())
            result = host.capabilities()
            assert result.frame_format == "BGR_CPU"
            assert result.snapshot == SnapshotCapabilities(True, True, limits)
        camera.stop()

        async with MCPHost(
            CameraReadAdapter(camera), CameraSnapshotAdapter(None, limits=limits)
        ) as host:
            assert host.capabilities().snapshot == SnapshotCapabilities(
                None, False, limits
            )

    asyncio.run(scenario())


@pytest.mark.parametrize("standalone", [False, True])
def test_clients_share_capture(photo: bytes, standalone: bool) -> None:
    """Share one publication across concurrent calls, reconnect and recovery."""

    async def scenario() -> None:
        camera = FakeCamera()

        if not standalone:
            camera.start()

        capabilities = CameraCapabilities()
        reads = CameraReadAdapter(
            camera, backend="cpu", capability_source=lambda: capabilities
        )
        snapshots = CameraSnapshotAdapter(camera)
        host = MCPHost(reads, snapshots, owned_camera=camera if standalone else None)

        with pytest.raises(MCPError, match="Camera unavailable"):
            host.status()

        async with host:
            assert host.status().state == "starting"
            assert host.capabilities() is capabilities
            first = asyncio.create_task(host.snapshot({"wait_ms": 1000}))
            second = asyncio.create_task(host.snapshot({"wait_ms": 1000}))
            await asyncio.wait_for(camera.two_reads.wait(), 1)
            assert camera.events == ["start"]
            camera.publication = JPEGPublication(photo, 3, 1, 1, 65, monotonic_ns())
            results = await asyncio.wait_for(asyncio.gather(first, second), 1)
            assert [result.metadata.frame_sequence for result in results] == [3, 3]
            assert all(result.jpeg is photo for result in results)
            assert host.status().state == "ready"

            camera.state = "recovering"
            assert host.status().state == "recovering"

            with pytest.raises(MCPError) as error:
                await host.snapshot()

            assert error.value.code == "camera_starting"
            camera.state = "running"
            camera.publication = JPEGPublication(photo, 4, 1, 1, 65, monotonic_ns())
            assert (await host.snapshot()).metadata.frame_sequence == 4

        await asyncio.gather(host.aclose(), host.aclose())

        if not standalone:
            assert camera.events == ["start"]
            assert camera.running

            async with MCPHost(reads, snapshots) as replacement:
                assert replacement.status().state == "ready"
                assert (await replacement.snapshot()).metadata.frame_sequence == 4

            camera.stop()

        assert camera.events == ["start", "stop"]

    asyncio.run(scenario())


def test_client_cancellation_then_host_shutdown() -> None:
    """Disconnect one client without stopping capture or the other read."""

    async def scenario() -> None:
        camera = FakeCamera()

        async with MCPHost(
            CameraReadAdapter(camera),
            CameraSnapshotAdapter(camera),
            owned_camera=camera,
        ) as host:
            first = asyncio.create_task(host.snapshot({"wait_ms": 1000}))
            second = asyncio.create_task(host.snapshot({"wait_ms": 1000}))
            await asyncio.wait_for(camera.two_reads.wait(), 1)
            first.cancel()

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first, 1)

            assert not second.done()
            assert camera.events == ["start"]

            await asyncio.wait_for(host.aclose(), 1)

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(second, 1)

            reads = camera.preview_reads

            with pytest.raises(MCPError, match="Camera unavailable"):
                await host.snapshot()

            with pytest.raises(MCPError, match="Camera unavailable"):
                host.capabilities()

            assert camera.preview_reads == reads

        assert camera.events == ["start", "stop"]

    asyncio.run(scenario())


def test_partial_startup_and_unentered_close() -> None:
    """Roll back a failed capture startup and leave an unentered camera untouched."""

    async def scenario() -> None:
        camera = FakeCamera()
        reads = CameraReadAdapter(camera)
        snapshots = CameraSnapshotAdapter(camera)
        unopened = MCPHost(reads, snapshots, owned_camera=camera)
        await unopened.aclose()
        assert camera.events == []

        with pytest.raises(RuntimeError, match="only be entered once"):
            async with unopened:
                pytest.fail("closed host accepted startup")

        camera.fail_start = True
        host = MCPHost(reads, snapshots, owned_camera=camera)

        with pytest.raises(RuntimeError, match="partial camera startup"):
            async with host:
                pytest.fail("failed startup enabled dispatch")

        await host.aclose()
        assert not camera.running
        assert camera.events == ["start", "stop"]

    asyncio.run(scenario())


def test_later_startup_failure_exits_entered_host() -> None:
    """Unwind an entered host if transport startup fails inside its lifespan."""

    async def scenario() -> None:
        camera = FakeCamera()
        host = MCPHost(
            CameraReadAdapter(camera),
            CameraSnapshotAdapter(camera),
            owned_camera=camera,
        )

        with pytest.raises(RuntimeError, match="could not start transport"):
            async with host:
                raise RuntimeError("could not start transport")

        await host.aclose()
        assert camera.events == ["start", "stop"]

    asyncio.run(scenario())


def test_cancelled_close_still_drains_read_before_stopping_camera() -> None:
    """Delay shutdown completion until a cancelled task releases its resources."""

    async def scenario() -> None:
        camera = FakeCamera()
        started = asyncio.Event()
        draining = asyncio.Event()
        release = asyncio.Event()
        closing_started = asyncio.Event()

        class DelayedCleanup(CameraSnapshotAdapter):
            """Hold cleanup at a deterministic point after cancellation."""

            async def snapshot(
                self,
                arguments: object = None,
                *,
                deadline_ns: int | None = None,
            ) -> CameraSnapshot:
                """Simulate an asynchronous operation with cleanup to complete."""
                started.set()

                try:
                    await asyncio.Future[None]()

                finally:
                    draining.set()
                    await asyncio.wait_for(release.wait(), 1)
                    camera.events.append("released")

                pytest.fail("the test read should have been cancelled")

        async with MCPHost(
            CameraReadAdapter(camera), DelayedCleanup(camera), owned_camera=camera
        ) as host:
            read = asyncio.create_task(host.snapshot())
            await asyncio.wait_for(started.wait(), 1)
            read.cancel()
            await asyncio.wait_for(draining.wait(), 1)

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(read, 1)

            async def close() -> None:
                """Signal when this caller begins the shared close."""
                closing_started.set()
                await host.aclose()

            closing = asyncio.create_task(close())
            second_close = asyncio.create_task(host.aclose())
            await asyncio.wait_for(closing_started.wait(), 1)
            closing.cancel()

            try:
                assert camera.events == ["start"]
                assert not closing.done()

                with pytest.raises(MCPError, match="Camera unavailable"):
                    host.status()

            finally:
                release.set()

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(closing, 1)

            await asyncio.wait_for(second_close, 1)

        assert camera.events == ["start", "released", "stop"]

    asyncio.run(scenario())
