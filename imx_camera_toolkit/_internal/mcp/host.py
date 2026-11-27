"""Manage one camera binding and its reads independently of the MCP transport."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from ..inference.contracts import InferenceResult
from .camera_adapter import CameraReadAdapter
from .contracts import (
    CameraCapabilities,
    CameraSnapshot,
    CameraStatus,
    InferenceStatus,
    SnapshotLimits,
)
from .errors import MCPError
from .inference_adapter import (
    InferenceReadAdapter,
    inference_duration_ms,
    require_fresh_inference,
)
from .inference_serialization import (
    InferenceLatest,
    InferencePayload,
    InferencePayloadSerializer,
)
from .snapshot_adapter import CameraSnapshotAdapter
from .validation import (
    EMPTY_ARGUMENTS,
    parse_inference_latest_input,
    parse_snapshot_input,
    validate_inference_status_input,
)

_MAX_ACTIVE_SNAPSHOTS = 2
_MAPPING_TIMEOUT_NS = 250_000_000


class CaptureLifecycle(Protocol):
    """Capture owned exclusively by a standalone application."""

    def start(self) -> None:
        """Start capture from the application's startup path."""
        ...

    def stop(self) -> None:
        """Close capture, including after a partial startup."""
        ...


class MCPHost:
    """Share one binding for the lifetime of an application host.

    Enter and use each host on one event loop. The application supplies the
    adapters for its primary camera. The host owns snapshot wait tasks, its
    optional mapper thread and, when supplied, the standalone camera.
    Diagnostics must use nonblocking cached reads. Callers enforce tool
    authorization and request rate limits before dispatch.
    """

    def __init__(
        self,
        read_adapter: CameraReadAdapter,
        snapshot_adapter: CameraSnapshotAdapter,
        *,
        owned_camera: CaptureLifecycle | None = None,
        inference_adapter: InferenceReadAdapter | None = None,
        inference_serializer: InferencePayloadSerializer | None = None,
    ) -> None:
        """Bind the adapters without opening or reading the source.

        Args:
            read_adapter: Cached diagnostics for the primary camera.
            snapshot_adapter: Preview reads for the same camera.
            owned_camera: Unstarted primary camera transferred by an explicit
                standalone application. Omit for application-owned capture.
            inference_adapter: Cached reads from an application-owned worker.
            inference_serializer: Optional projection for this inference source.

        Raises:
            ValueError: If a mapper is configured without an inference source.
        """
        if inference_serializer is not None and inference_adapter is None:
            raise ValueError("an inference mapper requires a bound inference source")

        self._read_adapter = read_adapter
        self._snapshot_adapter = snapshot_adapter
        self._owned_camera = owned_camera
        self._inference_adapter = inference_adapter
        self._inference_serializer = inference_serializer
        self._mapping_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-inference")
            if inference_serializer is not None
            else None
        )
        self._mapping: asyncio.Future[InferencePayload] | None = None
        self._entered = False
        self._accepting = False
        self._snapshots: dict[asyncio.Task[CameraSnapshot], bool] = {}
        self._close_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> MCPHost:
        """Start explicitly owned capture and then enable dispatch.

        Raises:
            RuntimeError: If this single-use host was already entered or closed.
        """
        if self._entered or self._close_task is not None:
            raise RuntimeError("MCP host can only be entered once")

        self._entered = True

        if self._owned_camera is not None:
            try:
                self._owned_camera.start()

            except BaseException:
                await self.aclose()
                raise

        self._accepting = True
        return self

    async def __aexit__(self, *_: object) -> None:
        """Drain the host before returning control to its application owner."""
        await self.aclose()

    def status(self, arguments: object = EMPTY_ARGUMENTS) -> CameraStatus:
        """Return cached diagnostics, including startup and recovery state.

        Args:
            arguments: Decoded tool arguments from an authorized caller.

        Raises:
            MCPError: If dispatch is closed, validation fails or diagnostics fail.
        """
        self._require_running()
        return self._read_adapter.status(arguments)

    def capabilities(self, arguments: object = EMPTY_ARGUMENTS) -> CameraCapabilities:
        """Read approved cached capabilities for the shared source.

        Args:
            arguments: Decoded tool arguments from an authorized caller.

        Raises:
            MCPError: If dispatch is closed, validation fails or the provider fails.
        """
        self._require_running()
        return self._read_adapter.capabilities(
            arguments, snapshot_adapter=self._snapshot_adapter
        )

    @property
    def snapshot_limits(self) -> SnapshotLimits:
        """Return the effective limits without sampling the preview source."""
        return self._snapshot_adapter.limits

    def inference_status(self, arguments: object = EMPTY_ARGUMENTS) -> InferenceStatus:
        """Read cached diagnostics for the bound inference source.

        Raises:
            MCPError: If dispatch is closed, arguments are invalid, or no
                inference source is configured.
        """
        self._require_running()
        validate_inference_status_input(arguments)

        if self._inference_adapter is None:
            raise MCPError("inference_not_configured")

        return self._inference_adapter.status()

    def inference_latest_result(self) -> InferenceResult:
        """Read one result for internal mapping and freshness checks.

        Raises:
            MCPError: If dispatch is closed or the source or result is missing.
        """
        self._require_running()

        if self._inference_adapter is None:
            raise MCPError("inference_not_configured")

        return self._inference_adapter.latest_result()

    async def inference_latest(
        self, arguments: object = EMPTY_ARGUMENTS
    ) -> InferenceLatest:
        """Map one fresh cached sample and check its age again after mapping.

        Raises:
            MCPError: If dispatch is closed, input is invalid, the binding
                or sample is missing, freshness fails, or the mapper fails.
            asyncio.CancelledError: If the caller cancels the wait.
        """
        self._require_running()
        max_age_ms = parse_inference_latest_input(arguments)
        adapter = self._inference_adapter

        if adapter is None or self._inference_serializer is None:
            raise MCPError("inference_not_configured")

        result = adapter.latest_result()
        require_fresh_inference(result, max_age_ms)
        payload = await self.inference_payload(result)
        self._require_running()
        age_ms = require_fresh_inference(result, max_age_ms)
        sequence = adapter.frame_sequence(result)
        return {
            **payload,
            "schema_version": 1,
            "status": "ready",
            "frame_sequence": sequence,
            "sequence_domain": "raw_capture" if sequence is not None else None,
            "snapshot_correlation": "unknown",
            "age_ms": age_ms,
            "age_clock": "server_monotonic",
            "inference_time_ms": inference_duration_ms(result),
        }

    async def inference_payload(self, result: InferenceResult) -> InferencePayload:
        """Map the caller's already sampled result without reading the slot again.

        The caller applies freshness policy before passing the result here.
        Admission stays occupied until the callback and JSON checks finish.

        Raises:
            MCPError: If dispatch is closed, no mapper is configured, or the
                mapping is busy, exceeds its deadline or fails validation.
            asyncio.CancelledError: If the caller cancels the wait.
        """
        self._require_running()
        serializer = self._inference_serializer
        executor = self._mapping_executor

        if serializer is None or executor is None:
            raise MCPError("inference_not_configured")

        if self._mapping is not None:
            raise MCPError("rate_limited")

        deadline_ns = time.monotonic_ns() + _MAPPING_TIMEOUT_NS
        future = asyncio.get_running_loop().run_in_executor(
            executor, serializer.serialize, result
        )
        self._mapping = future
        future.add_done_callback(self._forget_mapping)

        remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
        done, _ = await asyncio.wait((future,), timeout=remaining)

        if not done or time.monotonic_ns() >= deadline_ns:
            raise MCPError("request_timeout")

        return future.result()

    async def snapshot(
        self,
        arguments: object = EMPTY_ARGUMENTS,
        *,
        deadline_ns: int | None = None,
    ) -> CameraSnapshot:
        """Track a cooperative snapshot read until its task actually exits.

        Args:
            arguments: Tool arguments from an authorized caller.
            deadline_ns: Absolute deadline on the server's monotonic clock.

        Raises:
            MCPError: If dispatch is closed or the snapshot fails.
            asyncio.CancelledError: If the client or host cancels the read.
        """
        self._require_running()
        parse_snapshot_input(arguments, self.snapshot_limits)
        timeout = None
        if deadline_ns is not None:
            if type(deadline_ns) is not int or deadline_ns < 0:
                raise MCPError("internal_error")
            timeout = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
            if timeout <= 0:
                raise MCPError("request_timeout")
        if len(self._snapshots) >= _MAX_ACTIVE_SNAPSHOTS:
            raise MCPError("rate_limited")

        task = asyncio.create_task(
            self._snapshot_adapter.snapshot(arguments, deadline_ns=deadline_ns)
        )
        self._snapshots[task] = False
        task.add_done_callback(self._forget_snapshot)

        try:
            if deadline_ns is not None:
                done, _ = await asyncio.wait((task,), timeout=timeout)
                if not done or time.monotonic_ns() >= deadline_ns:
                    self._cancel_snapshot(task)
                    raise MCPError("request_timeout")
                return task.result()

            return await asyncio.shield(task)

        except asyncio.CancelledError:
            self._cancel_snapshot(task)
            raise

    async def aclose(self) -> None:
        """Close admission, cancel snapshot waits and drain active mapping.

        Propagate caller cancellation after cleanup finishes so an enclosing
        application lifespan can safely release the borrowed sources.
        """
        if self._close_task is None:
            self._accepting = False
            snapshots = tuple(self._snapshots)

            for task in snapshots:
                self._cancel_snapshot(task)

            self._close_task = asyncio.create_task(self._finish_close(snapshots))

        close_task = self._close_task
        cancellation: asyncio.CancelledError | None = None
        while not close_task.done():
            try:
                await asyncio.shield(close_task)

            except asyncio.CancelledError as error:
                cancellation = error

        close_task.result()

        if cancellation is not None:
            raise cancellation

    async def _finish_close(
        self, snapshots: tuple[asyncio.Task[CameraSnapshot], ...]
    ) -> None:
        """Wait for the host's reads before releasing standalone capture."""
        if snapshots:
            await asyncio.gather(*snapshots, return_exceptions=True)

        if self._mapping is not None:
            await asyncio.gather(self._mapping, return_exceptions=True)
        if self._mapping_executor is not None:
            self._mapping_executor.shutdown()

        if self._entered and self._owned_camera is not None:
            self._owned_camera.stop()

    def _forget_mapping(self, future: asyncio.Future[InferencePayload]) -> None:
        """Release admission and observe failures even after caller disconnect."""
        self._mapping = None

        if not future.cancelled():
            future.exception()

    def _cancel_snapshot(self, task: asyncio.Task[CameraSnapshot]) -> None:
        """Request cancellation once so a second disconnect cannot interrupt cleanup."""
        if not task.done() and not self._snapshots[task]:
            self._snapshots[task] = True
            task.cancel()

    def _forget_snapshot(self, task: asyncio.Task[CameraSnapshot]) -> None:
        """Release completed work and observe errors after caller disconnect."""
        del self._snapshots[task]

        if not task.cancelled():
            task.exception()

    def _require_running(self) -> None:
        """Reject calls outside the host lifespan before touching adapters."""
        if not self._accepting:
            raise MCPError("camera_unavailable")
