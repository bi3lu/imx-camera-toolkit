"""Read a bounded latest JPEG from an application-owned preview source."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from typing import Protocol

from ..camera.publishing.jpeg_slot import JPEGPreview, JPEGPublication
from .contracts import (
    HARD_SNAPSHOT_LIMITS,
    MAX_SAFE_INTEGER,
    CameraSnapshot,
    SnapshotCapabilities,
    SnapshotLimits,
    SnapshotMetadata,
)
from .errors import MCPError, map_tool_errors
from .jpeg_validation import jpeg_dimensions
from .validation import (
    EMPTY_ARGUMENTS,
    SnapshotRequest,
    parse_snapshot_input,
    validate_snapshot_limits,
)

_NS_PER_MS = 1_000_000
_POLL_NS = 10 * _NS_PER_MS
_WIRE_RESERVE_BYTES = 1024


class SnapshotSource(Protocol):
    """Borrowed, nonblocking source of a single detached JPEG publication."""

    def _jpeg_preview(self) -> JPEGPreview:
        """Read fixed source constraints, lifecycle state and the latest record."""
        ...


def _read_publication(
    source: SnapshotSource | None, request: SnapshotRequest
) -> JPEGPublication | None:
    """Apply source lifecycle, fixed constraints, and timing policy."""
    if source is None:
        raise MCPError("camera_unavailable")
    preview = source._jpeg_preview()
    if preview.state in ("starting", "recovering"):
        raise MCPError("camera_starting")
    if preview.state != "running":
        raise MCPError("camera_unavailable")
    if not preview.enabled:
        raise MCPError("snapshot_unavailable", reason="no_source")
    if (
        preview.width > request.max_width
        or preview.height > request.max_height
        or (
            preview.quality is not None
            and not 1 <= preview.quality <= request.max_jpeg_quality
        )
        or (
            request.jpeg_quality is not None and request.jpeg_quality != preview.quality
        )
    ):
        raise MCPError("snapshot_unavailable", reason="constraints_unsupported")
    if not preview.acquisition_timing:
        raise MCPError("snapshot_unavailable", reason="timing_unavailable")

    publication = preview.publication
    if publication is not None and (
        (publication.width, publication.height) != (preview.width, preview.height)
        or publication.quality != preview.quality
    ):
        raise ValueError("JPEG publication does not match the fixed source")
    return publication


def _estimated_wire_bytes(metadata: SnapshotMetadata) -> int:
    """Estimate UTF-8 JSON-RPC size using base64 expansion and a framing reserve."""
    structured = asdict(metadata)
    envelope = {
        "jsonrpc": "2.0",
        "id": MAX_SAFE_INTEGER,
        "result": {
            "content": [
                {"type": "text", "text": json.dumps(structured)},
                {"type": "image", "mimeType": "image/jpeg", "data": ""},
            ],
            "isError": False,
            "structuredContent": structured,
        },
    }
    encoded_bytes = 4 * ((metadata.jpeg_bytes + 2) // 3)
    return (
        len(json.dumps(envelope).encode("utf-8")) + encoded_bytes + _WIRE_RESERVE_BYTES
    )


class CameraSnapshotAdapter:
    """Return a fresh JPEG from one enabled fixed source without encoding frames.

    The application owns the source. Waiting is cooperative and retains no
    older publication across an await. A transport must enforce the final
    serialized response budget and admission limits.
    """

    def __init__(
        self,
        source: SnapshotSource | None,
        *,
        limits: SnapshotLimits = HARD_SNAPSHOT_LIMITS,
    ) -> None:
        """Bind an existing source and validate the server's effective limits.

        Raises:
            ValueError: If a configured limit exceeds the contract bounds.
        """
        validate_snapshot_limits(limits)
        self._source = source
        self.limits = limits

    def capabilities(self) -> SnapshotCapabilities:
        """Describe the bound preview from one sample without waiting or encoding."""
        if self._source is None:
            return SnapshotCapabilities(None, False, self.limits)

        preview = self._source._jpeg_preview()
        supported = None

        if not preview.acquisition_timing:
            supported = False

        elif (
            preview.enabled
            and preview.state == "running"
            and preview.publication is not None
            and preview.publication.acquired_at_ns is not None
        ):
            supported = True

        return SnapshotCapabilities(supported, preview.enabled, self.limits)

    async def snapshot(
        self,
        arguments: object = EMPTY_ARGUMENTS,
        *,
        deadline_ns: int | None = None,
    ) -> CameraSnapshot:
        """Return one snapshot or a typed failure before the effective deadline.

        Args:
            arguments: Tool arguments from an already authorized caller.
            deadline_ns: Optional absolute request deadline on the server's
                monotonic clock. This can shorten a request's positive wait.

        Raises:
            MCPError: If validation, the source, freshness or a budget fails.
            asyncio.CancelledError: If the caller cancels while awaiting data.
        """
        with map_tool_errors():
            request = parse_snapshot_input(arguments, self.limits)

            if deadline_ns is not None and (
                type(deadline_ns) is not int or deadline_ns < 0
            ):
                raise ValueError("invalid monotonic request deadline")

            expires_ns = deadline_ns

            if request.wait_ms:
                wait_until_ns = time.monotonic_ns() + request.wait_ms * _NS_PER_MS
                expires_ns = (
                    wait_until_ns
                    if expires_ns is None
                    else min(wait_until_ns, expires_ns)
                )

            while True:
                self._check_deadline(expires_ns)
                result = self._read_snapshot(request, expires_ns)
                if result is not None:
                    return result

                if expires_ns is None:
                    raise MCPError("internal_error")
                remaining_ns = expires_ns - time.monotonic_ns()
                await asyncio.sleep(max(0, min(_POLL_NS, remaining_ns)) / 1_000_000_000)

    @staticmethod
    def _check_deadline(expires_ns: int | None) -> None:
        """Raise the tool timeout when the effective deadline has elapsed."""
        if expires_ns is not None and time.monotonic_ns() >= expires_ns:
            raise MCPError("request_timeout")

    def _read_snapshot(
        self, request: SnapshotRequest, expires_ns: int | None
    ) -> CameraSnapshot | None:
        """Read and check one publication, returning no result when still waiting."""
        publication = _read_publication(self._source, request)
        self._check_deadline(expires_ns)
        if publication is None:
            if request.wait_ms:
                return None
            raise MCPError("snapshot_unavailable", reason="no_frame")
        acquired_ns = publication.acquired_at_ns
        if acquired_ns is None:
            raise MCPError("snapshot_unavailable", reason="timing_unavailable")
        if acquired_ns < 0:
            raise ValueError("invalid monotonic acquisition timestamp")
        if not self._is_fresh(time.monotonic_ns() - acquired_ns, request):
            return None
        if type(publication.data) is not bytes:
            raise ValueError("JPEG publication is not detached bytes")
        size = len(publication.data)
        if size > self.limits.max_jpeg_bytes:
            raise MCPError("payload_too_large")
        width, height = jpeg_dimensions(publication.data)
        if (width, height) != (publication.width, publication.height):
            raise ValueError("JPEG dimensions do not match the publication")
        if not 0 <= publication.sequence <= MAX_SAFE_INTEGER:
            raise ValueError("JPEG sequence is out of range")
        self._check_deadline(expires_ns)
        age_ns = time.monotonic_ns() - acquired_ns
        if not self._is_fresh(age_ns, request):
            return None
        metadata = SnapshotMetadata(
            width,
            height,
            publication.quality,
            size,
            publication.sequence,
            age_ns / _NS_PER_MS,
        )
        if _estimated_wire_bytes(metadata) > self.limits.max_response_bytes:
            raise MCPError("payload_too_large")
        self._check_deadline(expires_ns)
        return CameraSnapshot(metadata, publication.data)

    @staticmethod
    def _is_fresh(age_ns: int, request: SnapshotRequest) -> bool:
        """Apply an inclusive freshness bound and reject future timestamps."""
        if age_ns < 0:
            raise ValueError("invalid monotonic acquisition timestamp")
        if age_ns <= request.max_age_ms * _NS_PER_MS:
            return True
        if request.wait_ms:
            return False
        raise MCPError(
            "snapshot_stale", age_ms=age_ns / _NS_PER_MS, max_age_ms=request.max_age_ms
        )
