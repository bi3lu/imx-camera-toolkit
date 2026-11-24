"""Read a bounded latest JPEG from an application-owned preview source."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

from ..camera.publishing.jpeg_slot import JPEGPreview, JPEGPublication
from .contracts import CameraSnapshot, SnapshotLimits, SnapshotMetadata
from .errors import MCPError
from .jpeg_validation import jpeg_dimensions

HARD_SNAPSHOT_LIMITS = SnapshotLimits(1920, 1080, 95, 10_000, 1_000, 524_288, 786_432)
_NS_PER_MS = 1_000_000
_POLL_NS = 10 * _NS_PER_MS
_MAX_SAFE_INTEGER = (1 << 53) - 1
_WIRE_RESERVE_BYTES = 1024


class SnapshotSource(Protocol):
    """Borrowed, nonblocking source of a single detached JPEG publication."""

    def _jpeg_preview(self) -> JPEGPreview:
        """Read fixed source constraints, lifecycle state and the latest record."""
        ...


@dataclass(frozen=True, slots=True)
class _SnapshotRequest:
    max_width: int
    max_height: int
    jpeg_quality: int | None
    max_jpeg_quality: int
    max_age_ms: int
    wait_ms: int


def _integer(value: object, minimum: int, maximum: int) -> int:
    """Validate an integral field without accepting booleans or coercion."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise MCPError("invalid_argument")
    return value


def _parse_request(
    arguments: Mapping[str, object] | None, limits: SnapshotLimits
) -> _SnapshotRequest:
    """Validate the entire request before accessing a source."""
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping) or arguments.keys() - {
        "camera_id",
        "max_width",
        "max_height",
        "jpeg_quality",
        "max_age_ms",
        "wait_ms",
    }:
        raise MCPError("invalid_argument")
    if arguments.get("camera_id", "primary") != "primary":
        raise MCPError("invalid_argument")

    quality = (
        _integer(arguments["jpeg_quality"], 1, limits.max_jpeg_quality)
        if "jpeg_quality" in arguments
        else None
    )
    return _SnapshotRequest(
        max_width=_integer(
            arguments.get("max_width", limits.max_width), 1, limits.max_width
        ),
        max_height=_integer(
            arguments.get("max_height", limits.max_height), 1, limits.max_height
        ),
        jpeg_quality=quality,
        max_jpeg_quality=limits.max_jpeg_quality,
        max_age_ms=_integer(
            arguments.get("max_age_ms", min(2_000, limits.max_age_ms)),
            1,
            limits.max_age_ms,
        ),
        wait_ms=_integer(arguments.get("wait_ms", 0), 0, limits.max_wait_ms),
    )


def _read_publication(
    source: SnapshotSource | None, request: _SnapshotRequest
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
        "id": _MAX_SAFE_INTEGER,
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
        try:
            for name, maximum in asdict(HARD_SNAPSHOT_LIMITS).items():
                _integer(getattr(limits, name), int(name != "max_wait_ms"), maximum)
        except MCPError as error:
            raise ValueError("invalid server snapshot limits") from error
        self._source = source
        self.limits = limits

    async def snapshot(
        self,
        arguments: Mapping[str, object] | None = None,
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
        request = _parse_request(arguments, self.limits)
        expires_ns = deadline_ns
        if request.wait_ms:
            wait_until_ns = time.monotonic_ns() + request.wait_ms * _NS_PER_MS
            expires_ns = (
                wait_until_ns if expires_ns is None else min(wait_until_ns, expires_ns)
            )

        try:
            while True:
                self._check_deadline(expires_ns)
                result = self._read_snapshot(request, expires_ns)
                if result is not None:
                    return result

                if expires_ns is None:
                    raise MCPError("internal_error")
                remaining_ns = expires_ns - time.monotonic_ns()
                await asyncio.sleep(max(0, min(_POLL_NS, remaining_ns)) / 1_000_000_000)
        except MCPError:
            raise
        except Exception as error:
            raise MCPError("internal_error") from error

    @staticmethod
    def _check_deadline(expires_ns: int | None) -> None:
        """Raise the tool timeout when the effective deadline has elapsed."""
        if expires_ns is not None and time.monotonic_ns() >= expires_ns:
            raise MCPError("request_timeout")

    def _read_snapshot(
        self, request: _SnapshotRequest, expires_ns: int | None
    ) -> CameraSnapshot | None:
        """Read and check one publication, returning no result when still waiting."""
        publication = _read_publication(self._source, request)
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
        if not 0 <= publication.sequence <= _MAX_SAFE_INTEGER:
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
    def _is_fresh(age_ns: int, request: _SnapshotRequest) -> bool:
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
