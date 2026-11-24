"""Sanitized domain failures used by the MCP camera reads."""

from __future__ import annotations

from typing import Literal, TypedDict, overload

SimpleErrorCode = Literal[
    "camera_unavailable",
    "camera_starting",
    "invalid_argument",
    "payload_too_large",
    "request_timeout",
    "internal_error",
]
ErrorCode = SimpleErrorCode | Literal["snapshot_unavailable", "snapshot_stale"]
SnapshotUnavailableReason = Literal[
    "no_source", "constraints_unsupported", "timing_unavailable", "no_frame"
]

_ERRORS: dict[ErrorCode, tuple[str, bool]] = {
    "camera_unavailable": ("Camera unavailable.", False),
    "camera_starting": ("Camera is starting or recovering.", True),
    "snapshot_unavailable": ("The JPEG source cannot satisfy this request.", False),
    "snapshot_stale": ("The latest JPEG exceeds the maximum age.", True),
    "invalid_argument": ("Invalid tool arguments.", False),
    "payload_too_large": ("The response exceeds the size limit.", False),
    "request_timeout": ("Request timed out.", True),
    "internal_error": ("The tool could not complete the request.", False),
}


class SnapshotUnavailableDetails(TypedDict):
    """Allowlisted reason for a missing snapshot."""

    reason: SnapshotUnavailableReason


class SnapshotStaleDetails(TypedDict):
    """Observed and requested ages in milliseconds."""

    age_ms: float
    max_age_ms: int


class _OptionalErrorFields(TypedDict, total=False):
    details: SnapshotUnavailableDetails | SnapshotStaleDetails


class ErrorPayload(_OptionalErrorFields):
    """Serializable error fields safe to return to a tool caller."""

    schema_version: Literal[1]
    code: ErrorCode
    message: str
    retryable: bool


class MCPError(RuntimeError):
    """Tool failure with a fixed public message and an optional internal cause."""

    @overload
    def __init__(self, code: SimpleErrorCode, /) -> None: ...

    @overload
    def __init__(
        self,
        code: Literal["snapshot_unavailable"],
        /,
        *,
        reason: SnapshotUnavailableReason,
    ) -> None: ...

    @overload
    def __init__(
        self, code: Literal["snapshot_stale"], /, *, age_ms: float, max_age_ms: int
    ) -> None: ...

    def __init__(
        self,
        code: ErrorCode,
        /,
        *,
        reason: SnapshotUnavailableReason | None = None,
        age_ms: float | None = None,
        max_age_ms: int | None = None,
    ) -> None:
        """Select the public message, retry policy, and allowlisted details."""
        self.code = code
        self.message, self.retryable = _ERRORS[code]
        self._details: SnapshotUnavailableDetails | SnapshotStaleDetails | None = None
        if code == "snapshot_unavailable" and reason is not None:
            self._details = {"reason": reason}
            self.retryable = reason == "no_frame"
        if code == "snapshot_stale" and age_ms is not None and max_age_ms is not None:
            self._details = {"age_ms": age_ms, "max_age_ms": max_age_ms}
        super().__init__(self.message)

    def to_payload(self) -> ErrorPayload:
        """Return only the schema fields intended for clients."""
        payload: ErrorPayload = {
            "schema_version": 1,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self._details is not None:
            payload["details"] = self._details.copy()
        return payload
