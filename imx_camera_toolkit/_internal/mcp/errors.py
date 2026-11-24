"""Versioned domain failures and exception mapping independent of the MCP SDK."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from sys import float_info
from typing import Literal, TypedDict, overload

from ..camera.errors import CameraError, CameraTimeoutError
from .contracts import HARD_SNAPSHOT_LIMITS, MAX_SAFE_INTEGER

SimpleErrorCode = Literal[
    "camera_unavailable",
    "camera_starting",
    "invalid_argument",
    "permission_denied",
    "payload_too_large",
    "request_timeout",
    "internal_error",
    "inference_not_configured",
    "no_inference_result",
]
StaleErrorCode = Literal["snapshot_stale", "stale_inference_result"]
ErrorCode = (
    SimpleErrorCode | StaleErrorCode | Literal["snapshot_unavailable", "rate_limited"]
)
SnapshotUnavailableReason = Literal[
    "no_source", "constraints_unsupported", "timing_unavailable", "no_frame"
]
_SNAPSHOT_REASONS = (
    "no_source",
    "constraints_unsupported",
    "timing_unavailable",
    "no_frame",
)

_ERRORS: dict[ErrorCode, tuple[str, bool]] = {
    "camera_unavailable": ("Camera unavailable.", False),
    "camera_starting": ("Camera is starting or recovering.", True),
    "snapshot_unavailable": ("The JPEG source cannot satisfy this request.", False),
    "snapshot_stale": ("The latest JPEG exceeds the maximum age.", True),
    "invalid_argument": ("Invalid tool arguments.", False),
    "permission_denied": ("Permission denied.", False),
    "rate_limited": ("Tool request limit exceeded.", True),
    "payload_too_large": ("The response exceeds the size limit.", False),
    "request_timeout": ("Request timed out.", True),
    "internal_error": ("The tool could not complete the request.", False),
    "inference_not_configured": ("Inference is not configured.", False),
    "no_inference_result": ("No inference result is available.", True),
    "stale_inference_result": (
        "The latest inference result exceeds the maximum age.",
        True,
    ),
}


class SnapshotUnavailableDetails(TypedDict):
    """Allowlisted reason for a missing snapshot."""

    reason: SnapshotUnavailableReason


class StaleDetails(TypedDict):
    """Observed and requested ages in milliseconds."""

    age_ms: float
    max_age_ms: int


class RateLimitDetails(TypedDict):
    """Suggested delay until the next attempt, in milliseconds."""

    retry_after_ms: int


class _OptionalErrorFields(TypedDict, total=False):
    details: SnapshotUnavailableDetails | StaleDetails | RateLimitDetails


class ErrorPayload(_OptionalErrorFields):
    """Serializable error fields safe to return to a tool caller."""

    schema_version: Literal[1]
    code: ErrorCode
    message: str
    retryable: bool


class MCPError(RuntimeError):
    """Tool failure with a fixed public message and an optional internal cause."""

    _payload: ErrorPayload

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
        self, code: StaleErrorCode, /, *, age_ms: float, max_age_ms: int
    ) -> None: ...

    @overload
    def __init__(
        self, code: Literal["rate_limited"], /, *, retry_after_ms: int | None = None
    ) -> None: ...

    def __init__(
        self,
        code: ErrorCode,
        /,
        *,
        reason: SnapshotUnavailableReason | None = None,
        age_ms: float | None = None,
        max_age_ms: int | None = None,
        retry_after_ms: int | None = None,
    ) -> None:
        """Select the public message, retry policy, and allowlisted details.

        Raises:
            ValueError: If the code or its public detail values are invalid.
        """
        if type(code) is not str or code not in _ERRORS:
            raise ValueError("invalid MCP error code")

        message, retryable = _ERRORS[code]
        payload: ErrorPayload = {
            "schema_version": 1,
            "code": code,
            "message": message,
            "retryable": retryable,
        }

        if code == "snapshot_unavailable":
            if type(reason) is not str or reason not in _SNAPSHOT_REASONS:
                raise ValueError("invalid MCP snapshot reason")

            payload["details"] = {"reason": reason}
            payload["retryable"] = reason == "no_frame"

        elif code in ("snapshot_stale", "stale_inference_result"):
            if (
                type(age_ms) not in (int, float)
                or age_ms is None
                or not 0 <= age_ms <= float_info.max
                or type(max_age_ms) is not int
                or not 1 <= max_age_ms <= HARD_SNAPSHOT_LIMITS.max_age_ms
            ):
                raise ValueError("invalid MCP age details")

            payload["details"] = {"age_ms": age_ms, "max_age_ms": max_age_ms}

        elif code == "rate_limited" and retry_after_ms is not None:
            if (
                type(retry_after_ms) is not int
                or not 0 <= retry_after_ms <= MAX_SAFE_INTEGER
            ):
                raise ValueError("invalid MCP retry delay")

            payload["details"] = {"retry_after_ms": retry_after_ms}

        super().__init__(message)
        self._payload = payload

    @property
    def code(self) -> ErrorCode:
        """Return the stable domain code."""
        return self._payload["code"]

    def to_payload(self) -> ErrorPayload:
        """Return a copy of the approved fields without internal diagnostics."""
        payload = self._payload.copy()
        if "details" in payload:
            payload["details"] = payload["details"].copy()
        return payload


@contextmanager
def map_tool_errors() -> Iterator[None]:
    """Preserve domain failures and cancellation, mapping unexpected exceptions."""
    try:
        yield

    except MCPError:
        raise
    except CameraTimeoutError as error:
        raise MCPError("request_timeout") from error

    except CameraError as error:
        raise MCPError("camera_unavailable") from error

    except Exception as error:
        raise MCPError("internal_error") from error
