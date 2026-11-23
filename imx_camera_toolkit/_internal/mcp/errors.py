"""Sanitized domain failures used by the MCP camera reads."""

from __future__ import annotations

from typing import Literal, TypedDict

ErrorCode = Literal["camera_unavailable", "request_timeout", "internal_error"]
_ERRORS: dict[ErrorCode, tuple[str, bool]] = {
    "camera_unavailable": ("Camera unavailable.", False),
    "request_timeout": ("Request timed out.", True),
    "internal_error": ("The tool could not complete the request.", False),
}


class ErrorPayload(TypedDict):
    """Serializable error fields safe to return to a tool caller."""

    schema_version: Literal[1]
    code: ErrorCode
    message: str
    retryable: bool


class MCPError(RuntimeError):
    """Tool failure with a fixed public message and an optional internal cause."""

    def __init__(self, code: ErrorCode) -> None:
        """Select the public message and retry policy for the given code."""
        self.code = code
        self.message, self.retryable = _ERRORS[code]
        super().__init__(self.message)

    def to_payload(self) -> ErrorPayload:
        """Return only the schema fields intended for clients."""
        return {
            "schema_version": 1,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
