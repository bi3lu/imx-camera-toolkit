"""Verify the stable error catalog and its serialization boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from imx_camera_toolkit._internal.mcp.errors import (
    MCPError,
    SimpleErrorCode,
    SnapshotUnavailableReason,
    map_tool_errors,
)


def test_catalog_matches_contract() -> None:
    """Cover every approved code and verify its retry policy."""
    errors = [
        (MCPError("camera_unavailable"), False),
        (MCPError("camera_starting"), True),
        (MCPError("snapshot_unavailable", reason="no_frame"), True),
        (MCPError("snapshot_unavailable", reason="no_source"), False),
        (MCPError("snapshot_stale", age_ms=10_001, max_age_ms=10_000), True),
        (MCPError("invalid_argument"), False),
        (MCPError("permission_denied"), False),
        (MCPError("rate_limited"), True),
        (MCPError("request_timeout"), True),
        (MCPError("payload_too_large"), False),
        (MCPError("internal_error"), False),
        (MCPError("inference_not_configured"), False),
        (MCPError("no_inference_result"), True),
        (MCPError("stale_inference_result", age_ms=11.2, max_age_ms=10), True),
    ]
    schema_path = (
        Path(__file__).resolve().parents[2] / "docs/mcp/tool-contracts.schema.json"
    )
    code_enum = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"][
        "DomainError"
    ]["properties"]["code"]["enum"]
    assert {error.code for error, _ in errors} == set(code_enum)
    for error, retryable in errors:
        payload = error.to_payload()
        assert payload["retryable"] is retryable
        assert payload["message"] == str(error)
        json.dumps(payload, allow_nan=False)


def test_retry_delay_and_payload_copy() -> None:
    """Include a bounded delay when known and isolate each payload copy."""
    assert "details" not in MCPError("rate_limited").to_payload()
    for delay in (0, 9_007_199_254_740_991):
        error = MCPError("rate_limited", retry_after_ms=delay)
        payload = error.to_payload()
        assert payload["details"] == {"retry_after_ms": delay}
        cast(dict[str, object], payload["details"]).clear()
        payload["message"] = "modified"
        assert error.to_payload()["details"] == {"retry_after_ms": delay}
        assert error.to_payload()["message"] == str(error)


@pytest.mark.parametrize("delay", [-1, True, 1.0, 9_007_199_254_740_992])
def test_rejects_invalid_retry_delay(delay: object) -> None:
    """Require a nonnegative JSON-safe integer delay."""
    with pytest.raises(ValueError, match="invalid MCP retry delay"):
        MCPError("rate_limited", retry_after_ms=cast(int, delay))


@pytest.mark.parametrize(
    ("age", "maximum"),
    [
        (float("inf"), 1),
        (float("nan"), 1),
        (True, 1),
        (-1, 1),
        (100, 0),
        (100, 10_001),
        (100, False),
        (100, 10.0),
    ],
)
def test_rejects_invalid_age_details(age: object, maximum: object) -> None:
    """Enforce finite observed ages and a bounded integer maximum."""
    with pytest.raises(ValueError, match="invalid MCP age details"):
        MCPError(
            "snapshot_stale", age_ms=cast(float, age), max_age_ms=cast(int, maximum)
        )


def test_unapproved_error_data_fails_closed() -> None:
    """Exclude raw diagnostics supplied as a code or a detail value."""
    private_detail = "/private/camera token=secret"
    with pytest.raises(ValueError, match="invalid MCP error code"):
        MCPError(cast(SimpleErrorCode, private_detail))
    with pytest.raises(MCPError) as caught, map_tool_errors():
        raise MCPError(
            "snapshot_unavailable",
            reason=cast(SnapshotUnavailableReason, private_detail),
        )
    assert caught.value.code == "internal_error"
    assert isinstance(caught.value.__cause__, ValueError)
    assert private_detail not in json.dumps(caught.value.to_payload())
