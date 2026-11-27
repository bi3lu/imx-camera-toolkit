"""Verify freshness and provenance for a single cached inference sample."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from imx_camera_toolkit._internal.mcp import inference_adapter
from imx_camera_toolkit._internal.mcp.camera_adapter import CameraReadAdapter
from imx_camera_toolkit._internal.mcp.errors import MCPError
from imx_camera_toolkit._internal.mcp.host import MCPHost
from imx_camera_toolkit._internal.mcp.inference_adapter import InferenceReadAdapter
from imx_camera_toolkit._internal.mcp.inference_serialization import (
    InferenceMapper,
    InferencePayloadSerializer,
    JSONValue,
)
from imx_camera_toolkit._internal.mcp.snapshot_adapter import CameraSnapshotAdapter
from imx_camera_toolkit.inference import InferenceResult

_NOW = 20_000_000_000
_RESULT = InferenceResult(3, _NOW - 2_000_000_000, 1_500_000, ())


@dataclass
class _Source:
    """Record samples from a replaceable result slot."""

    result: InferenceResult | None = _RESULT
    reads: int = 0

    @property
    def latest_result(self) -> InferenceResult | None:
        """Return the cached slot and count reads."""
        self.reads += 1
        return self.result


def _host(
    source: _Source, mapper: InferenceMapper | None, *, raw_sequence: bool = False
) -> MCPHost:
    """Bind a cached source and optional mapper without taking ownership."""
    serializer = (
        None
        if mapper is None
        else InferencePayloadSerializer("capture", "model", "app.v1", mapper)
    )
    return MCPHost(
        CameraReadAdapter(None),
        CameraSnapshotAdapter(None),
        inference_adapter=InferenceReadAdapter(
            source, raw_capture_sequence=raw_sequence
        ),
        inference_serializer=serializer,
    )


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        {"max_age_ms": True},
        {"max_age_ms": 1.0},
        {"max_age_ms": "1"},
        {"max_age_ms": None},
        {"max_age_ms": 0},
        {"max_age_ms": 10_001},
        {"camera_id": "primary"},
    ],
)
def test_validates_before_reading_the_slot(invalid: object) -> None:
    """Reject malformed arguments before reading the source."""
    source = _Source()

    async def scenario() -> None:
        async with _host(source, lambda _: {}) as host:
            with pytest.raises(MCPError) as error:
                await host.inference_latest(invalid)

            assert error.value.code == "invalid_argument"

    asyncio.run(scenario())
    assert source.reads == 0


def test_requires_mapper_and_cached_result() -> None:
    """Distinguish an incomplete binding from an empty slot."""
    source = _Source(None)
    mapped: list[InferenceResult] = []

    def mapper(result: InferenceResult) -> dict[str, JSONValue]:
        """Record mapper admission."""
        mapped.append(result)
        return {}

    async def scenario() -> None:
        async with _host(source, None) as host:
            with pytest.raises(MCPError) as unconfigured:
                await host.inference_latest()

            assert unconfigured.value.code == "inference_not_configured"
            assert source.reads == 0

        async with _host(source, mapper) as host:
            with pytest.raises(MCPError) as missing:
                await host.inference_latest()

            assert missing.value.code == "no_inference_result"

    asyncio.run(scenario())
    assert source.reads == 1
    assert not mapped


@pytest.mark.parametrize(
    ("timestamp", "expected_age"),
    [(_RESULT.frame_timestamp_ns - 1, 2000.000001), (0, None), (_NOW + 1, None)],
)
def test_rejects_expired_or_unknown_age_before_mapping(
    monkeypatch: pytest.MonkeyPatch, timestamp: int, expected_age: float | None
) -> None:
    """Reject one nanosecond beyond the limit and ignore hardware timing."""
    monkeypatch.setattr(
        inference_adapter, "time", SimpleNamespace(monotonic_ns=lambda: _NOW)
    )
    source = _Source(
        replace(_RESULT, frame_timestamp_ns=timestamp, capture_timestamp_ns=_NOW - 1)
    )
    mapped: list[InferenceResult] = []

    def mapper(result: InferenceResult) -> dict[str, JSONValue]:
        """Record any unexpected mapping."""
        mapped.append(result)
        return {}

    async def scenario() -> None:
        async with _host(source, mapper) as host:
            with pytest.raises(MCPError) as stale:
                await host.inference_latest()

            assert stale.value.code == "stale_inference_result"
            assert stale.value.to_payload()["details"] == {
                "age_ms": expected_age,
                "max_age_ms": 2000,
            }

    asyncio.run(scenario())
    assert source.reads == 1
    assert not mapped


@pytest.mark.parametrize("raw_sequence", [False, True])
def test_rechecks_same_sample_after_mapping(
    monkeypatch: pytest.MonkeyPatch, raw_sequence: bool
) -> None:
    """Accept the inclusive boundary and keep a reused sequence uncorrelated."""
    now = _NOW
    monkeypatch.setattr(
        inference_adapter, "time", SimpleNamespace(monotonic_ns=lambda: now)
    )
    source = _Source()
    replacement = replace(_RESULT, frame_timestamp_ns=_NOW, inference_time_ns=20)
    mapped: list[InferenceResult] = []

    def mapper(result: InferenceResult) -> dict[str, JSONValue]:
        """Reuse the sequence on a newer capture while mapping the old result."""
        nonlocal now
        mapped.append(result)
        source.result = replacement

        if len(mapped) == 2:
            now += 1

        return {"sample": len(mapped)}

    async def scenario() -> None:
        async with _host(source, mapper, raw_sequence=raw_sequence) as host:
            ready = await host.inference_latest()
            assert ready["status"] == "ready"
            assert ready["age_ms"] == 2000
            assert ready["inference_time_ms"] == 1.5
            assert ready["payload"] == {"sample": 1}
            assert ready["frame_sequence"] == (3 if raw_sequence else None)
            assert ready["sequence_domain"] == ("raw_capture" if raw_sequence else None)
            assert ready["snapshot_correlation"] == "unknown"
            assert source.result is replacement

            source.result = _RESULT

            with pytest.raises(MCPError) as stale:
                await host.inference_latest()

            assert stale.value.code == "stale_inference_result"
            assert stale.value.to_payload()["details"] == {
                "age_ms": 2000.000001,
                "max_age_ms": 2000,
            }

    asyncio.run(scenario())
    assert source.reads == 2
    assert len(mapped) == 2
    assert all(result is _RESULT for result in mapped)
