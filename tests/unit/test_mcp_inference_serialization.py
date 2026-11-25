"""Verify the inference payload projection and its host admission limits."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import cast

import pytest

from imx_camera_toolkit._internal.mcp import host as host_module
from imx_camera_toolkit._internal.mcp.camera_adapter import CameraReadAdapter
from imx_camera_toolkit._internal.mcp.errors import MCPError
from imx_camera_toolkit._internal.mcp.host import MCPHost
from imx_camera_toolkit._internal.mcp.inference_adapter import InferenceReadAdapter
from imx_camera_toolkit._internal.mcp.inference_serialization import (
    MAX_COLLECTION_ITEMS,
    MAX_JSON_DEPTH,
    MAX_PAYLOAD_BYTES,
    InferencePayload,
    InferencePayloadSerializer,
    JSONValue,
)
from imx_camera_toolkit._internal.mcp.snapshot_adapter import CameraSnapshotAdapter
from imx_camera_toolkit.inference import InferenceResult, TensorOutput

_TENSOR = TensorOutput("private", (1,), "float32", object())
_RESULT = InferenceResult(
    1, 10, 20, (_TENSOR,), metadata={"private": "secret"}, overlays=(object(),)
)


def test_serializes_only_application_fields_and_detaches_containers() -> None:
    """Allow independent app schemas and copy even nested shared containers."""
    readings: list[JSONValue] = [True, None, -2, 0.5, "µ"]
    app_payload: dict[str, JSONValue] = {"a": readings, "b": readings}
    mapper = InferencePayloadSerializer(
        "primary", "model", "example.readings.v1", lambda _: app_payload
    )
    data = mapper.serialize(_RESULT)
    assert data == {
        "source_id": "primary",
        "model_id": "model",
        "payload_schema": "example.readings.v1",
        "payload": {"a": readings, "b": readings},
    }
    readings.clear()
    app_payload.clear()
    assert data["payload"] == {
        "a": [True, None, -2, 0.5, "µ"],
        "b": [True, None, -2, 0.5, "µ"],
    }
    summary = InferencePayloadSerializer(
        "secondary", "other-model", "example.summary.v2", lambda _: {"is_ready": True}
    ).serialize(_RESULT)
    assert summary["payload"] == {"is_ready": True}


@pytest.mark.parametrize(
    ("source", "model", "schema"),
    [
        ("", "model", "app.v1"),
        ("primary", "model name", "app.v1"),
        ("primary", "model", ""),
        ("primary", "model", "x" * 257),
        ("x" * 129, "model", "app.v1"),
    ],
)
def test_rejects_incomplete_descriptor(source: str, model: str, schema: str) -> None:
    """Validate explicit identifiers when configuring the application."""
    with pytest.raises(ValueError, match="invalid inference payload descriptor"):
        InferencePayloadSerializer(source, model, schema, lambda _: {})


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {1: 2},
        *(
            {"field": value}
            for value in (
                b"private",
                (1, 2),
                object(),
                _TENSOR,
                _RESULT.metadata,
                float("nan"),
                float("inf"),
                -(2**53),
                "\ud800",
            )
        ),
    ],
)
def test_rejects_unapproved_values(invalid: object) -> None:
    """Reject non-JSON types, invalid Unicode and unsafe numbers."""
    serializer = InferencePayloadSerializer(
        "primary", "model", "app.v1", lambda _: cast(dict[str, JSONValue], invalid)
    )
    with pytest.raises(MCPError) as error:
        serializer.serialize(_RESULT)
    assert error.value.code == "internal_error"


def test_cycles_and_mapper_exceptions_are_sanitized() -> None:
    """Prevent cyclic traversal and keep mapper errors out of diagnostics."""
    cyclic: dict[str, JSONValue] = {}
    cyclic["loop"] = [cyclic]

    def failing_mapper(_: InferenceResult) -> dict[str, JSONValue]:
        """Fail with private application details."""
        raise MCPError("camera_unavailable") from RuntimeError("secret")

    for mapper in (lambda result: cyclic, failing_mapper):
        serializer = InferencePayloadSerializer("primary", "model", "app.v1", mapper)
        with pytest.raises(MCPError) as error:
            serializer.serialize(_RESULT)
        assert error.value.code == "internal_error"
        assert error.value.__cause__ is None
        assert error.value.__context__ is None
        assert "secret" not in str(error.value.to_payload())


def test_enforces_exact_byte_limit_and_structural_budgets() -> None:
    """Measure escaped JSON at the boundary and reject expensive shapes early."""
    deep: JSONValue = None
    for _ in range(MAX_JSON_DEPTH):
        deep = [deep]

    payload: dict[str, JSONValue] = {"x": "a" * (MAX_PAYLOAD_BYTES - len('{"x":""}'))}
    serializer = InferencePayloadSerializer(
        "primary", "model", "app.v1", lambda _: payload
    )
    encoded = json.dumps(
        serializer.serialize(_RESULT)["payload"], separators=(",", ":")
    )
    assert len(encoded.encode("utf-8")) == MAX_PAYLOAD_BYTES

    too_large: list[JSONValue] = [
        cast(str, payload["x"]) + "a",
        "é" * 3000,
        "\\" * MAX_PAYLOAD_BYTES,
        {str(index): index for index in range(MAX_COLLECTION_ITEMS + 1)},
        [True] * (MAX_COLLECTION_ITEMS + 1),
        [[None] * MAX_COLLECTION_ITEMS for _ in range(8)],
        deep,
    ]
    for value in too_large:
        payload["x"] = value
        with pytest.raises(MCPError) as error:
            serializer.serialize(_RESULT)
        assert error.value.code == "payload_too_large"


@dataclass
class _Source:
    """Record reads of a replaceable latest slot."""

    result: InferenceResult = _RESULT
    reads: int = 0

    @property
    def latest_result(self) -> InferenceResult:
        """Return the cached result and record the sample."""
        self.reads += 1
        return self.result


def _host(
    source: _Source, serializer: InferencePayloadSerializer | None = None
) -> MCPHost:
    """Bind a source without owning capture or a worker."""
    return MCPHost(
        CameraReadAdapter(None),
        CameraSnapshotAdapter(None),
        inference_adapter=InferenceReadAdapter(source),
        inference_serializer=serializer,
    )


def test_source_without_mapper_still_reports_status() -> None:
    """Keep mapped payloads unavailable until the app opts in."""

    async def scenario() -> None:
        async with _host(_Source()) as host:
            assert host.inference_status().state == "unknown"
            with pytest.raises(MCPError) as error:
                await host.inference_payload(host.inference_latest_result())
            assert error.value.code == "inference_not_configured"

    asyncio.run(scenario())


@pytest.mark.parametrize("disconnect", [False, True])
def test_mapping_retains_capacity_until_callback_exits(
    disconnect: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Time out or cancel a wait without cancelling its actual mapping work."""
    monkeypatch.setattr(
        host_module, "_MAPPING_TIMEOUT_NS", 10_000_000_000 if disconnect else 25_000_000
    )

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        entered = asyncio.Event()
        release = threading.Event()
        mapped_sequences: list[int] = []

        def map_result(result: InferenceResult) -> dict[str, JSONValue]:
            """Hold the first read while the host receives other requests."""
            assert threading.get_ident() != loop_thread
            mapped_sequences.append(result.frame_sequence)
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(5):
                raise RuntimeError("test did not release mapping")
            return {"sequence": result.frame_sequence}

        source = _Source()
        serializer = InferencePayloadSerializer(
            "primary", "model", "app.v1", map_result
        )
        async with _host(source, serializer) as host:
            result = host.inference_latest_result()
            pending = asyncio.create_task(host.inference_payload(result))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                source.result = InferenceResult(2, 20, 30, ())
                assert host.inference_status().state == "unknown"
                if disconnect:
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                else:
                    with pytest.raises(MCPError) as timeout:
                        await asyncio.wait_for(pending, 5)
                    assert timeout.value.code == "request_timeout"

                with pytest.raises(MCPError) as busy:
                    await host.inference_payload(source.result)
                assert busy.value.code == "rate_limited"
                assert mapped_sequences == [1]
                assert source.reads == 2

                if disconnect:
                    close = asyncio.create_task(host.aclose())
                    await asyncio.sleep(0)
                    with pytest.raises(MCPError) as closing:
                        host.inference_status()
                    assert closing.value.code == "camera_unavailable"
                    assert not close.done()
                    release.set()
                    await asyncio.wait_for(close, 5)
                    return

                monkeypatch.setattr(host_module, "_MAPPING_TIMEOUT_NS", 2_000_000_000)
                release.set()

                async def retry() -> InferencePayload:
                    """Wait for actual worker completion and admission recovery."""
                    next_result = host.inference_latest_result()
                    while True:
                        try:
                            return await host.inference_payload(next_result)
                        except MCPError as error:
                            assert error.code == "rate_limited"
                            await asyncio.sleep(0.001)

                data = await asyncio.wait_for(retry(), 5)
                assert data["payload"] == {"sequence": 2}
                assert source.reads == 3
                assert mapped_sequences == [1, 2]
            finally:
                release.set()
                await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(scenario())
