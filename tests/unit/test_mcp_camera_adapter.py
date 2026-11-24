"""Host tests for MCP camera reads and publication readiness."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic_ns
from unittest.mock import Mock, call

import pytest

from imx_camera_toolkit import (
    Camera,
    CameraOpenError,
    CameraStats,
    CameraTimeoutError,
    GpuCamera,
    __version__,
)
from imx_camera_toolkit._internal.camera.backends.gstreamer import (
    GStreamerCaptureBackend,
)
from imx_camera_toolkit._internal.camera.profiles.registry import get_camera_profile
from imx_camera_toolkit._internal.camera_control.capabilities import (
    DEFAULT_CAPABILITIES,
)
from imx_camera_toolkit._internal.camera_control.models import (
    CameraCapabilities as ArgusCapabilities,
)
from imx_camera_toolkit._internal.camera_control.models import SensorMode as ArgusMode
from imx_camera_toolkit._internal.camera_control.runtime.controller import (
    CameraController,
)
from imx_camera_toolkit._internal.consumers.latest import LatestFrameHub
from imx_camera_toolkit._internal.mcp.camera_adapter import CameraReadAdapter
from imx_camera_toolkit._internal.mcp.contracts import (
    CameraBackend,
    CameraCapabilities,
    ControlCapabilities,
    ExposureBounds,
    SensorMode,
    SnapshotCapabilities,
    SnapshotLimits,
)
from imx_camera_toolkit._internal.mcp.errors import ErrorCode, MCPError
from imx_camera_toolkit.testing import MockCamera, mock_gpu_frame

_OBSERVED_AT = datetime(2026, 11, 23, 13, tzinfo=timezone(timedelta(hours=1)))
_STATS = CameraStats(
    captured_frames=42,
    dropped_frames=12,
    capture_fps=29.8,
    last_frame_timestamp_ns=123,
    recovery_count=1,
    consecutive_failures=0,
    running=True,
    state="running",
    has_published_frame=True,
    last_failure_reason="/private/camera token=secret",
    consumer_dropped_frames=(("/private/subscriber", 10), ("preview", 99)),
)


@pytest.fixture
def contract_results() -> dict[str, dict[str, object]]:
    """Load the approved example results for exact comparison on the wire."""
    fixtures = (
        Path(__file__).resolve().parents[2] / "docs/mcp/tool-contracts.examples.json"
    )
    return {
        case["name"]: case["expected"]["value"]
        for case in json.loads(fixtures.read_text(encoding="utf-8"))["cases"]
    }


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize(
    ("state", "running", "published", "expected"),
    [
        ("stopped", True, True, "unavailable"),
        ("starting", False, True, "starting"),
        ("running", True, False, "starting"),
        ("running", True, True, "ready"),
        ("running", False, True, "unavailable"),
        ("recovering", True, True, "recovering"),
        ("failed", False, True, "error"),
        ("failed", True, True, "error"),
    ],
)
def test_status_applies_lifecycle_precedence_with_one_read(
    backend: CameraBackend,
    state: str,
    running: bool,
    published: bool,
    expected: str,
) -> None:
    """Retained frames and old failure text must not override the lifecycle."""
    source = Mock(spec=Camera if backend == "cpu" else GpuCamera)
    source.stats.return_value = replace(
        _STATS, state=state, running=running, has_published_frame=published
    )
    provider = Mock(return_value=None)
    adapter = CameraReadAdapter(source, capability_source=provider)

    status = adapter.status()
    assert status.state == expected
    assert status.backend == backend
    assert status.subscriber_dropped_frames == {}
    provider.assert_not_called()

    assert adapter.capabilities() == CameraCapabilities()
    provider.assert_called_once_with()
    assert source.mock_calls == [call.stats()]


def test_status_matches_contract_and_reads_only_approved_subscription(
    contract_results: dict[str, dict[str, object]],
) -> None:
    """Use the instance count even when a camera metric has the same name."""
    source = Mock(spec=GpuCamera)
    source.stats.return_value = _STATS
    hub = LatestFrameHub[int]()
    subscription = hub.subscribe("/private/subscriber")
    subscription.record_drop(3)
    aliases = {"preview": subscription}
    adapter = CameraReadAdapter(source, subscribers=aliases, clock=lambda: _OBSERVED_AT)
    aliases.clear()

    expected = {**contract_results["status_ready"], "toolkit_version": __version__}
    try:
        assert asdict(adapter.status()) == expected
        subscription.close()
        replacement = hub.subscribe(subscription.name)
        assert adapter.status().subscriber_dropped_frames == {"preview": 3}
        rebound = CameraReadAdapter(source, subscribers={"preview": replacement})
        assert rebound.status().subscriber_dropped_frames == {"preview": 0}
    finally:
        hub.close()


def test_unregistered_camera_returns_unknown_data(
    contract_results: dict[str, dict[str, object]],
) -> None:
    """An unregistered camera skips capability and subscriber reads."""
    provider = Mock(side_effect=AssertionError("unexpected capability lookup"))
    subscription = Mock(spec=[])
    adapter = CameraReadAdapter(
        backend="gpu",
        capability_source=provider,
        subscribers={"preview": subscription},
        clock=lambda: _OBSERVED_AT,
    )

    expected = {
        **contract_results["status_unregistered"],
        "toolkit_version": __version__,
    }
    assert asdict(adapter.status()) == expected
    assert (
        asdict(adapter.capabilities()) == contract_results["capabilities_unregistered"]
    )
    provider.assert_not_called()


def test_capabilities_matches_contract_without_accessing_capture(
    contract_results: dict[str, dict[str, object]],
) -> None:
    """Expose only the typed projection from an existing capability source."""
    source = Mock(spec=Camera)
    capabilities = CameraCapabilities(
        sensor_model="IMX219",
        sensor_modes=(SensorMode(index=0, width=1920, height=1080, max_fps=30),),
        frame_format="BGR_CPU",
        controls=ControlCapabilities(exposure_us=ExposureBounds()),
        snapshot=SnapshotCapabilities(
            source_supported=True,
            configured=True,
            limits=SnapshotLimits(1920, 1080, 95, 10_000, 1000, 524_288, 786_432),
        ),
    )
    provider = Mock(return_value=capabilities)
    adapter = CameraReadAdapter(source, capability_source=provider)

    wire_value = json.loads(json.dumps(asdict(adapter.capabilities()), allow_nan=False))
    assert wire_value == contract_results["capabilities_known"]
    provider.assert_called_once_with()
    assert source.mock_calls == []


def test_capabilities_rejects_untyped_provider_data() -> None:
    """An untyped provider must not bypass the approved capability envelope."""
    provider = Mock(return_value={"pipeline": "/private/camera"})
    adapter = CameraReadAdapter(Mock(spec=GpuCamera), capability_source=provider)

    with pytest.raises(MCPError) as raised:
        adapter.capabilities()

    assert raised.value.code == "internal_error"
    assert "/private/camera" not in json.dumps(raised.value.to_payload())


@pytest.mark.parametrize("operation", ["status", "capabilities"])
@pytest.mark.parametrize(
    "arguments",
    [
        None,
        [],
        {"camera_id": "Primary"},
        {"camera_id": True},
        {"camera_id": None},
        {"camera_id": ["primary"]},
        {"max_width": 1},
    ],
)
def test_invalid_read_arguments_do_not_access_sources(
    operation: str, arguments: object
) -> None:
    """Reject undeclared keys, invalid camera IDs and non-object arguments."""
    source = Mock(spec=Camera)
    provider = Mock()
    clock = Mock()
    adapter = CameraReadAdapter(source, capability_source=provider, clock=clock)
    read = adapter.status if operation == "status" else adapter.capabilities

    with pytest.raises(MCPError) as raised:
        read(arguments)

    assert raised.value.code == "invalid_argument"
    assert source.mock_calls == []
    provider.assert_not_called()
    clock.assert_not_called()


@pytest.mark.parametrize("operation", ["status", "capabilities"])
@pytest.mark.parametrize(
    ("exception_type", "code", "retryable"),
    [
        (CameraOpenError, "camera_unavailable", False),
        (CameraTimeoutError, "request_timeout", True),
        (RuntimeError, "internal_error", False),
        (TimeoutError, "internal_error", False),
    ],
)
def test_read_errors_are_typed_and_sanitized(
    operation: str, exception_type: type[Exception], code: ErrorCode, retryable: bool
) -> None:
    """Keep original causes on the host and exclude them from public fields."""
    private_detail = "/private/camera token=secret"
    original = exception_type(private_detail)
    source = Mock(spec=Camera)
    source.stats.side_effect = original
    adapter = CameraReadAdapter(source, capability_source=Mock(side_effect=original))
    read = adapter.status if operation == "status" else adapter.capabilities

    with pytest.raises(MCPError) as raised:
        read()

    error = raised.value
    payload = error.to_payload()
    assert error.__cause__ is original
    assert payload == {
        "schema_version": 1,
        "code": code,
        "message": str(error),
        "retryable": retryable,
    }
    assert private_detail not in json.dumps(payload)


def test_public_counter_range_and_alias_validation() -> None:
    """Reject unsafe aliases and avoid emitting inexact JSON counts."""
    source = Mock(spec=Camera)
    source.stats.return_value = replace(_STATS, dropped_frames=1 << 53)
    hub = LatestFrameHub[int]()
    subscription = hub.subscribe("internal")
    try:
        with pytest.raises(ValueError, match="invalid public subscriber alias"):
            CameraReadAdapter(source, subscribers={"preview\n": subscription})

        adapter = CameraReadAdapter(source, subscribers={"public": subscription})
        subscription.record_drop((1 << 53) - 1)
        status = adapter.status()
        assert status.capture_dropped_frames is None
        assert status.subscriber_dropped_frames == {"public": (1 << 53) - 1}
        subscription.record_drop()
        assert adapter.status().subscriber_dropped_frames == {}
    finally:
        hub.close()


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_builtin_status_requires_publication_in_current_run(
    backend: CameraBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Start with fake backends and check publication across a failed worker restart."""
    camera: Camera | GpuCamera
    if backend == "cpu":
        camera = Camera(enable_preview=False)
        monkeypatch.setattr(GStreamerCaptureBackend, "available", lambda: True)
    else:
        camera = GpuCamera(enable_preview=False)
        monkeypatch.setattr(camera, "_backend_available", lambda: True)
        monkeypatch.setattr(camera, "_prepare_pipeline", lambda: None)
    monkeypatch.setattr(camera, "_create_backend", Mock())
    monkeypatch.setattr(camera, "_capture_loop", lambda: None)
    adapter = CameraReadAdapter(camera)

    def publish() -> None:
        """Simulate source capture and then publish its raw or NVMM frame."""
        if isinstance(camera, Camera):
            timestamp_ns = monotonic_ns()
            camera._record_capture(timestamp_ns)
            assert adapter.status().state == "starting"
            camera._publish_frame(object(), timestamp_ns)
        else:
            frame = camera._assign_sequence(mock_gpu_frame(object()))
            camera._record_capture(frame)
            assert adapter.status().state == "starting"
            camera._gpu_publisher.publish(frame)

    try:
        assert adapter.status().state == "unavailable"
        assert adapter.capabilities().frame_format is None
        camera.start()
        assert adapter.status().state == "starting"
        assert adapter.capabilities().frame_format is None
        publish()
        assert adapter.status().state == "ready"
        assert adapter.capabilities().frame_format == (
            "BGR_CPU" if backend == "cpu" else "NV12_NVMM"
        )

        worker = camera._thread
        assert worker is not None
        worker.join(timeout=1)
        assert not worker.is_alive()
        camera._recovery.set_state("failed")
        camera._running.clear()
        assert adapter.status().state == "error"
        assert adapter.capabilities().frame_format is None

        camera.start()
        assert adapter.status().state == "starting"
        publish()
        assert adapter.status().state == "ready"
    finally:
        camera.stop()
    assert adapter.status().state == "unavailable"


def test_mock_camera_publication_state_resets_on_restart() -> None:
    """Use the public mock without interpreting retained JPEGs as new frames."""
    camera = MockCamera(auto_start=False)
    adapter = CameraReadAdapter(camera)

    camera.start()
    assert adapter.status().state == "starting"
    camera.publish_jpeg(b"synthetic")
    assert adapter.status().state == "ready"
    camera.stop()
    assert adapter.status().state == "unavailable"
    camera.start()
    assert adapter.status().state == "starting"
    camera.stop()


def test_profile_requires_explicit_selection() -> None:
    """Matching pipeline defaults provide no evidence of attached hardware."""
    profile = get_camera_profile("imx219-1080p")
    camera = Camera(config=profile.config)
    assert CameraReadAdapter(camera).capabilities() == CameraCapabilities()
    assert CameraReadAdapter(camera, profile=profile).capabilities() == (
        CameraCapabilities(sensor_model="IMX219-77")
    )


@pytest.mark.parametrize(
    ("model", "mode", "backend", "frame_format"),
    [
        ("IMX219-77", ArgusMode(2, 1920, 1080, 30), "cpu", "BGR_CPU"),
        ("IMX477", ArgusMode(1, 3840, 2160, 30), "gpu", "NV12_NVMM"),
    ],
)
def test_sensor_projection_uses_cached_controller_and_inspection(
    model: str,
    mode: ArgusMode,
    backend: CameraBackend,
    frame_format: str,
) -> None:
    """Filter unsupported controls and allow only current published formats."""
    apply = Mock()
    controller = CameraController(
        capabilities=ArgusCapabilities(
            DEFAULT_CAPABILITIES.source_properties, (ArgusMode(0),), model
        ),
        runtime_handler=apply,
    )
    inspected = ArgusCapabilities(
        frozenset({"exposuretimerange", "ispdigitalgainrange", "wbmode"}),
        (replace(mode, name="/private/mode-name"),),
        model,
    )
    source = Mock()
    adapter = CameraReadAdapter(
        source,
        backend=backend,
        sensor_capabilities=controller.capabilities,
        inspected_capabilities=inspected,
    )
    source.stats.return_value = _STATS
    capabilities = adapter.capabilities()
    assert capabilities.frame_format == frame_format
    assert capabilities.sensor_model == model
    assert capabilities.sensor_modes == (
        SensorMode(mode.index, mode.width, mode.height, mode.max_fps, mode.hdr),
    )
    assert capabilities.controls == ControlCapabilities(ExposureBounds(), False)

    source.stats.return_value = replace(_STATS, state="recovering")
    assert adapter.capabilities() == replace(capabilities, frame_format=None)
    source.stats.return_value = replace(_STATS, has_published_frame=False)
    assert adapter.capabilities() == replace(capabilities, frame_format=None)
    assert source.mock_calls == [call.stats(), call.stats(), call.stats()]
    assert controller.revision == 0
    apply.assert_not_called()


def test_inspection_alone_only_establishes_negative_control_support() -> None:
    """Generic defaults and plugin properties do not identify sensor support."""
    inspected = ArgusCapabilities(frozenset({"exposuretimerange"}))
    camera = MockCamera(auto_start=False)
    adapter = CameraReadAdapter(
        camera,
        sensor_capabilities=DEFAULT_CAPABILITIES,
        inspected_capabilities=inspected,
    )
    assert adapter.capabilities() == CameraCapabilities(
        controls=ControlCapabilities(gain=False)
    )


def test_inspected_identity_discards_metadata_for_a_different_sensor() -> None:
    """Never attach stale mode or control declarations to another sensor model."""
    declared = ArgusCapabilities(frozenset({"gainrange"}), (ArgusMode(2),), "IMX219")
    inspected = ArgusCapabilities(frozenset({"gainrange"}), model="IMX477")
    adapter = CameraReadAdapter(
        MockCamera(auto_start=False),
        sensor_capabilities=declared,
        inspected_capabilities=inspected,
    )
    assert adapter.capabilities() == CameraCapabilities(
        sensor_model="IMX477", controls=ControlCapabilities(exposure_us=False)
    )


def test_capabilities_filter_unpublishable_sensor_metadata() -> None:
    """Drop private model labels and dimensions outside the JSON integer range."""
    declared = ArgusCapabilities(
        frozenset(), (ArgusMode(0, width=1 << 53),), "/private/model.yml"
    )
    adapter = CameraReadAdapter(
        MockCamera(auto_start=False), sensor_capabilities=declared
    )
    assert adapter.capabilities() == CameraCapabilities(
        sensor_modes=(SensorMode(0, hdr=False),),
        controls=ControlCapabilities(False, False),
    )
