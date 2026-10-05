"""Tests for opt-in camera diagnostics without physical hardware."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from imx_camera_toolkit._internal import diagnostics
from imx_camera_toolkit._internal.camera.config import CameraConfig
from imx_camera_toolkit.testing import mock_gpu_frame


@pytest.mark.parametrize(
    ("output", "timed_out", "status"),
    [
        ("imx219 9-0010: i2c read probe (-121)", False, "i2c-error"),
        ("Device or resource busy", False, "busy"),
        ("No cameras available", False, "missing"),
        ("pipeline did not finish", True, "no-frame"),
    ],
)
def test_argus_probe_failure_categories_are_operationally_distinct(
    output: str,
    timed_out: bool,
    status: str,
) -> None:
    """Sensor probes must preserve actionable failure categories."""
    actual_status, _ = diagnostics._classify_argus_failure(
        output,
        timed_out=timed_out,
    )

    assert actual_status == status


def test_argus_probe_tests_each_id_without_using_v4l2_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every requested Argus ID must run its own bounded profile pipeline."""
    commands: list[tuple[str, ...]] = []

    def run(
        command: Sequence[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(command))
        assert kwargs["timeout"] == 0.25
        if "sensor-id=0" in command:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="No cameras available",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "imx_camera_toolkit._internal.diagnostics.shutil.which",
        lambda _: "gst-launch-1.0",
    )
    monkeypatch.setattr(
        "imx_camera_toolkit._internal.diagnostics.subprocess.run",
        run,
    )

    checks = diagnostics.probe_argus_sensors((0, 1), timeout=0.25)

    assert [(check.name, check.status, check.detail) for check in checks] == [
        (
            "argus_sensor_0",
            "missing",
            "sensor-id=0; Argus sensor is not available: No cameras available",
        ),
        (
            "argus_sensor_1",
            "ok",
            "sensor-id=1; captured one frame",
        ),
        (
            "argus_mode_1_imx219-1080p",
            "ok",
            "sensor-id=1, profile=imx219-1080p, sensor-mode=2, "
            "1920x1080@30 FPS; captured one frame",
        ),
    ]
    assert all(command[0] == "gst-launch-1.0" for command in commands)
    assert all(
        not any("/dev/video" in part for part in command) for command in commands
    )


def test_device_tree_diagnostic_reports_sensor_nodes(tmp_path: Path) -> None:
    """Camera-compatible Device Tree nodes must be included in inventory."""
    sensor_node = tmp_path / "soc" / "i2c@0" / "imx219@10"
    sensor_node.mkdir(parents=True)
    (sensor_node / "compatible").write_bytes(b"nvidia,imx219\x00")

    check = diagnostics._device_tree_camera_check(tmp_path)

    assert check.status == "ok"
    assert "imx219@10" in check.detail
    assert "nvidia,imx219" in check.detail


def test_kernel_diagnostic_marks_i2c_minus_121_as_hardware_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I2C -121 must point operators to hardware or the sensor driver."""
    monkeypatch.setattr(
        "imx_camera_toolkit._internal.diagnostics.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=("imx219 9-0010: i2c read probe (-121)\n" "imx219: probe failed\n"),
            stderr="",
        ),
    )

    check = diagnostics._kernel_camera_check()

    assert check.status == "i2c-error"
    assert "hardware/driver" in check.detail
    assert "Python cannot repair" in check.detail


class _FakeCamera:
    """Minimal raw camera used to verify smoke-test lifecycle behavior."""

    def __init__(self, config: CameraConfig) -> None:
        """Store the requested raw-only configuration."""
        self.config = config
        self.running = False

    def start(self) -> None:
        """Mark the fake capture backend as opened."""
        self.running = True

    def wait_for_raw_frame(
        self,
        previous_frame_number: int,
        *,
        timeout: float,
    ) -> tuple[int, object | None]:
        """Return one distinct opaque frame for every smoke-test read."""
        assert timeout > 0
        return previous_frame_number + 1, object()

    def stop(self) -> None:
        """Mark the fake backend as released."""
        self.running = False


def test_camera_smoke_test_checks_open_frame_rate_and_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hardware diagnostic must cover the full declared lifecycle."""
    monkeypatch.setattr(diagnostics, "Camera", _FakeCamera)

    checks = diagnostics.run_camera_smoke_test(frames=3, timeout=0.1)

    assert [(check.name, check.status) for check in checks] == [
        ("camera_open", "ok"),
        ("first_frame", "ok"),
        ("capture_rate", "ok"),
        ("camera_release", "ok"),
    ]


def test_camera_smoke_test_supports_the_gpu_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hardware diagnostic must exercise retained NVMM subscriptions."""

    class _Subscription:
        def __init__(self, camera: _FakeGpuCamera) -> None:
            self.camera = camera

        def receive(self, timeout: float) -> object:
            assert timeout > 0
            self.camera.sequence += 1
            return mock_gpu_frame(object(), sequence=self.camera.sequence)

        def close(self) -> None:
            pass

    class _FakeGpuCamera:
        def __init__(self, config: CameraConfig) -> None:
            self.config = config
            self.running = False
            self.sequence = 0

        def subscribe_latest(self, name: str) -> _Subscription:
            assert name == "diagnostic-smoke-test"
            return _Subscription(self)

        def start(self) -> None:
            self.running = True

        def stop(self) -> None:
            self.running = False

    monkeypatch.setattr(diagnostics, "GpuCamera", _FakeGpuCamera)

    checks = diagnostics.run_camera_smoke_test(
        frames=3,
        timeout=0.1,
        backend="gpu",
    )

    assert [(check.name, check.status) for check in checks] == [
        ("camera_open", "ok"),
        ("first_frame", "ok"),
        ("capture_rate", "ok"),
        ("camera_release", "ok"),
    ]
