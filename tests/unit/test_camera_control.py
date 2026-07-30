"""Unit tests for validated Argus control state transitions."""

from __future__ import annotations

from packages.camera_control.camera_control import (
    CameraCapabilities,
    CameraController,
    SensorMode,
)


def test_controller_builds_manual_controls_and_restores_profile() -> None:
    """Manual exposure/gain must be emitted and profiles restored atomically."""
    controller = CameraController()
    update = controller.update(exposure_us=5_000, gain=2.0, awb_mode="daylight")

    assert 'exposuretimerange="5000000 5000000"' in update.source_properties
    assert 'gainrange="2 2"' in update.source_properties
    controller.save_profile("manual")
    controller.set_gain(3.0)
    assert controller.apply_profile("manual").settings.gain == 2.0


def test_controller_selects_declared_native_hdr_mode() -> None:
    """Native HDR must select an explicit declared HDR sensor mode."""
    capabilities = CameraCapabilities(
        source_properties=frozenset(
            {"wbmode", "awblock", "tnr-mode", "sensor-mode"}
        ),
        sensor_modes=(SensorMode(0, hdr=False), SensorMode(1, hdr=True)),
    )
    controller = CameraController(capabilities=capabilities)

    update = controller.set_hdr(True)

    assert update.settings.sensor_mode == 1
    assert update.restart_required
