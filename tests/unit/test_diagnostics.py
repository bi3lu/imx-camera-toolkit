"""Tests for opt-in camera diagnostics without physical hardware."""

from __future__ import annotations

import pytest

from imx_camera_toolkit._internal import diagnostics
from imx_camera_toolkit._internal.camera.config import CameraConfig


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
