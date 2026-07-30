"""Unit tests for camera backend recovery without physical hardware."""

from __future__ import annotations

from typing import Any

from packages.camera.backends.base import CaptureBackend
from packages.camera.camera import Camera, CameraRecoveryPolicy


class RecordingBackend(CaptureBackend):
    """Minimal backend used to verify recovery resource ownership."""

    def __init__(self) -> None:
        """Initialize counters for the test assertions."""
        self.opened = False
        self.closed = False

    def open(self) -> None:
        """Record an open operation."""
        self.opened = True

    def read(self) -> tuple[bool, Any | None]:
        """Return no frame; this method is not used by this test."""
        return False, None

    def close(self) -> None:
        """Record a close operation."""
        self.closed = True


class RecoverableCamera(Camera):
    """Camera with a deterministic replacement backend."""

    def __init__(self, replacement: RecordingBackend) -> None:
        """Initialize a camera configured for immediate recovery retries."""
        super().__init__(
            recovery_policy=CameraRecoveryPolicy(max_attempts=1, initial_backoff=0),
        )
        self.replacement = replacement

    def _create_backend(self) -> CaptureBackend:
        """Return the replacement backend instead of hardware capture."""
        return self.replacement


def test_camera_reopens_backend_after_capture_failure() -> None:
    """Recovery must close the failed backend and install an opened replacement."""
    failed_backend = RecordingBackend()
    replacement = RecordingBackend()
    camera = RecoverableCamera(replacement)
    camera._backend = failed_backend
    camera._running.set()

    assert camera._recover_backend()
    assert failed_backend.closed
    assert replacement.opened
    assert camera.recovery_attempts == 1
    assert camera.recoveries == 1

    camera._running.clear()
