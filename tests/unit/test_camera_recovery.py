"""Unit tests for camera backend recovery without physical hardware."""

from __future__ import annotations

from typing import Any

import pytest

from imx_camera_toolkit import CameraRecoveryError
from imx_camera_toolkit._internal.camera.backends.base import CaptureBackend
from imx_camera_toolkit._internal.camera.camera import Camera, CameraRecoveryPolicy


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


@pytest.mark.parametrize("max_attempts", [0, 1, 3])
@pytest.mark.parametrize("valid_frame", [False, True])
@pytest.mark.parametrize("read_threshold", [1, 2])
def test_recovery_budget_resets_only_after_a_valid_frame(
    monkeypatch: pytest.MonkeyPatch,
    max_attempts: int,
    valid_frame: bool,
    read_threshold: int,
) -> None:
    """Successful opens cannot renew retries; a valid source frame can."""
    camera = Camera(
        enable_preview=False,
        recovery_policy=CameraRecoveryPolicy(
            max_attempts=max_attempts,
            initial_backoff=0,
            max_consecutive_read_failures=read_threshold,
        ),
    )
    backend = RecordingBackend()
    reads = 0
    opens = 0
    expected_attempts = max_attempts * (2 if valid_frame else 1)
    expected_reads = (expected_attempts + 1) * read_threshold + int(valid_frame)

    def read() -> tuple[bool, Any | None]:
        """Bound a broken recovery loop without relying on thread timing."""
        nonlocal reads
        reads += 1

        if reads > expected_reads:
            camera._running.clear()

        if valid_frame and reads == max_attempts * read_threshold + 1:
            return True, bytearray(b"frame")

        return False, None

    def open_backend() -> None:
        """Count successful recovery opens independently of camera metrics."""
        nonlocal opens
        opens += 1

    monkeypatch.setattr(backend, "read", read)
    monkeypatch.setattr(backend, "open", open_backend)
    monkeypatch.setattr(camera, "_create_backend", lambda: backend)
    camera._backend = backend
    camera._running.set()
    subscription = camera.subscribe_latest("recovery-test")
    try:
        camera._capture_loop()

        assert reads == expected_reads
        assert opens == expected_attempts
        assert camera.recovery_attempts == expected_attempts
        assert camera.recoveries == expected_attempts
        assert camera.frames_captured == int(valid_frame)
        assert not camera.running
        assert subscription.closed
        assert isinstance(camera.last_error, CameraRecoveryError)
        assert "exhausted" in str(camera.last_error)

    finally:
        subscription.close()
        camera.stop()

    assert backend.closed


def test_failed_and_successful_opens_share_recovery_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open failures and reopened backends without frames spend one budget."""
    camera = Camera(
        enable_preview=False,
        recovery_policy=CameraRecoveryPolicy(max_attempts=3, initial_backoff=0),
    )
    backend = RecordingBackend()
    opens = 0

    def open_backend() -> None:
        """Fail the first two attempts, then allow the backend to reopen."""
        nonlocal opens
        opens += 1

        if opens < 3:
            raise RuntimeError("backend unavailable")

    monkeypatch.setattr(backend, "open", open_backend)
    monkeypatch.setattr(camera, "_create_backend", lambda: backend)
    camera._running.set()
    try:
        assert camera._recover_backend()
        assert not camera._recover_backend()
        assert opens == 3
        assert camera.recovery_attempts == 3
        assert camera.recoveries == 1

    finally:
        camera.stop()
