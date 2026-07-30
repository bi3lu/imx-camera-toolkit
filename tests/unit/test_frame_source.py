"""Tests for the minimal external frame-source integration contract."""

from __future__ import annotations

import pytest

from imx_camera_toolkit import Frame
from imx_camera_toolkit.frames import CameraFrameSource, FrameSource


class RecordingCamera:
    """Minimal camera substitute exposing the stable raw-frame read method."""

    def __init__(self, frame: Frame | None) -> None:
        """Store one deterministic frame and an empty call log."""
        self.frame = frame
        self.calls: list[tuple[float, bool]] = []

    def read(self, timeout: float = 2.0, copy: bool = True) -> Frame | None:
        """Record source options and return the configured raw frame."""
        self.calls.append((timeout, copy))
        return self.frame


def _frame() -> Frame:
    """Create a deterministic camera frame for adapter tests."""
    return Frame(
        image=bytearray(b"frame"),
        sequence=1,
        timestamp_ns=1,
        capture_timestamp_ns=None,
        width=1,
        height=1,
        format="BGR",
    )


def test_camera_frame_source_forwards_raw_frame_reads() -> None:
    """Adapter must forward timeout and default to a shared raw payload."""
    camera = RecordingCamera(_frame())
    source = CameraFrameSource(camera)

    frame = source.read(timeout=0.5)

    assert isinstance(source, FrameSource)
    assert frame is camera.frame
    assert camera.calls == [(0.5, False)]


def test_camera_frame_source_uses_camera_timeout_when_omitted() -> None:
    """An omitted timeout must defer to the underlying camera default."""
    camera = RecordingCamera(_frame())
    source = CameraFrameSource(camera, copy=True)

    assert source.read() is camera.frame
    assert camera.calls == [(2.0, True)]


@pytest.mark.parametrize(
    ("copy", "message"),
    [(1, "copy")],
)
def test_camera_frame_source_validates_copy_configuration(
    copy: object,
    message: str,
) -> None:
    """Image ownership must be chosen explicitly as a boolean."""
    with pytest.raises(ValueError, match=message):
        CameraFrameSource(RecordingCamera(_frame()), copy=copy)  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout", [-1, float("inf"), True, "one"])
def test_camera_frame_source_validates_timeout(timeout: object) -> None:
    """Source timeout must be numeric, non-negative, or omitted."""
    source = CameraFrameSource(RecordingCamera(_frame()))

    with pytest.raises(ValueError, match="timeout"):
        source.read(timeout=timeout)  # type: ignore[arg-type]
