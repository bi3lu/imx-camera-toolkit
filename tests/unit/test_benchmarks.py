"""Unit tests for deterministic benchmark helpers."""

from __future__ import annotations

import time

import pytest

from packages import benchmarks
from packages.benchmarks import (
    benchmark_capture,
    benchmark_cpu_capture,
    benchmark_cpu_capture_jpeg,
    benchmark_cpu_capture_model,
    benchmark_streaming,
)
from packages.camera.config import CameraConfig
from packages.camera.models import CameraStats, Frame


class _FakePhysicalCamera:
    """Deterministic BGR camera for hardware-benchmark control-flow tests."""

    configurations: list[CameraConfig] = []

    def __init__(self, config: CameraConfig) -> None:
        """Retain the requested preview mode and initialize one latest slot."""
        self.config = config
        self.sequence = 0
        self.frame: Frame | None = None
        self.configurations.append(config)

    def __enter__(self) -> _FakePhysicalCamera:
        """Return this already available synthetic camera."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release no external resources."""

    def stats(self) -> CameraStats:
        """Return counters matching frames produced so far."""
        return CameraStats(
            captured_frames=self.sequence,
            dropped_frames=0,
            capture_fps=0.0,
            last_frame_timestamp_ns=(
                self.frame.timestamp_ns if self.frame is not None else None
            ),
            recovery_count=0,
            consecutive_failures=0,
            running=True,
        )

    def wait_for_raw_frame(
        self,
        previous_frame_number: int,
        *,
        timeout: float,
    ) -> tuple[int, object]:
        """Publish one distinct synthetic BGR image per request."""
        assert timeout > 0
        assert previous_frame_number < self.sequence + 1
        self.sequence += 1
        image = bytearray((self.sequence,))
        self.frame = Frame(
            image=image,
            sequence=self.sequence,
            timestamp_ns=time.monotonic_ns(),
            capture_timestamp_ns=None,
            width=1,
            height=1,
            format="BGR",
        )
        return self.sequence, image

    def latest_frame(self, *, copy: bool) -> Frame | None:
        """Return the synthetic latest frame without another API copy."""
        assert copy is False
        return self.frame


@pytest.mark.benchmark
def test_capture_benchmark_reports_throughput() -> None:
    """Capture benchmark must report every requested synthetic frame."""
    result = benchmark_capture(10)

    assert result.frames == 10
    assert result.frames_per_second > 0


@pytest.mark.benchmark
def test_streaming_benchmark_reports_throughput() -> None:
    """Streaming benchmark must frame every requested synthetic frame."""
    result = benchmark_streaming(10)

    assert result.frames == 10
    assert result.frames_per_second > 0


def test_cpu_benchmarks_cover_capture_jpeg_and_application_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All CPU modes must share capture while preserving their distinct cost."""
    _FakePhysicalCamera.configurations.clear()
    monkeypatch.setattr(benchmarks, "Camera", _FakePhysicalCamera)
    modeled_images: list[object] = []

    capture = benchmark_cpu_capture(frames=3)
    jpeg = benchmark_cpu_capture_jpeg(frames=3)
    model = benchmark_cpu_capture_model(modeled_images.append, frames=3)

    assert capture.name == "camera-raw"
    assert jpeg.name == "camera-preview"
    assert model.name == "camera-cpu-model"
    assert [config.enable_preview for config in _FakePhysicalCamera.configurations] == [
        False,
        True,
        False,
    ]
    assert modeled_images == [bytearray((1,)), bytearray((2,)), bytearray((3,))]


def test_cpu_model_benchmark_rejects_incompatible_options() -> None:
    """CPU model and JPEG costs must remain independently measurable."""
    with pytest.raises(ValueError, match="separate"):
        benchmarks.benchmark_camera_capture(
            frames=1,
            preview=True,
            model=lambda image: image,
        )
