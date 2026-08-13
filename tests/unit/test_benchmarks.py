"""Unit tests for deterministic benchmark helpers."""

from __future__ import annotations

import time

import pytest

from imx_camera_toolkit._internal import benchmarks
from imx_camera_toolkit._internal.benchmarks import (
    benchmark_camera_capture,
    benchmark_capture,
    benchmark_cpu_capture,
    benchmark_cpu_capture_jpeg,
    benchmark_cpu_capture_model,
    benchmark_gpu_capture,
    benchmark_streaming,
)
from imx_camera_toolkit._internal.camera.config import CameraConfig
from imx_camera_toolkit._internal.camera.models import CameraStats, Frame
from imx_camera_toolkit._internal.telemetry import TegrastatsSampler
from imx_camera_toolkit._internal.testing import mock_gpu_frame


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


class _FakeResourceSampler:
    """Return deterministic Jetson GPU utilization without tegrastats."""

    def __init__(self, gpu_percent: float = 37.5) -> None:
        self.gpu_percent = gpu_percent
        self.started = False

    def start(self) -> None:
        """Record benchmark sampling startup."""
        self.started = True

    def stop(self) -> float:
        """Return one deterministic aggregate."""
        assert self.started
        return self.gpu_percent


class _FakeGpuSubscription:
    """Produce one fresh borrowed test frame on every receive."""

    def __init__(self, camera: _FakeGpuCamera) -> None:
        self.camera = camera

    def receive(self, timeout: float) -> object:
        """Return one unique frame immediately."""
        assert timeout > 0
        self.camera.sequence += 1
        return mock_gpu_frame(
            object(),
            sequence=self.camera.sequence,
            timestamp_ns=time.monotonic_ns(),
            width=self.camera.config.output_width,
            height=self.camera.config.output_height,
        )


class _FakeGpuCamera:
    """Stable GPU camera test double for benchmark orchestration."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.sequence = 0

    def subscribe_latest(self, name: str) -> _FakeGpuSubscription:
        """Create the benchmark's independent latest-frame slot."""
        assert name == "benchmark-gpu"
        return _FakeGpuSubscription(self)

    def __enter__(self) -> _FakeGpuCamera:
        """Start no external resources."""
        return self

    def __exit__(self, *_: object) -> None:
        """Stop no external resources."""

    def stats(self) -> CameraStats:
        """Expose counters matching frames emitted by the subscription."""
        return CameraStats(
            captured_frames=self.sequence,
            dropped_frames=0,
            capture_fps=30.0,
            last_frame_timestamp_ns=None,
            recovery_count=0,
            consecutive_failures=0,
            running=True,
        )


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


def test_camera_benchmark_reports_latency_cpu_gpu_and_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Physical reports must contain deployable resource and latency fields."""
    monkeypatch.setattr(benchmarks, "Camera", _FakePhysicalCamera)
    sampler = _FakeResourceSampler()

    result = benchmark_camera_capture(
        frames=3,
        config=CameraConfig(output_width=1280, output_height=720),
        resource_sampler=sampler,
    )

    assert result.mean_latency_ms >= 0
    assert result.p95_latency_ms >= result.mean_latency_ms
    assert result.process_cpu_percent >= 0
    assert result.gpu_utilization_percent == 37.5
    assert (result.width, result.height) == (1280, 720)
    assert result.backend == "cpu"


def test_gpu_benchmark_uses_stable_latest_frame_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU reports must avoid polling duplicate frames and expose resources."""
    monkeypatch.setattr(benchmarks, "GpuCamera", _FakeGpuCamera)
    consumed: list[int] = []

    result = benchmark_gpu_capture(
        frames=3,
        config=CameraConfig(output_width=1920, output_height=1080),
        consumer=lambda frame: consumed.append(frame.sequence),
        resource_sampler=_FakeResourceSampler(62.5),
    )

    assert consumed == [1, 2, 3]
    assert result.frames_per_second > 0
    assert result.gpu_utilization_percent == 62.5
    assert (result.width, result.height) == (1920, 1080)
    assert result.backend == "gpu"


def test_tegrastats_parser_accepts_jetpack_gr3d_output() -> None:
    """GPU sampling must parse documented Jetson utilization fields."""
    line = "RAM 1000/8000MB GR3D_FREQ 42%@624 EMC_FREQ 1%@1600"

    assert TegrastatsSampler.parse_gpu_percent(line) == 42.0
    assert TegrastatsSampler.parse_gpu_percent("RAM 1000/8000MB") is None


def test_cpu_model_benchmark_rejects_incompatible_options() -> None:
    """CPU model and JPEG costs must remain independently measurable."""
    with pytest.raises(ValueError, match="separate"):
        benchmarks.benchmark_camera_capture(
            frames=1,
            preview=True,
            model=lambda image: image,
        )
