"""Deterministic capture and MJPEG streaming microbenchmarks."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from math import ceil
from typing import Protocol

from .camera.camera import Camera, CameraConfig, CameraTimeoutError
from .camera.gpu_camera import GpuCamera
from .camera.models import GpuFrame
from .stream.stream import MJPEGStream
from .telemetry import TegrastatsSampler
from .testing.mock_camera import MockCamera

BENCHMARK_JPEG = b"\xff\xd8\xff\xd9"
CpuModel = Callable[[object], object]


class ResourceSampler(Protocol):
    """Resource sampler boundary used by deterministic unit tests."""

    def start(self) -> None:
        """Start collecting samples."""
        ...

    def stop(self) -> float | None:
        """Stop sampling and return average GPU utilization."""
        ...


@dataclass(frozen=True)
class BenchmarkResult:
    """Throughput result for one deterministic benchmark."""

    name: str
    frames: int
    duration_seconds: float
    frames_per_second: float

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-ready benchmark result."""
        return asdict(self)


@dataclass(frozen=True)
class CameraBenchmarkResult:
    """End-to-end physical-camera capture benchmark result.

    Unlike :class:`BenchmarkResult`, this records the actual connected camera's
    capture counters and frame latency. It is intentionally opt-in and must not
    be interpreted as a benchmark of a different Jetson, sensor, network, or
    browser configuration.
    """

    name: str
    frames: int
    duration_seconds: float
    frames_per_second: float
    captured_frames: int
    dropped_frames: int
    mean_latency_ms: float
    p95_latency_ms: float = 0.0
    process_cpu_percent: float = 0.0
    gpu_utilization_percent: float | None = None
    width: int = 0
    height: int = 0
    backend: str = "cpu"

    def as_dict(self) -> dict[str, float | int | str | None]:
        """Return a JSON-ready physical-camera benchmark result."""
        return asdict(self)


def _result(name: str, frames: int, started_at: float) -> BenchmarkResult:
    """Build a result while avoiding division by zero on fast hosts."""
    duration = max(time.perf_counter() - started_at, 1e-9)
    return BenchmarkResult(name, frames, duration, frames / duration)


def _p95_latency_ms(latencies_ns: list[int]) -> float:
    """Return the nearest-rank 95th percentile in milliseconds."""
    ordered = sorted(latencies_ns)
    index = max(ceil(0.95 * len(ordered)) - 1, 0)
    return ordered[index] / 1_000_000


def _resource_sampler(sampler: ResourceSampler | None) -> ResourceSampler:
    """Resolve an injectable sampler without probing hardware during tests."""
    return sampler or TegrastatsSampler()


def benchmark_capture(frames: int = 1_000) -> BenchmarkResult:
    """Measure in-memory JPEG publication throughput through ``MockCamera``.

    Args:
        frames: Number of synthetic capture frames to publish.

    Returns:
        Capture publication throughput. It is deterministic and does not
        represent CSI sensor, ISP, or JPEG encoder performance.
    """
    if frames <= 0:
        raise ValueError("frames must be greater than zero")

    camera = MockCamera()
    started_at = time.perf_counter()

    for _ in range(frames):
        camera.publish_jpeg(BENCHMARK_JPEG)

    return _result("capture", frames, started_at)


def benchmark_streaming(frames: int = 1_000) -> BenchmarkResult:
    """Measure multipart MJPEG framing throughput with a mock camera.

    Args:
        frames: Number of synthetic JPEG frames to frame as MJPEG parts.

    Returns:
        Streaming throughput. It excludes network transport and browser work.
    """
    if frames <= 0:
        raise ValueError("frames must be greater than zero")

    camera = MockCamera()
    camera.publish_jpeg(BENCHMARK_JPEG)
    iterator = iter(MJPEGStream(camera, timeout=0.01))
    started_at = time.perf_counter()
    next(iterator)

    for _ in range(frames - 1):
        camera.publish_jpeg(BENCHMARK_JPEG)
        next(iterator)

    return _result("streaming", frames, started_at)


def benchmark_camera_capture(
    frames: int = 300,
    *,
    preview: bool = False,
    timeout: float = 5.0,
    config: CameraConfig | None = None,
    model: CpuModel | None = None,
    resource_sampler: ResourceSampler | None = None,
) -> CameraBenchmarkResult:
    """Measure raw capture from a connected CSI camera.

    The benchmark opens one camera, waits for distinct latest raw frames, and
    closes it before returning. Preview measures capture plus JPEG encoding;
    an optional callable measures capture plus a real application-owned CPU
    model. It measures raw capture throughput, source read failures, and
    publication-to-consumer latency; it does not measure network or browser
    performance.

    Args:
        frames: Number of distinct raw frames to observe.
        preview: Whether to enable the independent JPEG preview path.
        timeout: Maximum wait for each raw frame in seconds.
        config: Optional base camera configuration for the tested sensor.
        model: Optional CPU model callable receiving each BGR image. It cannot
            be combined with JPEG preview.
        resource_sampler: Optional callback returning process and device
            resource measurements for the completed benchmark.

    Returns:
        End-to-end result for the current local Jetson and sensor setup.

    Raises:
        CameraTimeoutError: If the camera does not produce a required frame.
        ValueError: If benchmark arguments are invalid.
    """
    if frames <= 0:
        raise ValueError("frames must be greater than zero")

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    if not isinstance(preview, bool):
        raise ValueError("preview must be a boolean")

    if model is not None and not callable(model):
        raise ValueError("model must be callable or None")

    if preview and model is not None:
        raise ValueError("CPU model and JPEG preview benchmarks are separate")

    resolved_config = replace(
        config or CameraConfig(),
        enable_preview=preview,
    )
    camera = Camera(resolved_config)
    previous_frame_number = -1
    latencies_ns: list[int] = []
    sampler = _resource_sampler(resource_sampler)

    with camera:
        initial_stats = camera.stats()
        sampler.start()
        started_at = time.monotonic()
        started_cpu = time.process_time()

        try:
            for _ in range(frames):
                frame_number, image = camera.wait_for_raw_frame(
                    previous_frame_number,
                    timeout=timeout,
                )

                if image is None or frame_number == previous_frame_number:
                    raise CameraTimeoutError(
                        f"camera did not provide a frame within {timeout:.1f}s"
                    )

                previous_frame_number = frame_number
                frame = camera.latest_frame(copy=False)

                if frame is None:
                    raise CameraTimeoutError(
                        "camera published a frame without raw metadata"
                    )

                if model is not None:
                    model(frame.image)

                latencies_ns.append(max(time.monotonic_ns() - frame.timestamp_ns, 0))

        finally:
            cpu_seconds = time.process_time() - started_cpu
            duration = max(time.monotonic() - started_at, 1e-9)
            gpu_percent = sampler.stop()

        final_stats = camera.stats()

    mean_latency_ms = sum(latencies_ns) / len(latencies_ns) / 1_000_000

    if preview:
        benchmark_name = "camera-preview"

    elif model is not None:
        benchmark_name = "camera-cpu-model"

    else:
        benchmark_name = "camera-raw"

    return CameraBenchmarkResult(
        name=benchmark_name,
        frames=frames,
        duration_seconds=duration,
        frames_per_second=frames / duration,
        captured_frames=final_stats.captured_frames - initial_stats.captured_frames,
        dropped_frames=final_stats.dropped_frames - initial_stats.dropped_frames,
        mean_latency_ms=mean_latency_ms,
        p95_latency_ms=_p95_latency_ms(latencies_ns),
        process_cpu_percent=cpu_seconds / duration * 100,
        gpu_utilization_percent=gpu_percent,
        width=resolved_config.output_width,
        height=resolved_config.output_height,
        backend="cpu",
    )


def benchmark_gpu_capture(
    frames: int = 300,
    *,
    timeout: float = 5.0,
    config: CameraConfig | None = None,
    consumer: Callable[[GpuFrame], object] | None = None,
    resource_sampler: ResourceSampler | None = None,
) -> CameraBenchmarkResult:
    """Benchmark stable NVMM capture and an optional GPU consumer."""
    if frames <= 0:
        raise ValueError("frames must be greater than zero")

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    if consumer is not None and not callable(consumer):
        raise ValueError("consumer must be callable or None")

    resolved_config = replace(config or CameraConfig(), enable_preview=False)
    camera = GpuCamera(resolved_config)
    subscription = camera.subscribe_latest("benchmark-gpu")
    sampler = _resource_sampler(resource_sampler)
    latencies_ns: list[int] = []

    with camera:
        initial_stats = camera.stats()
        sampler.start()
        started_at = time.monotonic()
        started_cpu = time.process_time()

        try:
            for _ in range(frames):
                frame = subscription.receive(timeout)

                if frame is None:
                    raise CameraTimeoutError(
                        f"GPU camera did not provide a frame within {timeout:.1f}s"
                    )

                if consumer is not None:
                    consumer(frame)

                latencies_ns.append(max(time.monotonic_ns() - frame.timestamp_ns, 0))

        finally:
            cpu_seconds = time.process_time() - started_cpu
            duration = max(time.monotonic() - started_at, 1e-9)
            gpu_percent = sampler.stop()

        final_stats = camera.stats()

    mean_latency_ms = sum(latencies_ns) / len(latencies_ns) / 1_000_000
    return CameraBenchmarkResult(
        name="gpu-consumer" if consumer is not None else "gpu-capture",
        frames=frames,
        duration_seconds=duration,
        frames_per_second=frames / duration,
        captured_frames=final_stats.captured_frames - initial_stats.captured_frames,
        dropped_frames=final_stats.dropped_frames - initial_stats.dropped_frames,
        mean_latency_ms=mean_latency_ms,
        p95_latency_ms=_p95_latency_ms(latencies_ns),
        process_cpu_percent=cpu_seconds / duration * 100,
        gpu_utilization_percent=gpu_percent,
        width=resolved_config.output_width,
        height=resolved_config.output_height,
        backend="gpu",
    )


def benchmark_cpu_capture(
    frames: int = 300,
    *,
    timeout: float = 5.0,
    config: CameraConfig | None = None,
) -> CameraBenchmarkResult:
    """Benchmark compatible BGR/CPU capture without JPEG or a model."""
    return benchmark_camera_capture(
        frames,
        preview=False,
        timeout=timeout,
        config=config,
    )


def benchmark_cpu_capture_jpeg(
    frames: int = 300,
    *,
    timeout: float = 5.0,
    config: CameraConfig | None = None,
) -> CameraBenchmarkResult:
    """Benchmark compatible BGR/CPU capture with JPEG encoding enabled."""
    return benchmark_camera_capture(
        frames,
        preview=True,
        timeout=timeout,
        config=config,
    )


def benchmark_cpu_capture_model(
    model: CpuModel,
    frames: int = 300,
    *,
    timeout: float = 5.0,
    config: CameraConfig | None = None,
) -> CameraBenchmarkResult:
    """Benchmark compatible BGR/CPU capture plus an application CPU model."""
    return benchmark_camera_capture(
        frames,
        preview=False,
        timeout=timeout,
        config=config,
        model=model,
    )
