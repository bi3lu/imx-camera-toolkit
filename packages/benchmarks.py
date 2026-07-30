"""Deterministic capture and MJPEG streaming microbenchmarks."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .stream.stream import MJPEGStream
from .testing.mock_camera import MockCamera

BENCHMARK_JPEG = b"\xff\xd8\xff\xd9"


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


def _result(name: str, frames: int, started_at: float) -> BenchmarkResult:
    """Build a result while avoiding division by zero on fast hosts."""
    duration = max(time.perf_counter() - started_at, 1e-9)
    return BenchmarkResult(name, frames, duration, frames / duration)


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
