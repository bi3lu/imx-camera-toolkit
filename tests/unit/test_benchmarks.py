"""Unit tests for deterministic benchmark helpers."""

from __future__ import annotations

import pytest

from packages.benchmarks import benchmark_capture, benchmark_streaming


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
