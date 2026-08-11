"""Opt-in benchmark test entry points for local performance checks."""

from __future__ import annotations

import pytest

from imx_camera_toolkit._internal.benchmarks import (
    benchmark_capture,
    benchmark_streaming,
)


@pytest.mark.benchmark
def test_capture_throughput_smoke() -> None:
    """Run a larger synthetic capture benchmark outside normal CI."""
    assert benchmark_capture(1_000).frames_per_second > 0


@pytest.mark.benchmark
def test_streaming_throughput_smoke() -> None:
    """Run a larger synthetic streaming benchmark outside normal CI."""
    assert benchmark_streaming(1_000).frames_per_second > 0
