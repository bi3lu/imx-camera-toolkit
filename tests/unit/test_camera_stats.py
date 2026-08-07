"""Tests for the stable camera diagnostics contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from time import monotonic_ns

import pytest

from imx_camera_toolkit import Camera, CameraStats, PipelineStage


def test_camera_stats_reports_a_consistent_capture_snapshot() -> None:
    """Diagnostics must expose counters, timing, and state without hardware."""
    camera = Camera()
    camera._running.set()
    timestamp_ns = monotonic_ns()
    camera._record_capture(timestamp_ns - 200_000_000)
    camera._record_capture(timestamp_ns - 100_000_000)
    camera._record_capture(timestamp_ns)
    camera._record_dropped_frame(failed_read=True)
    camera._record_dropped_frame(failed_read=True)
    camera.recoveries = 1

    stats = camera.stats()

    assert stats == CameraStats(
        captured_frames=3,
        dropped_frames=2,
        capture_fps=10.0,
        last_frame_timestamp_ns=timestamp_ns,
        recovery_count=1,
        consecutive_failures=2,
        running=True,
    )


def test_camera_stats_resets_failure_streak_and_rate_after_shutdown() -> None:
    """A successful read clears failures and stopped capture reports zero FPS."""
    camera = Camera()
    camera._running.set()
    camera._record_dropped_frame(failed_read=True)
    timestamp_ns = monotonic_ns()
    camera._record_capture(timestamp_ns - 100_000_000)
    camera._record_capture(timestamp_ns)

    assert camera.stats().consecutive_failures == 0
    assert camera.stats().capture_fps == 10.0

    camera._running.clear()

    assert camera.stats().running is False
    assert camera.stats().capture_fps == 0.0


def test_camera_stats_is_immutable() -> None:
    """Consumers must receive an immutable diagnostic value object."""
    stats = Camera().stats()

    with pytest.raises(FrozenInstanceError):
        stats.captured_frames = 1  # type: ignore[misc]


def test_camera_stats_aggregates_stage_latency_and_consumer_drops() -> None:
    """Consumers can report timings and skips without retaining frame history."""
    camera = Camera()
    camera.record_stage_latency(PipelineStage.INFERENCE, 2_000_000)
    camera.record_stage_latency("inference", 4_000_000)
    camera.record_consumer_drop("inference", 2)

    stats = camera.stats()

    assert stats.pipeline.inference.samples == 2
    assert stats.pipeline.inference.last_duration_ns == 4_000_000
    assert stats.pipeline.inference.mean_duration_ns == 3_000_000
    assert dict(stats.consumer_dropped_frames) == {"inference": 2}
    assert camera.consumer_dropped_frames == {"inference": 2}
