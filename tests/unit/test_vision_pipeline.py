"""Unit tests for the latest-frame AI Vision pipeline core."""

from __future__ import annotations

import time

import pytest

from packages.vision import (
    FileFrameSource,
    Frame,
    InferenceResult,
    NoopFrameProcessor,
    OverlayFrame,
    PipelineEventType,
    SyntheticFrameSource,
    VisionPipeline,
)


class SlowProcessor:
    """Processor intentionally slower than a synthetic producer."""

    def process(self, frame: Frame) -> InferenceResult:
        """Return one independent result after a small deterministic delay."""
        time.sleep(0.01)
        return InferenceResult(
            frame_sequence=frame.sequence,
            values={"processed_sequence": frame.sequence},
        )


class RecordingOverlay:
    """Overlay substitute that does not require OpenCV or image arrays."""

    def render(self, frame: Frame, result: InferenceResult) -> OverlayFrame:
        """Create an independent, testable rendered representation."""
        return OverlayFrame(
            frame_sequence=frame.sequence,
            image={"overlay_for": result.frame_sequence},
        )


def test_pipeline_keeps_only_latest_pending_frame_and_result_has_no_image() -> None:
    """A fast source must drop stale work without embedding images in results."""
    pipeline = VisionPipeline(
        SyntheticFrameSource(max_frames=50),
        SlowProcessor(),
    )

    pipeline.start()

    assert pipeline.wait_until_stopped(timeout=2.0)
    assert pipeline.latest_frame is not None
    assert pipeline.latest_frame.sequence == 49
    assert pipeline.latest_result is not None
    assert pipeline.latest_result.frame_sequence <= pipeline.latest_frame.sequence
    assert not hasattr(pipeline.latest_result, "image")
    assert pipeline.stats.frames_captured == 50
    assert pipeline.stats.frames_dropped > 0
    assert pipeline.stats.frames_processed < pipeline.stats.frames_captured


def test_pipeline_emits_lifecycle_result_and_overlay_events() -> None:
    """Finite sources must publish observable events and an optional overlay."""
    events: list[PipelineEventType] = []
    pipeline = VisionPipeline(
        SyntheticFrameSource(max_frames=3),
        NoopFrameProcessor(),
        overlay=RecordingOverlay(),
    )
    pipeline.subscribe(lambda event: events.append(event.type))

    pipeline.start()

    assert pipeline.wait_until_stopped(timeout=2.0)
    assert events[0] is PipelineEventType.STARTED
    assert PipelineEventType.FRAME_CAPTURED in events
    assert PipelineEventType.RESULT_AVAILABLE in events
    assert PipelineEventType.OVERLAY_AVAILABLE in events
    assert PipelineEventType.SOURCE_EXHAUSTED in events
    assert events[-1] is PipelineEventType.STOPPED
    assert pipeline.latest_overlay is not None
    assert pipeline.latest_overlay.image == {
        "overlay_for": pipeline.latest_overlay.frame_sequence
    }


def test_synthetic_source_resets_when_reopened() -> None:
    """A source can be reused by a later pipeline lifecycle."""
    source = SyntheticFrameSource(max_frames=2)
    source.open()
    assert source.read() == {"synthetic_frame": 0}
    assert source.read() == {"synthetic_frame": 1}
    assert source.exhausted
    source.close()

    source.open()
    assert source.read() == {"synthetic_frame": 0}


def test_file_source_reports_missing_input_before_opening_decoder() -> None:
    """A missing file must fail predictably without requiring OpenCV support."""
    source = FileFrameSource("/tmp/this-file-should-not-exist-imx-vision.mp4")

    with pytest.raises(FileNotFoundError, match="frame-source file does not exist"):
        source.open()
