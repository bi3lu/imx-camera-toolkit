"""Unit tests for the latest-frame AI Vision pipeline core."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from packages.vision import (
    CameraFrameSource,
    FileFrameSource,
    Frame,
    InferenceResult,
    NoopFrameProcessor,
    OverlayFrame,
    PipelineEventType,
    PlaybackMode,
    SyntheticFrameSource,
    VisionPipeline,
)
from packages.vision.sources import file as file_source_module


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


class ManagedProcessor:
    """Processor recording optional model-resource lifecycle calls."""

    def __init__(self) -> None:
        """Initialize lifecycle counters."""
        self.opened = 0
        self.closed = 0

    def open(self) -> None:
        """Record allocation of model resources."""
        self.opened += 1

    def process(self, frame: Frame) -> InferenceResult:
        """Return a result for the supplied frame."""
        return InferenceResult(frame_sequence=frame.sequence)

    def close(self) -> None:
        """Record release of model resources."""
        self.closed += 1


class FailingCloseSource:
    """Finite frame source whose cleanup failure must not block shutdown."""

    def __init__(self) -> None:
        """Initialize a one-frame source."""
        self._opened = False
        self._emitted = False

    @property
    def exhausted(self) -> bool:
        """bool: Whether the source emitted its only frame."""
        return self._emitted

    def open(self) -> None:
        """Prepare the single frame."""
        self._opened = True

    def read(self) -> object | None:
        """Return a frame once and then finish."""
        if not self._opened:
            raise RuntimeError("source is not open")
        if self._emitted:
            return None
        self._emitted = True
        return {"frame": 1}

    def close(self) -> None:
        """Simulate a custom source cleanup failure."""
        raise RuntimeError("source close failed")


class FakeRawCamera:
    """Minimal raw-frame camera for adapter tests without Jetson hardware."""

    def __init__(self) -> None:
        """Initialize camera lifecycle and a single raw frame."""
        self.running = False
        self.raw_frame_number = 1
        self.frame: object | None = {"bgr": "frame"}
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        """Start fake capture."""
        self.start_calls += 1
        self.running = True

    def stop(self) -> None:
        """Stop fake capture."""
        self.stop_calls += 1
        self.running = False

    def wait_for_raw_frame(
        self,
        previous_frame_number: int,
        timeout: float = 2.0,
    ) -> tuple[int, object | None]:
        """Return the configured raw frame without JPEG encoding."""
        del timeout

        if previous_frame_number != self.raw_frame_number:
            return self.raw_frame_number, self.frame

        return self.raw_frame_number, None


class FakeVideoCapture:
    """OpenCV-like deterministic video decoder with a declared native FPS."""

    def __init__(self, frames: list[object], fps: float) -> None:
        """Store video data for source-FPS playback tests."""
        self._frames = frames
        self._fps = fps
        self._index = 0

    def isOpened(self) -> bool:
        """Report a successfully opened decoder."""
        return True

    def read(self) -> tuple[bool, object | None]:
        """Read one fake video frame."""
        if self._index >= len(self._frames):
            return False, None

        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def get(self, _: int) -> float:
        """Return the declared source FPS."""
        return self._fps

    def set(self, _: int, value: int) -> bool:
        """Seek to a frame index for loop playback."""
        self._index = value
        return True

    def release(self) -> None:
        """Release the fake decoder."""


class FakeCV2:
    """Small subset of OpenCV consumed by ``FileFrameSource``."""

    CAP_PROP_FPS = 5
    CAP_PROP_POS_FRAMES = 1

    def __init__(self, capture: FakeVideoCapture) -> None:
        """Store the capture returned by ``VideoCapture``."""
        self._capture = capture

    def VideoCapture(self, _: str) -> FakeVideoCapture:
        """Return the configured fake video decoder."""
        return self._capture


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


def test_file_source_can_pace_video_by_declared_source_fps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Source-FPS playback must delay later frames without slowing the first."""
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"video")
    capture = FakeVideoCapture([{"frame": 1}, {"frame": 2}], fps=50.0)
    fake_cv2 = FakeCV2(capture)
    monkeypatch.setattr(
        file_source_module,
        "_load_opencv",
        lambda: fake_cv2,
    )
    source = FileFrameSource(video_path, playback=PlaybackMode.SOURCE_FPS)
    source.open()

    assert source.read() == {"frame": 1}
    started_at = time.monotonic()
    assert source.read() == {"frame": 2}

    assert time.monotonic() - started_at >= 0.01
    source.close()


def test_stop_from_event_handler_requests_shutdown_without_joining_itself() -> None:
    """A synchronous pipeline callback may safely request a terminal stop."""
    pipeline = VisionPipeline(
        SyntheticFrameSource(max_frames=100),
        NoopFrameProcessor(),
    )

    def stop_after_result(event: Any) -> None:
        """Stop from the processing worker once its first result is available."""
        if event.type is PipelineEventType.RESULT_AVAILABLE:
            pipeline.stop()

    pipeline.subscribe(stop_after_result)
    pipeline.start()

    assert pipeline.wait_until_stopped(timeout=2.0)
    assert pipeline.state.value == "stopped"


def test_close_error_is_recorded_without_preventing_pipeline_shutdown() -> None:
    """Custom source cleanup errors must not strand the lifecycle in stopping."""
    pipeline = VisionPipeline(FailingCloseSource(), NoopFrameProcessor())

    pipeline.start()

    assert pipeline.wait_until_stopped(timeout=2.0)
    assert pipeline.stats.source_errors == 1
    assert pipeline.last_error is not None
    assert "source close failed" in str(pipeline.last_error)


def test_managed_processor_lifecycle_is_optional_and_automatic() -> None:
    """A model-backed processor must be opened and closed once per lifecycle."""
    processor = ManagedProcessor()
    pipeline = VisionPipeline(SyntheticFrameSource(max_frames=1), processor)

    pipeline.start()

    assert pipeline.wait_until_stopped(timeout=2.0)
    assert processor.opened == 1
    assert processor.closed == 1


def test_camera_frame_source_reads_raw_payload_without_jpeg_round_trip() -> None:
    """Camera adapter must expose the raw frame object and manage camera lifecycle."""
    camera = FakeRawCamera()
    source = CameraFrameSource(camera)
    source.open()

    frame = source.read()

    assert frame is camera.frame
    assert camera.start_calls == 1
    source.close()
    assert camera.stop_calls == 1
