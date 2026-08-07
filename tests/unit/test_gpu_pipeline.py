"""Contract tests for the NVMM capture pipeline and GPU camera lifecycle."""

from __future__ import annotations

import inspect

import pytest

import packages.camera.backends.gpu_gstreamer as gpu_gstreamer_backend
from imx_camera_toolkit import (
    CameraConfig,
    CameraOpenError,
    FrameFormat,
    GpuCamera,
    MemoryType,
    build_gpu_gstreamer_pipeline,
)
from imx_camera_toolkit.frames import GpuFrameSource
from imx_camera_toolkit.testing import mock_gpu_frame
from packages.camera.backends.gpu_gstreamer import GpuGStreamerCaptureBackend


@pytest.mark.parametrize(
    ("sensor", "sensor_id", "width", "height"),
    [
        ("IMX219", 0, 1280, 720),
        ("IMX219", 0, 1920, 1080),
        ("IMX477", 1, 1280, 720),
        ("IMX477", 1, 1920, 1080),
    ],
)
def test_gpu_pipeline_matrix_retains_nv12_nvmm_to_appsink(
    sensor: str,
    sensor_id: int,
    width: int,
    height: int,
) -> None:
    """Target sensor scenarios must build the same bounded NVMM contract."""
    pipeline = build_gpu_gstreamer_pipeline(
        sensor_id=sensor_id,
        capture_width=width,
        capture_height=height,
        output_width=width,
        output_height=height,
        framerate=30,
        enable_preview=True,
    )

    assert sensor in {"IMX219", "IMX477"}
    assert "tee name=camera_tee" in pipeline
    assert pipeline.count("video/x-raw(memory:NVMM)") >= 4
    assert "video/x-raw, " not in pipeline
    assert "videoconvert" not in pipeline
    assert "format=(string)NV12" in pipeline
    assert f"width=(int){width}" in pipeline
    assert f"height=(int){height}" in pipeline
    assert "framerate=(fraction)30/1" in pipeline


def test_gpu_pipeline_isolates_both_latest_frame_branches() -> None:
    """Inference and preview must each have a one-buffer leaky boundary."""
    pipeline = build_gpu_gstreamer_pipeline(enable_preview=True)
    queue_policy = (
        "max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
    )
    sink_policy = (
        "max-buffers=1 drop=true sync=false enable-last-sample=false "
        "wait-on-eos=false"
    )

    assert "queue name=gpu_queue" in pipeline
    assert "queue name=preview_queue" in pipeline
    assert pipeline.count(queue_policy) == 2
    assert "appsink name=gpu_sink" in pipeline
    assert "nvjpegenc quality=65" in pipeline
    assert "appsink name=preview_sink" in pipeline
    assert pipeline.count(sink_policy) == 2


def test_gpu_backend_never_maps_or_imports_numpy_for_inference() -> None:
    """The NVMM backend must forward Gst.Buffer rather than host pixels."""
    module_source = inspect.getsource(gpu_gstreamer_backend)
    read_source = inspect.getsource(GpuGStreamerCaptureBackend.read)

    assert "numpy" not in module_source
    assert ".map(" not in read_source
    assert ".copy(" not in read_source


def test_gpu_camera_selects_nvmm_without_changing_cpu_config_input() -> None:
    """GpuCamera must be explicit while accepting existing hardware settings."""
    cpu_config = CameraConfig(output_width=1920, output_height=1080)
    camera = GpuCamera(cpu_config, enable_preview=True)

    assert cpu_config.output_format is FrameFormat.BGR_CPU
    assert camera.config.output_format is FrameFormat.NV12_NVMM
    assert camera.config.output_memory is MemoryType.NVMM
    assert camera.config.copies_to_host_memory is False
    assert isinstance(camera, GpuFrameSource)
    assert camera.frame_resolution == (1920, 1080)
    assert "appsink name=gpu_sink" in camera.pipeline
    assert "appsink name=preview_sink" in camera.pipeline


class _FakeGpuBackend:
    """Record whole-pipeline open and close operations."""

    backend_name = "fake-nvmm"

    def __init__(self, *, open_error: Exception | None = None) -> None:
        """Initialize a backend with an optional open failure."""
        self.open_error = open_error
        self.opened = False
        self.closed = False

    def open(self) -> None:
        """Open both conceptual branches or raise."""
        if self.open_error is not None:
            raise self.open_error
        self.opened = True

    def read(self) -> tuple[bool, None]:
        """Produce no frame in lifecycle-only tests."""
        return False, None

    def read_preview(self) -> bytes | None:
        """Produce no preview in lifecycle-only tests."""
        return None

    def read_video(self) -> None:
        """Produce no encoded video in lifecycle-only tests."""
        return None

    def close(self) -> None:
        """Record closure of the shared tee pipeline."""
        self.closed = True


class _RecoverableGpuCamera(GpuCamera):
    """GpuCamera using deterministic replacement backends."""

    def __init__(self, backends: list[_FakeGpuBackend]) -> None:
        """Store backend instances in creation order."""
        self.backends = backends
        super().__init__()

    def _backend_available(self) -> bool:
        """Allow lifecycle tests on hosts without GStreamer."""
        return True

    def _create_backend(self) -> _FakeGpuBackend:
        """Return the next complete fake pipeline."""
        return self.backends.pop(0)


def test_gpu_recovery_replaces_and_closes_the_whole_tee_pipeline() -> None:
    """Recovery must never restart only one of the two branches."""
    failed = _FakeGpuBackend()
    replacement = _FakeGpuBackend()
    camera = _RecoverableGpuCamera([replacement])
    camera._backend = failed
    camera._running.set()

    assert camera._recover_backend() is True
    assert failed.closed is True
    assert replacement.opened is True
    assert camera.active_backend == "fake-nvmm"

    camera.stop()
    assert replacement.closed is True


def test_gpu_start_closes_a_pipeline_that_fails_to_open() -> None:
    """A partial open must not leave either tee branch allocated."""
    failed = _FakeGpuBackend(open_error=CameraOpenError("open failed"))
    camera = _RecoverableGpuCamera([failed])

    with pytest.raises(CameraOpenError, match="open failed"):
        camera.start()

    assert failed.closed is True
    assert camera.running is False


def test_gpu_publication_invalidates_the_previous_nvmm_lease_on_stop() -> None:
    """Shutdown must release the last buffer back to the GStreamer pool."""
    camera = GpuCamera()
    frame = mock_gpu_frame(object())
    camera._gpu_publisher.publish(frame)
    camera._running.set()

    assert camera.read(timeout=0) is frame
    camera.stop()

    assert frame.valid is False
    assert camera.latest_frame() is None
