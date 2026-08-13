"""Opt-in Jetson validation for concurrent TensorRT and 720p production encode."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from imx_camera_toolkit import (
    CameraConfig,
    FrameFormat,
    GpuCamera,
    MemoryType,
    VideoCodec,
    VideoEncoderBackend,
    VideoEncoderConfig,
)
from imx_camera_toolkit.consumers import InferenceConsumer
from imx_camera_toolkit.inference import FrameSpec, ShapeProfile, TensorRTRunner
from imx_camera_toolkit.production_preview import (
    CudaOverlayRenderer,
    OverlayRectangle,
)

pytestmark = [pytest.mark.hardware, pytest.mark.benchmark]


def test_720p_production_preview_runs_during_tensorrt(
    tmp_path: Path,
) -> None:
    """Inference and overlay threads must share the retained primary context."""
    if os.getenv("IMX_PRODUCTION_PREVIEW_HARDWARE") != "1":
        pytest.skip("set IMX_PRODUCTION_PREVIEW_HARDWARE=1 on the target Jetson")

    model_value = os.getenv("IMX_TENSORRT_ONNX")

    if not model_value:
        pytest.skip("set IMX_TENSORRT_ONNX to a compatible image model")

    model_path = Path(model_value)

    if not model_path.is_file():
        pytest.skip(f"TensorRT validation model is missing: {model_path}")

    runner = TensorRTRunner(
        model_path,
        cache_dir=tmp_path / "engines",
        shape_profile=ShapeProfile(
            minimum=(1, 3, 320, 320),
            optimum=(1, 3, 640, 640),
            maximum=(1, 3, 1280, 1280),
        ),
        inference_shape=(1, 3, 640, 640),
    )
    requested_backend = VideoEncoderBackend(
        os.getenv("IMX_VIDEO_ENCODER_BACKEND", "auto")
    )
    camera = GpuCamera(
        CameraConfig(
            capture_width=1280,
            capture_height=720,
            output_width=1280,
            output_height=720,
            fps=30,
            enable_preview=False,
        ),
        video_config=VideoEncoderConfig(
            codec=VideoCodec.H264,
            backend=requested_backend,
            bitrate_bps=4_000_000,
            keyframe_interval=30,
        ),
    )
    frame_spec = FrameSpec(
        width=1280,
        height=720,
        format=FrameFormat.NV12_NVMM,
        memory_type=MemoryType.NVMM,
    )
    overlay: CudaOverlayRenderer | None = None

    try:
        runner.prepare(frame_spec)
        inference = InferenceConsumer(
            camera.subscribe_latest("cross-thread-context"),
            runner,
            prepared_spec=frame_spec,
        )
        overlay = CudaOverlayRenderer(
            inference,
            mapper=lambda result: (OverlayRectangle(16, 16, 96, 64),),
        )
        camera.set_video_overlay(overlay)

        with camera, inference:
            started_wall = time.monotonic()
            started_cpu = time.process_time()
            deadline = started_wall + float(
                os.getenv("IMX_PREVIEW_BENCHMARK_SECONDS", "5")
            )

            while (
                inference.processed_frames < 10 or overlay.rendered_frames == 0
            ) and time.monotonic() < deadline:
                time.sleep(0.05)

            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds > 0:
                time.sleep(remaining_seconds)

            elapsed_cpu = time.process_time() - started_cpu
            elapsed_wall = time.monotonic() - started_wall
            video = camera.video_stats
            selected_backend = camera.video_encoder_backend

    finally:
        if overlay is not None:
            overlay.close()

        runner.close()

    assert overlay is not None
    assert video.encode_fps >= 25.0
    assert inference.processed_frames >= 10
    assert inference.failed_frames == 0
    assert inference.last_error is None
    assert overlay.rendered_frames > 0
    assert overlay.failed_frames == 0
    assert overlay.last_error is None

    if selected_backend == VideoEncoderBackend.NVENC.value:
        assert elapsed_cpu / elapsed_wall < 0.5

    else:
        assert selected_backend == VideoEncoderBackend.X264.value
