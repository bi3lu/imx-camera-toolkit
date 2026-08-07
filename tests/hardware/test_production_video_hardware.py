"""Opt-in Jetson validation for concurrent TensorRT and 720p production encode."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from imx_camera_toolkit import (
    CameraConfig,
    GpuCamera,
    HardwareVideoConfig,
    VideoCodec,
)
from imx_camera_toolkit.consumers import InferenceConsumer
from imx_camera_toolkit.inference import ShapeProfile, TensorRTRunner

pytestmark = [pytest.mark.hardware, pytest.mark.benchmark]


def test_720p_hardware_preview_remains_lightweight_during_tensorrt(
    tmp_path: Path,
) -> None:
    """Hardware encode must sustain preview without consuming most of one CPU."""
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
    camera = GpuCamera(
        CameraConfig(
            capture_width=1280,
            capture_height=720,
            output_width=1280,
            output_height=720,
            fps=30,
            enable_preview=False,
        ),
        video_config=HardwareVideoConfig(
            codec=VideoCodec.H264,
            bitrate_bps=4_000_000,
            keyframe_interval=30,
        ),
    )

    with camera:
        inference = InferenceConsumer(
            camera.subscribe_latest("benchmark-inference"),
            runner,
        )
        with inference:
            started_wall = time.monotonic()
            started_cpu = time.process_time()
            time.sleep(float(os.getenv("IMX_PREVIEW_BENCHMARK_SECONDS", "5")))
            elapsed_cpu = time.process_time() - started_cpu
            elapsed_wall = time.monotonic() - started_wall
            video = camera.video_stats

    assert video.encode_fps >= 25.0
    assert inference.processed_frames > 0
    assert elapsed_cpu / elapsed_wall < 0.5
