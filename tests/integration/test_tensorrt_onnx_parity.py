"""Opt-in TensorRT/ONNX Runtime parity on one physical NVMM frame."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from imx_camera_toolkit import FrameFormat, MemoryType
from imx_camera_toolkit._internal.camera.backends import GpuGStreamerCaptureBackend
from imx_camera_toolkit._internal.camera.pipeline import build_gpu_gstreamer_pipeline
from imx_camera_toolkit._internal.inference.interop import NativeCudaInterop
from imx_camera_toolkit.inference import FrameSpec, ShapeProfile, TensorRTRunner

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
TensorProto = onnx.TensorProto
helper = onnx.helper

pytestmark = [pytest.mark.integration, pytest.mark.hardware]

if os.environ.get("IMX_TENSORRT_INTEGRATION") != "1":
    pytest.skip(
        "set IMX_TENSORRT_INTEGRATION=1 on a JetPack camera host",
        allow_module_level=True,
    )


def _write_dynamic_box_model(path: Path) -> None:
    """Create a tiny model whose box coordinates depend on input pixels."""
    image = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 3, "height", "width"],
    )
    boxes = helper.make_tensor_value_info(
        "boxes",
        TensorProto.FLOAT,
        [1, 1, 4],
    )
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [1, 1], [0.0])
    output_shape = helper.make_tensor(
        "output_shape",
        TensorProto.INT64,
        [3],
        [1, 1, 4],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceMean",
                ["input"],
                ["channel_means"],
                axes=[2, 3],
                keepdims=0,
            ),
            helper.make_node(
                "Concat",
                ["channel_means", "zero"],
                ["flat_boxes"],
                axis=1,
            ),
            helper.make_node(
                "Reshape",
                ["flat_boxes", "output_shape"],
                ["boxes"],
            ),
        ],
        "dynamic-box-reference",
        [image],
        [boxes],
        [zero, output_shape],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.save(model, path)


def test_tensorrt_and_onnx_runtime_boxes_match_for_one_nvmm_frame(
    tmp_path: Path,
) -> None:
    """Compare generic boxes after identical NV12 CUDA preprocessing."""
    model_path = tmp_path / "dynamic_boxes.onnx"
    _write_dynamic_box_model(model_path)
    pipeline = build_gpu_gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        output_width=1280,
        output_height=720,
        framerate=30,
    )
    backend = GpuGStreamerCaptureBackend(
        pipeline,
        1280,
        720,
        enable_preview=False,
    )
    interop = NativeCudaInterop()
    shape_profile = ShapeProfile(
        minimum=(1, 3, 32, 32),
        optimum=(1, 3, 64, 64),
        maximum=(1, 3, 96, 96),
    )
    runner = TensorRTRunner(
        model_path,
        cache_dir=tmp_path / "engines",
        shape_profile=shape_profile,
        inference_shape=(1, 3, 64, 64),
        interop=interop,
    )
    cached_runner: TensorRTRunner | None = None

    backend.open()

    try:
        success, frame = backend.read()
        assert success and frame is not None

        stream = interop.create_stream()
        reference_input = interop.allocate(1 * 3 * 64 * 64 * 4)
        surface = interop.import_frame(frame)
        interop.preprocess_nv12(
            surface,
            reference_input,
            width=64,
            height=64,
            channel_order="RGB",
            scale=1.0 / 255.0,
            mean=(0.0, 0.0, 0.0),
            standard_deviation=(1.0, 1.0, 1.0),
            stream=stream,
        )
        host_input = np.frombuffer(
            reference_input.copy_to_host(stream),
            dtype=np.float32,
        ).reshape(1, 3, 64, 64)

        del surface

        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        expected_boxes = session.run(["boxes"], {"input": host_input})[0]

        runner.prepare(
            FrameSpec(
                width=1280,
                height=720,
                format=FrameFormat.NV12_NVMM,
                memory_type=MemoryType.NVMM,
            )
        )
        result = runner.infer(frame)
        actual_boxes = np.asarray(
            next(output.data for output in result.outputs if output.name == "boxes")
        )

        np.testing.assert_allclose(actual_boxes, expected_boxes, rtol=1e-3, atol=1e-3)
        assert result.inference_time_ns > 0
        assert result.metadata["cuda_stream"] != 0
        preprocessing = result.metadata["preprocessing"]
        assert preprocessing == {
            "source": "NV12_NVMM",
            "layout": "NCHW",
            "channel_order": "RGB",
            "scale": 1.0 / 255.0,
            "padding_value": [114.0, 114.0, 114.0],
            "transform": {
                "resize_mode": "stretch",
                "scale": [0.05, 64.0 / 720.0],
                "pad_x": 0,
                "pad_y": 0,
                "source_shape": [720, 1280],
                "model_shape": [64, 64],
            },
        }

        runner.close()
        cached_runner = TensorRTRunner(
            model_path,
            cache_dir=tmp_path / "engines",
            shape_profile=shape_profile,
            inference_shape=(1, 3, 64, 64),
            interop=interop,
        )
        cached_runner.prepare(FrameSpec.from_gpu_frame(frame))
        assert cached_runner.cache_hit is True

    finally:
        if cached_runner is not None:
            cached_runner.close()
        runner.close()
        backend.close()
