"""Build and run a model-neutral custom TensorRT engine from ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from imx_camera_toolkit import GpuCamera, ShapeProfile, TensorRTRunner
from imx_camera_toolkit.consumers import InferenceConsumer


def main() -> None:
    """Report every named output without assuming detection or segmentation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    runner = TensorRTRunner(
        args.model,
        cache_dir=Path(".cache/custom-engine"),
        input_name=args.input_name,
        precision=args.precision,
        shape_profile=ShapeProfile(
            minimum=(1, 3, 320, 320),
            optimum=(1, 3, 640, 640),
            maximum=(1, 3, 1920, 1920),
        ),
        inference_shape=(1, 3, 640, 640),
    )
    camera = GpuCamera(experimental=True)
    inference = InferenceConsumer(camera.subscribe_latest("custom-engine"), runner)
    results = inference.subscribe_results("output-printer")

    with camera, inference:
        for _ in range(args.frames):
            result = results.receive(timeout=2.0)

            if result is None:
                continue

            outputs = ", ".join(
                f"{output.name}:{output.shape}/{output.dtype}"
                for output in result.outputs
            )
            print(
                f"frame={result.frame_sequence} "
                f"inference_ms={result.inference_time_ns / 1_000_000:.2f} "
                f"outputs=[{outputs}]"
            )


if __name__ == "__main__":
    main()
