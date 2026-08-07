"""Consume a segmentation tensor without coupling capture to its schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from imx_camera_toolkit import GpuCamera, ShapeProfile, TensorRTRunner
from imx_camera_toolkit.consumers import InferenceConsumer


def main() -> None:
    """Print model-owned mask tensor metadata from newest-frame inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--mask-output", default="masks")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    runner = TensorRTRunner(
        args.model,
        cache_dir=Path(".cache/tensorrt-segmentation"),
        shape_profile=ShapeProfile(
            minimum=(1, 3, 320, 320),
            optimum=(1, 3, 640, 640),
            maximum=(1, 3, 1280, 1280),
        ),
        inference_shape=(1, 3, 640, 640),
    )
    camera = GpuCamera(experimental=True)
    inference = InferenceConsumer(
        camera.subscribe_latest("segmentation"),
        runner,
    )
    results = inference.subscribe_results("mask-consumer")

    with camera, inference:
        for _ in range(args.frames):
            result = results.receive(timeout=2.0)

            if result is None:
                continue

            mask = next(
                output
                for output in result.outputs
                if output.name == args.mask_output
            )
            print(
                f"frame={result.frame_sequence} mask={mask.name} "
                f"shape={mask.shape} dtype={mask.dtype}"
            )


if __name__ == "__main__":
    main()
