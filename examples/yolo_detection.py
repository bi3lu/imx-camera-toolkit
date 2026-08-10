"""Run an application-owned end-to-end YOLO decoder with WebRTC preview."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn

from imx_camera_toolkit import (
    CameraConfig,
    GpuCamera,
    ShapeProfile,
    TensorRTRunner,
    VideoEncoderConfig,
)
from imx_camera_toolkit.consumers import InferenceConsumer
from imx_camera_toolkit.inference import InferenceResult
from imx_camera_toolkit.production_preview import (
    CudaOverlayRenderer,
    OverlayRectangle,
    ProductionPreviewConfig,
    ProductionPreviewServer,
    create_production_preview_app,
)


def _rows(result: InferenceResult, output_name: str) -> list[list[float]]:
    """Decode one common end-to-end YOLO Nx6 export outside the toolkit."""
    output = next(item for item in result.outputs if item.name == output_name)
    tolist = getattr(output.data, "tolist", None)

    if not callable(tolist):
        raise TypeError("YOLO output must provide ndarray-compatible tolist()")

    values: Any = tolist()

    if (
        values
        and isinstance(values[0], list)
        and values[0]
        and isinstance(values[0][0], list)
    ):
        values = values[0]

    return [[float(value) for value in row] for row in values]


def main() -> None:
    """Serve H.264 WebRTC with rectangles decoded from a custom YOLO export."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", default="boxes")
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    runner = TensorRTRunner(
        args.model,
        cache_dir=Path(".cache/tensorrt"),
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
        ),
        video_config=VideoEncoderConfig(),
        experimental=True,
    )
    inference = InferenceConsumer(
        camera.subscribe_latest("yolo"),
        runner,
    )

    def rectangles(result: InferenceResult) -> tuple[OverlayRectangle, ...]:
        """Map the application's chosen YOLO schema to toolkit overlays."""
        overlays: list[OverlayRectangle] = []

        for row in _rows(result, args.output):
            if len(row) < 6 or row[4] < args.score:
                continue

            left, top, right, bottom = (max(round(value), 0) for value in row[:4])

            if right > left and bottom > top:
                overlays.append(OverlayRectangle(left, top, right - left, bottom - top))
        return tuple(overlays)

    overlay = CudaOverlayRenderer(inference, mapper=rectangles)
    camera.set_video_overlay(overlay)
    transport = ProductionPreviewServer(camera, ProductionPreviewConfig())
    app = create_production_preview_app(transport)

    try:
        with camera, inference:
            uvicorn.run(app, host="127.0.0.1", port=args.port)

    finally:
        overlay.close()


if __name__ == "__main__":
    main()
