"""Run one stable BGR camera for a raw consumer and MJPEG preview."""

from __future__ import annotations

import uvicorn

from imx_camera_toolkit import Camera, CameraConfig, FrameConsumer, create_preview_app
from imx_camera_toolkit.camera import Frame


def process_raw_frame(frame: Frame) -> None:
    """Replace this body with an application-owned CPU model or processor."""
    print(f"processed frame={frame.sequence} size={frame.width}x{frame.height}")


def main() -> None:
    """Keep raw processing independent from browser delivery throughput."""
    camera = Camera(CameraConfig(enable_preview=False))
    processor = FrameConsumer(
        camera.subscribe_latest("raw-processor"),
        process_raw_frame,
    )
    app = create_preview_app(camera)

    with camera, processor:
        uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
