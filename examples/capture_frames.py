"""Read newest raw frames for an external image-processing pipeline."""

from imx_camera_toolkit import Camera, CameraConfig


def main() -> None:
    """Capture raw frames without paying for browser-preview JPEG encoding."""
    config = CameraConfig(enable_preview=False)

    with Camera(config) as camera:
        while True:
            frame = camera.read(timeout=1.0, copy=False)

            if frame is None:
                continue

            print(f"sequence={frame.sequence} format={frame.format}")


if __name__ == "__main__":
    main()
