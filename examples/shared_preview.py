"""Serve a processed image source without creating a second camera pipeline."""

from imx_camera_toolkit import Camera, CameraConfig
from imx_camera_toolkit.preview import serve


def main() -> None:
    """Start a camera once and expose its newest raw image through HTTP."""
    with Camera(CameraConfig(enable_preview=False)) as camera:
        serve(camera, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
