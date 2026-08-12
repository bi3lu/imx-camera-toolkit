"""Start the one-line browser preview facade."""

from imx_camera_toolkit import preview


def main() -> None:
    """Start the stable GPU preview with the documented camera settings."""
    preview(
        sensor_id=0,
        width=1280,
        height=720,
        fps=30,
        backend="gpu",
        port=8000,
    )


if __name__ == "__main__":
    main()
