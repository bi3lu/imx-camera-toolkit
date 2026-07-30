"""OpenCV GStreamer fallback backend."""

from __future__ import annotations

from typing import Any

from .base import CaptureBackend

try:
    import cv2

except ImportError:
    cv2: Any | None = None


class OpenCVCaptureBackend(CaptureBackend):
    """Capture Argus frames through OpenCV's GStreamer integration."""

    def __init__(self, pipeline: str) -> None:
        """Initialize a backend without opening the pipeline."""
        self._pipeline = pipeline
        self._capture: Any | None = None

    def open(self) -> None:
        """Open the configured GStreamer pipeline with OpenCV."""
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not available. Use the JetPack-provided Python/OpenCV "
                "environment with GStreamer support."
            )

        capture = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not capture.isOpened():
            capture.release()

            raise RuntimeError(
                "Could not open the IMX camera. Check CSI connection, sensor-id, "
                "and that nvarguscamerasrc is available."
            )

        self._capture = capture

    def read(self) -> tuple[bool, Any | None]:
        """Read the next BGR frame through OpenCV."""
        if self._capture is None:
            return False, None

        return self._capture.read()

    def close(self) -> None:
        """Release the OpenCV capture handle."""
        if self._capture is not None:
            self._capture.release()

        self._capture = None
