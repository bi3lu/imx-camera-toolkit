"""OpenCV GStreamer fallback backend."""

from __future__ import annotations

import importlib
from typing import Any

from ..errors import CameraDependencyError, CameraOpenError
from .base import CaptureBackend

try:
    cv2_module: Any | None = importlib.import_module("cv2")

except ImportError:
    cv2_module = None


class OpenCVCaptureBackend(CaptureBackend):
    """Capture Argus frames through OpenCV's GStreamer integration."""

    def __init__(self, pipeline: str) -> None:
        """Initialize a backend without opening the pipeline."""
        self._pipeline = pipeline
        self._capture: Any | None = None

    def open(self) -> None:
        """Open the configured GStreamer pipeline with OpenCV."""
        if cv2_module is None:
            raise CameraDependencyError(
                "System OpenCV with GStreamer support is required."
            )

        capture = cv2_module.VideoCapture(
            self._pipeline,
            cv2_module.CAP_GSTREAMER,
        )
        if not capture.isOpened():
            capture.release()

            raise CameraOpenError(
                "Could not open the IMX camera. Check CSI connection, sensor-id, "
                "and that nvarguscamerasrc is available."
            )

        self._capture = capture

    def read(self) -> tuple[bool, Any | None]:
        """Read the next BGR frame through OpenCV."""
        if self._capture is None:
            return False, None

        success, frame = self._capture.read()
        return bool(success), frame

    def close(self) -> None:
        """Release the OpenCV capture handle."""
        if self._capture is not None:
            self._capture.release()

        self._capture = None
