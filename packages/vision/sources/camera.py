"""Adapter exposing the camera package's raw BGR frames to Vision Pipeline."""

from __future__ import annotations

from .base import RawFrameCamera


class CameraFrameSource:
    """Adapt a raw-frame camera to the generic :class:`FrameSource` contract.

    The adapter consumes processed BGR frames directly from ``Camera``. It does
    not read the JPEG preview stream and therefore avoids a BGR-to-JPEG-to-BGR
    conversion cycle.

    Args:
        camera: Camera exposing the :class:`RawFrameCamera` protocol.
        timeout: Maximum wait for a newer camera frame in seconds.
        manage_lifecycle: Whether ``open()`` starts and ``close()`` stops the
            supplied camera. Set this to ``False`` when another component owns
            a shared camera lifecycle, such as a preview API.
    """

    def __init__(
        self,
        camera: RawFrameCamera,
        *,
        timeout: float = 0.2,
        manage_lifecycle: bool = True,
    ) -> None:
        """Store the camera adapter configuration without starting capture."""
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a positive number")
        if timeout <= 0:
            raise ValueError("timeout must be a positive number")

        self._camera = camera
        self._timeout = timeout
        self._manage_lifecycle = manage_lifecycle
        self._opened = False
        self._exhausted = False
        self._previous_frame_number = -1

    @property
    def exhausted(self) -> bool:
        """bool: Whether the camera stopped and no more frames can arrive."""
        return self._exhausted

    def open(self) -> None:
        """Start the camera when owned and prepare raw-frame consumption."""
        if self._manage_lifecycle:
            self._camera.start()
        elif not self._camera.running:
            raise RuntimeError("shared camera must be running before vision starts")

        self._opened = True
        self._exhausted = False
        self._previous_frame_number = -1

    def read(self) -> object | None:
        """Wait for and return one newer raw BGR frame without decoding JPEG."""
        if not self._opened:
            raise RuntimeError("camera frame source is not open")

        frame_number, frame = self._camera.wait_for_raw_frame(
            self._previous_frame_number,
            timeout=self._timeout,
        )
        if frame is not None and frame_number != self._previous_frame_number:
            self._previous_frame_number = frame_number
            return frame

        if not self._camera.running:
            self._exhausted = True
        return None

    def close(self) -> None:
        """Stop an owned camera or detach from a shared running camera."""
        if self._manage_lifecycle:
            self._camera.stop()
        self._opened = False
        self._exhausted = True
