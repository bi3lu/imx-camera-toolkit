"""Capture JPEG frames from IMX CSI cameras on NVIDIA Jetson devices."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

try:
    import cv2  # OpenCV is supplied by JetPack 6.2.2.

except ImportError:
    cv2: Any | None = None


logger = logging.getLogger(__name__)


def build_gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    output_width: int = 640,
    output_height: int = 360,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    """Build an OpenCV-compatible Argus pipeline for a CSI camera.

    Args:
        sensor_id: Zero-based CSI sensor identifier used by Argus.
        capture_width: Width captured directly from the sensor, in pixels.
        capture_height: Height captured directly from the sensor, in pixels.
        output_width: Width of frames delivered to OpenCV, in pixels.
        output_height: Height of frames delivered to OpenCV, in pixels.
        framerate: Camera capture rate, in frames per second.
        flip_method: NVIDIA ``nvvidconv`` flip transformation, from 0 to 7.

    Returns:
        A GStreamer pipeline string suitable for ``cv2.VideoCapture``.

    Raises:
        ValueError: If an identifier, dimension, frame rate, or flip method is
            outside its supported range.
    """
    if sensor_id < 0:
        raise ValueError("sensor_id must be greater than or equal to zero")
    if min(capture_width, capture_height, output_width, output_height, framerate) <= 0:
        raise ValueError("frame dimensions and framerate must be greater than zero")
    if not 0 <= flip_method <= 7:
        raise ValueError("flip_method must be between 0 and 7")

    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, "
        f"height=(int){capture_height}, "
        "format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, "
        f"width=(int){output_width}, "
        f"height=(int){output_height}, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink max-buffers=1 drop=true sync=false"
    )


class Camera:
    """Capture and JPEG-encode the latest image from one CSI sensor.

    Only the most recent encoded frame is retained in memory. This makes the
    class appropriate for live previews and streaming, where stale frames are
    less useful than current ones.

    Args:
        quality: JPEG quality from 0 to 100.
        max_fps: Maximum JPEG encoding rate in frames per second.
        sensor_id: Zero-based CSI sensor identifier used by Argus.
        capture_width: Width captured directly from the sensor, in pixels.
        capture_height: Height captured directly from the sensor, in pixels.
        output_width: Width of frames delivered to OpenCV, in pixels.
        output_height: Height of frames delivered to OpenCV, in pixels.
        capture_fps: Camera capture rate, in frames per second.
        flip_method: NVIDIA ``nvvidconv`` flip transformation, from 0 to 7.

    Attributes:
        frames_captured: Number of frames read from the camera.
        frames_encoded: Number of frames successfully JPEG-encoded.
        last_frame_time: Unix timestamp of the latest encoded frame, if any.
        last_error: Most recent background capture exception, if any.
    """

    def __init__(
        self,
        quality: int = 65,
        max_fps: float = 30.0,
        *,
        sensor_id: int = 0,
        capture_width: int = 1280,
        capture_height: int = 720,
        output_width: int = 640,
        output_height: int = 360,
        capture_fps: int = 30,
        flip_method: int = 0,
    ) -> None:
        """Initialize a camera without opening its capture device.

        Raises:
            ValueError: If JPEG quality or encoding rate is invalid, or if a
                pipeline configuration argument is invalid.
        """
        if not 0 <= quality <= 100:
            raise ValueError("quality must be between 0 and 100")

        if max_fps <= 0:
            raise ValueError("max_fps must be greater than zero")

        self._pipeline = build_gstreamer_pipeline(
            sensor_id=sensor_id,
            capture_width=capture_width,
            capture_height=capture_height,
            output_width=output_width,
            output_height=output_height,
            framerate=capture_fps,
            flip_method=flip_method,
        )
        self._jpeg_quality = quality
        self._jpeg_interval = 1.0 / max_fps

        self._capture: Any | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_number = 0

        self.frames_captured = 0
        self.frames_encoded = 0
        self.last_frame_time: float | None = None
        self.last_error: Exception | None = None

    @property
    def pipeline(self) -> str:
        """str: GStreamer pipeline used when :meth:`start` is called."""
        return self._pipeline

    @property
    def running(self) -> bool:
        """bool: Whether the background capture loop is active."""
        return self._running.is_set()

    @property
    def frame_available(self) -> bool:
        """bool: Whether at least one JPEG frame is currently available."""
        with self._condition:
            return self._jpeg is not None

    @property
    def frame_number(self) -> int:
        """int: Monotonically increasing identifier of the latest JPEG frame."""
        with self._condition:
            return self._frame_number

    @property
    def jpeg(self) -> bytes | None:
        """bytes | None: Latest JPEG frame, or ``None`` when unavailable."""
        with self._condition:
            return self._jpeg

    def start(self) -> None:
        """Open the camera and start the background capture thread.

        Calling this method while capture is already active has no effect.

        Raises:
            RuntimeError: If OpenCV is unavailable, the previous capture thread
                has not stopped, or the Argus camera cannot be opened.
        """
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is not available. Use the JetPack-provided Python/OpenCV "
                "environment with GStreamer support."
            )

        with self._lifecycle_lock:
            if self.running:
                return

            self._release_finished_capture()
            capture = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)

            if not capture.isOpened():
                capture.release()
                raise RuntimeError(
                    "Could not open the IMX camera. Check CSI connection, sensor-id, "
                    "and that nvarguscamerasrc is available."
                )

            self._capture = capture
            self.last_error = None
            self._running.set()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="imx-camera-capture",
                daemon=True,
            )
            self._thread.start()

    def _release_finished_capture(self) -> None:
        """Release resources retained after an unexpectedly ended capture loop.

        Raises:
            RuntimeError: If the previous capture thread is still alive.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("camera capture thread is still stopping")

        if self._capture is not None:
            self._capture.release()

        self._thread = None
        self._capture = None

    def _capture_loop(self) -> None:
        """Read frames, rate-limit JPEG encoding, and publish the latest frame."""
        last_encode_time = 0.0

        try:
            while self.running:
                capture = self._capture

                if capture is None:
                    break

                success, frame = capture.read()

                if not success:
                    time.sleep(0.02)
                    continue

                self.frames_captured += 1
                now = time.monotonic()

                if now - last_encode_time < self._jpeg_interval:
                    continue

                last_encode_time = now

                success, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
                )

                if not success:
                    continue

                with self._condition:
                    self._jpeg = encoded.tobytes()
                    self._frame_number += 1
                    self._condition.notify_all()

                self.frames_encoded += 1
                self.last_frame_time = time.time()

        except Exception as error:
            self.last_error = error
            logger.exception("IMX camera capture failed")

        finally:
            self._running.clear()
            with self._condition:
                self._condition.notify_all()

    def wait_for_jpeg(
        self, previous_frame_number: int, timeout: float = 2.0
    ) -> tuple[int, bytes | None]:
        """Wait for a JPEG frame newer than a known frame number.

        Args:
            previous_frame_number: Frame number already consumed by the caller.
            timeout: Maximum time to wait, in seconds.

        Returns:
            A pair containing the latest frame number and JPEG bytes. The bytes
            are ``None`` until the first frame is encoded.

        Raises:
            ValueError: If ``timeout`` is negative.
        """
        if timeout < 0:
            raise ValueError("timeout must be greater than or equal to zero")

        with self._condition:
            self._condition.wait_for(
                lambda: self._frame_number != previous_frame_number or not self.running,
                timeout=timeout,
            )
            return self._frame_number, self._jpeg

    def stop(self) -> None:
        """Stop capture, release the camera handle, and discard the last frame.

        If the capture thread does not end within three seconds, the camera
        handle remains open and a warning is logged.
        """
        with self._lifecycle_lock:
            self._running.clear()

            with self._condition:
                self._condition.notify_all()

            thread = self._thread

            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.0)

                if thread.is_alive():
                    logger.warning("IMX camera thread did not stop within 3 seconds")
                    return

            if self._capture is not None:
                self._capture.release()

            self._capture = None
            self._thread = None
            with self._condition:
                self._jpeg = None

    def __enter__(self) -> Camera:
        """Open the camera and return this instance.

        Returns:
            The started camera instance.
        """
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop the camera when leaving a context-manager block."""
        self.stop()


def get_camera(**kwargs: Any) -> Camera:
    """Create a camera instance.

    Args:
        **kwargs: Keyword arguments accepted by :class:`Camera`.

    Returns:
        A configured but not yet started camera instance.
    """
    return Camera(**kwargs)
