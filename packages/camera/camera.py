from __future__ import annotations

import logging
import threading
import time
from typing import Any

try:
    import cv2  # NOTE: OpenCV is supplied by JetPack 6.2.2.

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
    """Continuously capture and JPEG-encode the newest image from one sensor.

    Call :meth:`start` before consuming frames and :meth:`stop` when finished,
    or use the camera as a context manager.  OpenCV with GStreamer support is
    required; NVIDIA's JetPack image provides it system-wide.
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
                return self._pipeline

    @property
    def running(self) -> bool:
        return self._running.is_set()

    @property
    def frame_available(self) -> bool:
        with self._condition:
            return self._jpeg is not None

    @property
    def frame_number(self) -> int:
        with self._condition:
            return self._frame_number

    @property
    def jpeg(self) -> bytes | None:
                with self._condition:
            return self._jpeg

    def start(self) -> None:
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
                if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("camera capture thread is still stopping")

        if self._capture is not None:
            self._capture.release()

        self._thread = None
        self._capture = None

    def _capture_loop(self) -> None:
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
        """Wait for a newer JPEG frame and return ``(frame_number, jpeg)``.

        A timeout or a stopped camera returns the latest available frame.
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
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def get_camera(**kwargs: Any) -> Camera:
        return Camera(**kwargs)
