"""Capture JPEG frames from IMX CSI cameras on NVIDIA Jetson devices."""

from __future__ import annotations

import logging
import re
import threading
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

try:
    import cv2  # NOTE: OpenCV is supplied by JetPack 6.2.2.

except ImportError:
    cv2: Any | None = None

try:
    import yaml

except ImportError:
    yaml: Any | None = None

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yml")


@dataclass(frozen=True)
class CameraConfig:
    """Validated settings used to create an IMX camera pipeline.

    Attributes:
        quality: JPEG quality from 0 to 100.
        max_fps: Maximum JPEG encoding rate in frames per second.
        sensor_id: Zero-based CSI sensor identifier used by Argus.
        capture_width: Width captured directly from the sensor, in pixels.
        capture_height: Height captured directly from the sensor, in pixels.
        output_width: Width of frames delivered to OpenCV, in pixels.
        output_height: Height of frames delivered to OpenCV, in pixels.
        capture_fps: Camera capture rate, in frames per second.
        flip_method: NVIDIA ``nvvidconv`` flip transformation, from 0 to 7.
    """

    quality: int = 65
    max_fps: float = 45.0
    sensor_id: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    output_width: int = 640
    output_height: int = 360
    capture_fps: int = 45
    flip_method: int = 0


DEFAULT_CAMERA_CONFIG = CameraConfig()


def build_gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    output_width: int = 640,
    output_height: int = 360,
    framerate: int = 30,
    flip_method: int = 0,
    argus_properties: Sequence[str] = (),
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
        argus_properties: Validated ``nvarguscamerasrc`` properties to append
            after the sensor identifier.

    Returns:
        A GStreamer pipeline string suitable for ``cv2.VideoCapture``.

    Raises:
        ValueError: If an identifier, dimension, frame rate, flip method, or
            source property is outside its supported range.
    """
    if sensor_id < 0:
        raise ValueError("sensor_id must be greater than or equal to zero")

    if min(capture_width, capture_height, output_width, output_height, framerate) <= 0:
        raise ValueError("frame dimensions and framerate must be greater than zero")

    if not 0 <= flip_method <= 7:
        raise ValueError("flip_method must be between 0 and 7")

    source_properties = _normalize_argus_properties(argus_properties)
    source_arguments = " ".join(source_properties)
    source_suffix = f" {source_arguments}" if source_arguments else ""

    return (
        f"nvarguscamerasrc sensor-id={sensor_id}{source_suffix} ! "
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


def _normalize_argus_properties(properties: Sequence[str]) -> tuple[str, ...]:
    """Validate properties that are safe to place in an Argus pipeline.

    Args:
        properties: ``nvarguscamerasrc`` property assignments.

    Returns:
        Normalized property assignments in their original order.

    Raises:
        ValueError: If an assignment is malformed or could alter pipeline
            structure.
    """
    if isinstance(properties, str):
        raise ValueError("argus_properties must be a sequence of assignments")

    normalized: list[str] = []
    property_pattern = re.compile(
        r'[A-Za-z][A-Za-z0-9-]*=(?:[A-Za-z0-9_.-]+|"[A-Za-z0-9_. -]+")'
    )

    for property_value in properties:
        if not isinstance(property_value, str):
            raise ValueError("each Argus property must be a string")

        if not property_pattern.fullmatch(property_value):
            raise ValueError(f"invalid Argus property: {property_value!r}")

        normalized.append(property_value)

    return tuple(normalized)


def _validate_config(config: CameraConfig) -> None:
    """Validate values that are not checked by the GStreamer pipeline builder.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If JPEG quality or encoding rate is invalid.
    """
    integer_fields = (
        "quality",
        "sensor_id",
        "capture_width",
        "capture_height",
        "output_width",
        "output_height",
        "capture_fps",
        "flip_method",
    )
    for field_name in integer_fields:
        value = getattr(config, field_name)

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer")

    if isinstance(config.max_fps, bool) or not isinstance(config.max_fps, (int, float)):
        raise ValueError("max_fps must be a number")

    if not 0 <= config.quality <= 100:
        raise ValueError("quality must be between 0 and 100")

    if config.max_fps <= 0:
        raise ValueError("max_fps must be greater than zero")

    if config.sensor_id < 0:
        raise ValueError("sensor_id must be greater than or equal to zero")
    if min(
        config.capture_width,
        config.capture_height,
        config.output_width,
        config.output_height,
        config.capture_fps,
    ) <= 0:
        raise ValueError("frame dimensions and framerate must be greater than zero")
    if not 0 <= config.flip_method <= 7:
        raise ValueError("flip_method must be between 0 and 7")


def _read_config_values(config_data: dict[str, Any]) -> CameraConfig:
    """Convert a parsed YAML mapping into a validated configuration.

    Args:
        config_data: Mapping stored under ``camera_config`` in the YAML file.

    Returns:
        Validated camera configuration.

    Raises:
        ValueError: If keys are unknown or values have invalid types or ranges.
    """
    defaults = DEFAULT_CAMERA_CONFIG
    valid_keys = set(defaults.__dataclass_fields__)
    unknown_keys = set(config_data) - valid_keys

    if unknown_keys:
        formatted_keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unknown camera configuration key(s): {formatted_keys}")

    values: dict[str, int | float] = {}

    for key in valid_keys:
        value = config_data.get(key, getattr(defaults, key))
        default_value = getattr(defaults, key)

        if isinstance(default_value, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")

        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")

        values[key] = value

    config = CameraConfig(**values)
    _validate_config(config)
    return config


def load_camera_config(config_path: str | Path | None = None) -> CameraConfig:
    """Load camera settings from YAML, falling back to built-in defaults.

    Args:
        config_path: Path to a YAML file. When omitted, uses the ``config.yml``
            located next to this module.

    Returns:
        A validated configuration. Built-in defaults are returned when the file
        is missing, cannot be read, is malformed, or contains invalid values.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    try:
        raw_config = path.read_text(encoding="utf-8")

    except FileNotFoundError:
        return DEFAULT_CAMERA_CONFIG

    except OSError as error:
        logger.warning("Could not read camera configuration %s: %s", path, error)
        return DEFAULT_CAMERA_CONFIG

    if yaml is None:
        logger.warning(
            "PyYAML is unavailable; using built-in camera configuration defaults"
        )
        return DEFAULT_CAMERA_CONFIG

    try:
        parsed_config = yaml.safe_load(raw_config)

        if not isinstance(parsed_config, dict):
            raise ValueError("the YAML document must be a mapping")

        config_data = parsed_config.get("camera_config")

        if not isinstance(config_data, dict):
            raise ValueError("camera_config must be a mapping")

        return _read_config_values(config_data)

    except (ValueError, yaml.YAMLError) as error:
        logger.warning("Invalid camera configuration %s: %s", path, error)
        return DEFAULT_CAMERA_CONFIG


class Camera:
    """Capture and JPEG-encode the latest image from one CSI sensor.

    Only the most recent encoded frame is retained in memory. This makes the
    class appropriate for live previews and streaming, where stale frames are
    less useful than current ones.

    Args:
        quality: JPEG quality from 0 to 100. Overrides ``config.yml``.
        max_fps: Maximum JPEG encoding rate in frames per second. Overrides
            ``config.yml``.
        config_path: Optional path to a YAML configuration file.
        sensor_id: Zero-based CSI sensor identifier used by Argus. Overrides
            ``config.yml``.
        capture_width: Width captured directly from the sensor, in pixels.
            Overrides ``config.yml``.
        capture_height: Height captured directly from the sensor, in pixels.
            Overrides ``config.yml``.
        output_width: Width of frames delivered to OpenCV, in pixels. Overrides
            ``config.yml``.
        output_height: Height of frames delivered to OpenCV, in pixels.
            Overrides ``config.yml``.
        capture_fps: Camera capture rate, in frames per second. Overrides
            ``config.yml``.
        flip_method: NVIDIA ``nvvidconv`` flip transformation, from 0 to 7.
            Overrides ``config.yml``.

    Attributes:
        frames_captured: Number of frames read from the camera.
        frames_encoded: Number of frames successfully JPEG-encoded.
        last_frame_time: Unix timestamp of the latest encoded frame, if any.
        last_error: Most recent background capture exception, if any.
    """

    def __init__(
        self,
        quality: int | None = None,
        max_fps: float | None = None,
        *,
        config_path: str | Path | None = None,
        sensor_id: int | None = None,
        capture_width: int | None = None,
        capture_height: int | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        capture_fps: int | None = None,
        flip_method: int | None = None,
        argus_properties: Sequence[str] = (),
    ) -> None:
        """Initialize a camera without opening its capture device.

        Raises:
            ValueError: If JPEG quality or encoding rate is invalid, or if a
                pipeline configuration argument is invalid.
        """
        loaded_config = load_camera_config(config_path)
        config = CameraConfig(
            quality=loaded_config.quality if quality is None else quality,
            max_fps=loaded_config.max_fps if max_fps is None else max_fps,
            sensor_id=loaded_config.sensor_id if sensor_id is None else sensor_id,
            capture_width=(
                loaded_config.capture_width if capture_width is None else capture_width
            ),
            capture_height=(
                loaded_config.capture_height if capture_height is None else capture_height
            ),
            output_width=(loaded_config.output_width if output_width is None else output_width),
            output_height=(
                loaded_config.output_height if output_height is None else output_height
            ),
            capture_fps=loaded_config.capture_fps if capture_fps is None else capture_fps,
            flip_method=loaded_config.flip_method if flip_method is None else flip_method,
        )
        _validate_config(config)

        self._pipeline = build_gstreamer_pipeline(
            sensor_id=config.sensor_id,
            capture_width=config.capture_width,
            capture_height=config.capture_height,
            output_width=config.output_width,
            output_height=config.output_height,
            framerate=config.capture_fps,
            flip_method=config.flip_method,
            argus_properties=argus_properties,
        )
        self._config = config
        self._argus_properties = _normalize_argus_properties(argus_properties)
        self._jpeg_quality = config.quality
        self._jpeg_interval = 1.0 / config.max_fps

        self._capture: Any | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lifecycle_lock = threading.RLock()
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
    def argus_properties(self) -> tuple[str, ...]:
        """tuple[str, ...]: Current Argus source property assignments."""
        with self._lifecycle_lock:
            return self._argus_properties

    @property
    def config(self) -> CameraConfig:
        """CameraConfig: Resolved configuration used by this camera instance."""
        return self._config

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

    def reconfigure_argus_properties(self, properties: Sequence[str]) -> None:
        """Apply Argus source properties, restarting active capture if needed.

        The OpenCV GStreamer backend does not expose a live source element.
        An active capture is therefore stopped and reopened using a pipeline
        containing the requested properties. If the new pipeline cannot open,
        the previous pipeline is restored when possible.

        Args:
            properties: Valid ``nvarguscamerasrc`` property assignments.

        Raises:
            RuntimeError: If the running capture cannot stop or the new
                pipeline cannot open.
            ValueError: If a source property is malformed.
        """
        normalized = _normalize_argus_properties(properties)
        new_pipeline = build_gstreamer_pipeline(
            sensor_id=self._config.sensor_id,
            capture_width=self._config.capture_width,
            capture_height=self._config.capture_height,
            output_width=self._config.output_width,
            output_height=self._config.output_height,
            framerate=self._config.capture_fps,
            flip_method=self._config.flip_method,
            argus_properties=normalized,
        )

        with self._lifecycle_lock:
            if normalized == self._argus_properties:
                return

            old_properties = self._argus_properties
            old_pipeline = self._pipeline
            was_running = self.running

            if was_running:
                self.stop()
                if self._capture is not None:
                    raise RuntimeError("camera capture thread did not stop")

            self._argus_properties = normalized
            self._pipeline = new_pipeline

            if not was_running:
                return

            try:
                self.start()

            except Exception:
                self._argus_properties = old_properties
                self._pipeline = old_pipeline

                try:
                    self.start()

                except Exception:
                    logger.exception("Could not restore the previous IMX pipeline")

                raise

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
