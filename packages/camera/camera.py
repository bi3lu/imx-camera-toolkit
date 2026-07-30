"""High-level coordination of capture, processing, controls, and publishing."""

from __future__ import annotations

import logging
import threading
import time

from pathlib import Path
from typing import Any, Sequence

from .backends import CaptureBackend, GStreamerCaptureBackend, OpenCVCaptureBackend
from .config import (
    CameraConfig,
    DEFAULT_CAMERA_CONFIG,
    load_camera_config,
    validate_camera_config,
)
from .controls import (
    V4L2Controls,
    apply_live_properties,
    manual_control_properties,
    non_manual_control_properties,
)
from .pipeline import build_gstreamer_pipeline, normalize_argus_properties
from .processing import SoftwareHDRProcessor, SoftwareHDRSettings
from .publishing import JPEGPublisher, opencv_available

logger = logging.getLogger(__name__)


class Camera:
    """Coordinate one CSI camera pipeline and publish its newest JPEG frame.

    ``Camera`` owns the lifecycle of exactly one capture backend. It reads BGR
    frames, optionally passes them through software HDR, and delegates JPEG
    encoding and consumer synchronization to :class:`JPEGPublisher`.

    Args:
        quality: JPEG quality from 0 to 100. Overrides ``config.yml``.
        max_fps: Maximum JPEG encoding rate in frames per second. Overrides
            ``config.yml``.
        config_path: Optional path to a YAML configuration file.
        sensor_id: Zero-based CSI sensor identifier used by Argus. Overrides
            ``config.yml``.
        capture_width: Width captured directly from the sensor, in pixels.
        capture_height: Height captured directly from the sensor, in pixels.
        output_width: Width delivered to capture backends, in pixels.
        output_height: Height delivered to capture backends, in pixels.
        capture_fps: Camera capture rate, in frames per second.
        flip_method: NVIDIA ``nvvidconv`` flip transformation, from 0 to 7.
        argus_properties: Initial validated ``nvarguscamerasrc`` properties.
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
        """Initialize a camera without opening the capture source."""
        loaded_config = load_camera_config(config_path)
        config = CameraConfig(
            quality=loaded_config.quality if quality is None else quality,
            max_fps=loaded_config.max_fps if max_fps is None else max_fps,
            sensor_id=loaded_config.sensor_id if sensor_id is None else sensor_id,
            capture_width=(
                loaded_config.capture_width if capture_width is None else capture_width
            ),
            capture_height=(
                loaded_config.capture_height
                if capture_height is None
                else capture_height
            ),
            output_width=(
                loaded_config.output_width if output_width is None else output_width
            ),
            output_height=(
                loaded_config.output_height
                if output_height is None
                else output_height
            ),
            capture_fps=(
                loaded_config.capture_fps if capture_fps is None else capture_fps
            ),
            flip_method=(
                loaded_config.flip_method if flip_method is None else flip_method
            ),
        )
        validate_camera_config(config)

        self._config = config
        self._argus_properties = normalize_argus_properties(argus_properties)
        self._pipeline = self._build_pipeline(self._argus_properties)
        self._publisher = JPEGPublisher(config.quality, config.max_fps)
        self._v4l2_controls = V4L2Controls(config.sensor_id)
        self._software_hdr_settings = SoftwareHDRSettings()
        self._software_hdr_processor: SoftwareHDRProcessor | None = None
        self._software_hdr_lock = threading.RLock()

        self._backend: CaptureBackend | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lifecycle_lock = threading.RLock()

        self.frames_captured = 0
        self.last_error: Exception | None = None

    def _build_pipeline(self, argus_properties: Sequence[str]) -> str:
        """Build a pipeline using this camera's resolved static configuration."""
        return build_gstreamer_pipeline(
            sensor_id=self._config.sensor_id,
            capture_width=self._config.capture_width,
            capture_height=self._config.capture_height,
            output_width=self._config.output_width,
            output_height=self._config.output_height,
            framerate=self._config.capture_fps,
            flip_method=self._config.flip_method,
            argus_properties=argus_properties,
        )

    @property
    def pipeline(self) -> str:
        """str: GStreamer pipeline used by the next capture open operation."""
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
    def software_hdr_state(self) -> dict[str, object]:
        """Return the active software HDR configuration and bracket exposures."""
        with self._software_hdr_lock:
            exposures_us: list[int] = []

            if self._software_hdr_processor is not None:
                exposures_us = list(self._software_hdr_processor.exposures_us)

            return {
                "enabled": self._software_hdr_settings.enabled,
                "base_exposure_us": self._software_hdr_settings.base_exposure_us,
                "settle_frames": self._software_hdr_settings.settle_frames,
                "bracket_ev": list(SoftwareHDRProcessor.bracket_ev),
                "exposures_us": exposures_us,
            }

    @property
    def running(self) -> bool:
        """bool: Whether the background capture loop is active."""
        return self._running.is_set()

    @property
    def frame_available(self) -> bool:
        """bool: Whether at least one JPEG frame is currently available."""
        return self._publisher.frame_available

    @property
    def frame_number(self) -> int:
        """int: Monotonically increasing identifier of the latest JPEG frame."""
        return self._publisher.frame_number

    @property
    def jpeg(self) -> bytes | None:
        """bytes | None: Latest JPEG frame, or ``None`` when unavailable."""
        return self._publisher.jpeg

    @property
    def frames_encoded(self) -> int:
        """int: Number of frames successfully JPEG-encoded."""
        return self._publisher.frames_encoded

    @property
    def last_frame_time(self) -> float | None:
        """float | None: Unix timestamp of the latest encoded JPEG frame."""
        return self._publisher.last_frame_time

    def start(self) -> None:
        """Open the selected backend and start the background capture thread.

        Raises:
            RuntimeError: If OpenCV is unavailable, a previous thread is still
                stopping, or the Argus camera cannot be opened.
        """
        if not opencv_available():
            raise RuntimeError(
                "OpenCV is not available. Use the JetPack-provided Python/OpenCV "
                "environment with GStreamer support."
            )

        with self._lifecycle_lock:
            if self.running:
                return

            self._release_finished_capture()
            self._backend = self._create_backend()
            self._backend.open()

            with self._software_hdr_lock:
                if self._software_hdr_settings.enabled:
                    self._software_hdr_processor = self._create_software_hdr_processor()
                    self._software_hdr_processor.start(self._v4l2_controls.set_exposure)

            self.last_error = None
            self._running.set()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="imx-camera-capture",
                daemon=True,
            )
            self._thread.start()

    def _create_backend(self) -> CaptureBackend:
        """Create the preferred backend for the current Python environment."""
        if GStreamerCaptureBackend.available():
            return GStreamerCaptureBackend(
                self._pipeline,
                self._config.output_width,
                self._config.output_height,
            )
        return OpenCVCaptureBackend(self._pipeline)

    def _release_finished_capture(self) -> None:
        """Release resources retained after an unexpectedly stopped loop."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("camera capture thread is still stopping")

        self._release_backend()
        self._thread = None

    def _release_backend(self) -> None:
        """Close and discard the active capture backend."""
        if self._backend is not None:
            self._backend.close()
        self._backend = None

    def _capture_loop(self) -> None:
        """Read, process, and publish frames until capture stops."""
        try:
            while self.running:
                backend = self._backend

                if backend is None:
                    break

                success, frame = backend.read()

                if not success:
                    time.sleep(0.02)
                    continue

                self.frames_captured += 1
                processed_frame = self._process_frame(frame)

                if processed_frame is None:
                    continue

                self._publisher.publish(processed_frame)

        except Exception as error:
            self.last_error = error
            logger.exception("IMX camera capture failed")

        finally:
            self._running.clear()
            self._publisher.notify_waiters()

    def _process_frame(self, frame: Any) -> Any | None:
        """Run the active optional image processor for one BGR frame."""
        with self._software_hdr_lock:
            if self._software_hdr_processor is None:
                return frame

            return self._software_hdr_processor.process(
                frame,
                self._v4l2_controls.set_exposure,
            )

    def wait_for_jpeg(
        self,
        previous_frame_number: int,
        timeout: float = 2.0,
    ) -> tuple[int, bytes | None]:
        """Wait for a JPEG frame newer than a known frame number.

        Args:
            previous_frame_number: Frame number already consumed by the caller.
            timeout: Maximum time to wait, in seconds.

        Returns:
            The newest frame number and JPEG bytes. The bytes are ``None``
            until the first frame is encoded.

        Raises:
            ValueError: If ``timeout`` is negative.
        """
        if timeout < 0:
            raise ValueError("timeout must be greater than or equal to zero")

        return self._publisher.wait_for_jpeg(
            previous_frame_number,
            timeout,
            lambda: self.running,
        )

    def apply_argus_properties(
        self,
        properties: Sequence[str],
        *,
        restart_required: bool = False,
    ) -> None:
        """Apply Argus controls live when safe, otherwise rebuild capture.

        Manual exposure and gain use V4L2 to avoid the JetPack 6 Argus dynamic
        range issue. Non-manual controls are applied to the live Argus GObject.

        Args:
            properties: Valid ``nvarguscamerasrc`` property assignments.
            restart_required: Whether the requested controls change sensor mode.
        """
        normalized = normalize_argus_properties(properties)
        new_pipeline = self._build_pipeline(normalized)

        with self._lifecycle_lock:
            if normalized == self._argus_properties:
                return

            with self._software_hdr_lock:
                if self._software_hdr_settings.enabled and (
                    manual_control_properties(normalized)
                    != manual_control_properties(self._argus_properties)
                ):
                    raise RuntimeError(
                        "Disable software HDR before changing exposure or gain controls"
                    )

            backend = self._backend
            source = backend.argus_source if backend is not None else None

            if self.running and source is not None and not restart_required:
                self._v4l2_controls.apply_manual_controls(
                    manual_control_properties(self._argus_properties),
                    manual_control_properties(normalized),
                )

                apply_live_properties(
                    source,
                    non_manual_control_properties(self._argus_properties),
                    non_manual_control_properties(normalized),
                )

                self._argus_properties = normalized
                self._pipeline = new_pipeline
                return

        self.reconfigure_argus_properties(normalized)

    def configure_software_hdr(
        self,
        *,
        enabled: bool,
        base_exposure_us: int | None = None,
        settle_frames: int | None = None,
    ) -> dict[str, object]:
        """Enable or configure Jetson-side three-exposure HDR fusion.

        Args:
            enabled: Whether to enable software HDR.
            base_exposure_us: Middle exposure in microseconds. When omitted,
                retains the previous value.
            settle_frames: Frames discarded after each exposure change. When
                omitted, retains the previous value.

        Returns:
            JSON-ready software HDR state including resolved bracket exposures.
        """
        with self._lifecycle_lock, self._software_hdr_lock:
            current = self._software_hdr_settings
            settings = SoftwareHDRSettings(
                enabled=enabled,
                base_exposure_us=(
                    current.base_exposure_us
                    if base_exposure_us is None
                    else base_exposure_us
                ),
                settle_frames=(
                    current.settle_frames if settle_frames is None else settle_frames
                ),
            )
            if not settings.enabled:
                self._software_hdr_settings = settings
                self._software_hdr_processor = None
                return self.software_hdr_state

            processor = self._create_software_hdr_processor(settings)

            if self.running:
                processor.start(self._v4l2_controls.set_exposure)

            self._software_hdr_settings = settings
            self._software_hdr_processor = processor
            return self.software_hdr_state

    def _create_software_hdr_processor(
        self,
        settings: SoftwareHDRSettings | None = None,
    ) -> SoftwareHDRProcessor:
        """Create a processor whose longest bracket fits the capture period."""
        resolved_settings = settings or self._software_hdr_settings
        max_exposure_us = max(100, 1_000_000 // self._config.capture_fps)
        return SoftwareHDRProcessor(resolved_settings, max_exposure_us)

    def reconfigure_argus_properties(self, properties: Sequence[str]) -> None:
        """Rebuild capture with new Argus source properties.

        Args:
            properties: Valid ``nvarguscamerasrc`` property assignments.

        Raises:
            RuntimeError: If the running capture cannot stop or restart.
            ValueError: If a source property is malformed.
        """
        normalized = normalize_argus_properties(properties)
        new_pipeline = self._build_pipeline(normalized)

        with self._lifecycle_lock:
            if normalized == self._argus_properties:
                return

            old_properties = self._argus_properties
            old_pipeline = self._pipeline
            was_running = self.running

            if was_running:
                self.stop()
                if self._backend is not None:
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
        """Stop capture, release its backend, and discard the latest JPEG."""
        with self._lifecycle_lock:
            self._running.clear()
            self._publisher.notify_waiters()
            thread = self._thread

            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.0)

                if thread.is_alive():
                    logger.warning("IMX camera thread did not stop within 3 seconds")
                    return

            self._release_backend()
            self._thread = None
            self._publisher.clear()

    def __enter__(self) -> Camera:
        """Open the camera and return this instance."""
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop the camera when leaving a context-manager block."""
        self.stop()


def get_camera(**kwargs: Any) -> Camera:
    """Create a configured but not yet started camera instance."""
    return Camera(**kwargs)


__all__ = [
    "Camera",
    "CameraConfig",
    "DEFAULT_CAMERA_CONFIG",
    "SoftwareHDRSettings",
    "build_gstreamer_pipeline",
    "get_camera",
    "load_camera_config",
]
