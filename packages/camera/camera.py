"""High-level coordination of capture, processing, controls, and publishing."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Sequence
from copy import copy as copy_image
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any

from packages.consumers.latest import LatestFrameHub, LatestFrameSubscription

from .backends import CaptureBackend, GStreamerCaptureBackend, OpenCVCaptureBackend
from .config import (
    DEFAULT_CAMERA_CONFIG,
    CameraConfig,
    load_camera_config,
    validate_camera_config,
)
from .controls import (
    V4L2Controls,
    apply_live_properties,
    manual_control_properties,
    non_manual_control_properties,
)
from .errors import (
    CameraConfigurationError,
    CameraDependencyError,
    CameraError,
    CameraOpenError,
    CameraReadError,
    CameraRecoveryError,
    CameraTimeoutError,
)
from .models import (
    CameraFrame,
    CameraStats,
    Frame,
    FrameFormat,
    MemoryType,
    MetricsRecorder,
    PipelineMetrics,
    PipelineStage,
)
from .pipeline import build_gstreamer_pipeline, normalize_argus_properties
from .processing import SoftwareHDRProcessor, SoftwareHDRSettings
from .publishing import JPEGPublisher, RawFramePublisher, opencv_available

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraRecoveryPolicy:
    """Validated policy used to recover from transient capture failures.

    Args:
        max_attempts: Number of backend reopen attempts after a capture error.
        initial_backoff: Delay before the first reopen attempt, in seconds.
        max_consecutive_read_failures: Unsuccessful reads tolerated before a
            backend recovery is attempted.
    """

    max_attempts: int = 3
    initial_backoff: float = 0.25
    max_consecutive_read_failures: int = 20

    def __post_init__(self) -> None:
        """Validate recovery policy values."""
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 0
        ):
            raise CameraConfigurationError(
                "max_attempts must be a non-negative integer"
            )

        if (
            isinstance(self.initial_backoff, bool)
            or not isinstance(self.initial_backoff, (int, float))
            or self.initial_backoff < 0
        ):
            raise CameraConfigurationError(
                "initial_backoff must be a non-negative number"
            )

        if (
            isinstance(self.max_consecutive_read_failures, bool)
            or not isinstance(self.max_consecutive_read_failures, int)
            or self.max_consecutive_read_failures <= 0
        ):
            raise CameraConfigurationError(
                "max_consecutive_read_failures must be positive"
            )


class Camera:
    """Coordinate one CSI camera pipeline and publish its newest BGR frame.

    ``Camera`` owns the lifecycle of exactly one capture backend. It reads BGR
    frames, optionally passes them through software HDR, then publishes one
    newest raw frame for external consumers and optionally encodes JPEG for
    preview and streaming consumers.

    Args:
        config: Immutable static camera configuration. When omitted, settings
            are loaded from ``config.yml`` with built-in defaults as fallback.
        quality: JPEG quality from 0 to 100. Legacy override for ``config``.
        max_fps: Maximum JPEG encoding rate in frames per second. Overrides
            ``config.yml``.
        config_path: Optional path to a YAML configuration file.
        sensor_id: Zero-based CSI sensor identifier used by Argus. Overrides
            ``config.yml``.
        capture_width: Width captured directly from the sensor, in pixels.
        capture_height: Height captured directly from the sensor, in pixels.
        output_width: Width delivered to capture backends, in pixels.
        output_height: Height delivered to capture backends, in pixels.
        fps: Camera capture rate, in frames per second.
        capture_fps: Deprecated alias for ``fps``.
        flip_method: NVIDIA ``nvvidconv`` flip transformation, from 0 to 7.
        argus_properties: Initial validated ``nvarguscamerasrc`` properties.
        enable_preview: Legacy override for JPEG preview encoding. Prefer
            :attr:`CameraConfig.enable_preview` for new code.
    """

    STATS_WINDOW_NS = 1_000_000_000

    def __init__(
        self,
        config: CameraConfig | int | None = None,
        quality: int | float | None = None,
        max_fps: float | None = None,
        *,
        config_path: str | Path | None = None,
        sensor_id: int | None = None,
        capture_width: int | None = None,
        capture_height: int | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        fps: int | None = None,
        capture_fps: int | None = None,
        flip_method: int | None = None,
        argus_properties: Sequence[str] = (),
        recovery_policy: CameraRecoveryPolicy | None = None,
        enable_preview: bool | None = None,
    ) -> None:
        """Initialize a camera without opening the capture source."""
        if enable_preview is not None and not isinstance(enable_preview, bool):
            raise CameraConfigurationError("enable_preview must be a boolean")

        if fps is not None and capture_fps is not None:
            raise CameraConfigurationError(
                "use either fps or legacy capture_fps, not both"
            )

        if isinstance(config, int) and not isinstance(config, bool):
            if quality is None:
                quality = config
            else:
                max_fps = quality
                quality = config
            config = None

        if quality is not None and (
            isinstance(quality, bool) or not isinstance(quality, int)
        ):
            raise CameraConfigurationError("quality must be an integer")

        if config is not None and not isinstance(config, CameraConfig):
            raise TypeError("config must be a CameraConfig or None")

        if config is not None and config_path is not None:
            raise CameraConfigurationError(
                "config and config_path cannot be used together"
            )

        base_config = config if config is not None else load_camera_config(config_path)
        resolved_fps = fps if fps is not None else capture_fps
        resolved_config = replace(
            base_config,
            quality=base_config.quality if quality is None else quality,
            max_fps=base_config.max_fps if max_fps is None else max_fps,
            sensor_id=base_config.sensor_id if sensor_id is None else sensor_id,
            capture_width=(
                base_config.capture_width if capture_width is None else capture_width
            ),
            capture_height=(
                base_config.capture_height if capture_height is None else capture_height
            ),
            output_width=(
                base_config.output_width if output_width is None else output_width
            ),
            output_height=(
                base_config.output_height if output_height is None else output_height
            ),
            fps=base_config.fps if resolved_fps is None else resolved_fps,
            flip_method=(
                base_config.flip_method if flip_method is None else flip_method
            ),
            enable_preview=(
                base_config.enable_preview if enable_preview is None else enable_preview
            ),
        )
        validate_camera_config(resolved_config)

        if resolved_config.output_format is not FrameFormat.BGR_CPU:
            raise CameraConfigurationError(
                "Camera supports only FrameFormat.BGR_CPU; use GpuCamera for "
                "NV12_NVMM"
            )

        self._config = resolved_config
        self._enable_preview = resolved_config.enable_preview
        self._recovery_policy = recovery_policy or CameraRecoveryPolicy()
        if resolved_config.sensor_mode is not None and any(
            property_value.startswith("sensor-mode=")
            for property_value in argus_properties
        ):
            raise CameraConfigurationError(
                "CameraConfig.sensor_mode cannot be combined with an "
                "argus_properties sensor-mode setting"
            )

        config_properties = (
            ()
            if resolved_config.sensor_mode is None
            else (f"sensor-mode={resolved_config.sensor_mode}",)
        )
        self._argus_properties = normalize_argus_properties(
            (*config_properties, *argus_properties)
        )
        self._pipeline = self._build_pipeline(self._argus_properties)
        self._publisher = JPEGPublisher(
            resolved_config.quality,
            resolved_config.preview_fps,
        )
        self._raw_publisher = RawFramePublisher()
        self._frame_hub = LatestFrameHub[Frame](self.record_consumer_drop)
        self._v4l2_controls = V4L2Controls(resolved_config.sensor_id)
        self._software_hdr_settings = SoftwareHDRSettings()
        self._software_hdr_processor: SoftwareHDRProcessor | None = None
        self._software_hdr_lock = threading.RLock()

        self._backend: CaptureBackend | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._capture_timestamps_ns: deque[int] = deque()
        self._metrics = MetricsRecorder()

        self.frames_captured = 0
        self.dropped_frames = 0
        self._last_frame_timestamp_ns: int | None = None
        self._last_capture_timestamp_ns: int | None = None
        self._consecutive_failures = 0
        self.last_error: Exception | None = None
        self.recovery_attempts = 0
        self.recoveries = 0
        self.last_recovery_error: Exception | None = None

    def _build_pipeline(self, argus_properties: Sequence[str]) -> str:
        """Build a pipeline using this camera's resolved static configuration."""
        return build_gstreamer_pipeline(
            sensor_id=self._config.sensor_id,
            capture_width=self._config.capture_width,
            capture_height=self._config.capture_height,
            output_width=self._config.output_width,
            output_height=self._config.output_height,
            framerate=self._config.fps,
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

    def stats(self) -> CameraStats:
        """Return a consistent point-in-time snapshot of capture diagnostics.

        The method performs no I/O and does not require optional telemetry
        integrations. ``capture_fps`` is calculated from successful source
        reads within the most recent one-second window; it is zero when the
        capture worker is stopped or has not produced at least two recent
        frames.

        Returns:
            Immutable capture diagnostics suitable for health checks,
            watchdogs, metrics adapters, dashboards, or alerting systems.
        """
        running = self.running
        now_ns = time.monotonic_ns()

        with self._stats_lock:
            self._prune_capture_timestamps(now_ns)
            capture_fps = self._capture_fps() if running else 0.0
            return CameraStats(
                captured_frames=self.frames_captured,
                dropped_frames=self.dropped_frames,
                capture_fps=capture_fps,
                last_frame_timestamp_ns=self._last_frame_timestamp_ns,
                recovery_count=self.recoveries,
                consecutive_failures=self._consecutive_failures,
                running=running,
                pipeline=self._metrics.snapshot(),
                consumer_dropped_frames=self._metrics.consumer_drops(),
                last_capture_timestamp_ns=self._last_capture_timestamp_ns,
            )

    @property
    def active_backend(self) -> str | None:
        """Stable name of the currently opened capture backend."""
        with self._lifecycle_lock:
            if not self.running or self._backend is None:
                return None
            return self._backend.backend_name

    @property
    def frame_format(self) -> FrameFormat:
        """Explicit output format for the legacy CPU camera path."""
        return self._config.output_format

    @property
    def memory_type(self) -> MemoryType:
        """Memory domain used by frames from this camera."""
        return self._config.output_memory

    @property
    def frame_resolution(self) -> tuple[int, int]:
        """Output frame width and height in pixels."""
        return self._config.output_width, self._config.output_height

    @property
    def pipeline_metrics(self) -> PipelineMetrics:
        """Return immutable per-stage latency aggregates."""
        return self._metrics.snapshot()

    @property
    def consumer_dropped_frames(self) -> dict[str, int]:
        """Return a copy of latest-frame drops reported per consumer."""
        return dict(self._metrics.consumer_drops())

    def record_stage_latency(
        self,
        stage: PipelineStage | str,
        duration_ns: int,
    ) -> None:
        """Record model-agnostic transfer, inference, encoder, or E2E latency.

        Capture records the stages it owns. External inference consumers may
        use the same public method for their own stage without introducing a
        model or TensorRT dependency into the toolkit.
        """
        self._metrics.record_stage(stage, duration_ns)

    def record_consumer_drop(self, consumer: str, count: int = 1) -> None:
        """Record frames skipped by one named latest-frame consumer."""
        self._metrics.record_consumer_drop(consumer, count)

    def _prune_capture_timestamps(self, now_ns: int) -> None:
        """Discard timestamps outside the fixed recent-FPS sampling window."""
        oldest_timestamp_ns = now_ns - self.STATS_WINDOW_NS

        while (
            self._capture_timestamps_ns
            and self._capture_timestamps_ns[0] < oldest_timestamp_ns
        ):
            self._capture_timestamps_ns.popleft()

    def _capture_fps(self) -> float:
        """Calculate the recent source-read rate while holding ``_stats_lock``."""
        if len(self._capture_timestamps_ns) < 2:
            return 0.0

        first_timestamp_ns = self._capture_timestamps_ns[0]
        last_timestamp_ns = self._capture_timestamps_ns[-1]
        elapsed_ns = last_timestamp_ns - first_timestamp_ns

        if elapsed_ns <= 0:
            return 0.0

        return (len(self._capture_timestamps_ns) - 1) * 1_000_000_000 / elapsed_ns

    def _record_capture(
        self,
        timestamp_ns: int,
        capture_timestamp_ns: int | None = None,
    ) -> None:
        """Record one successful source read for diagnostics."""
        with self._stats_lock:
            self.frames_captured += 1
            self._last_frame_timestamp_ns = timestamp_ns
            self._last_capture_timestamp_ns = capture_timestamp_ns
            self._consecutive_failures = 0
            self._capture_timestamps_ns.append(timestamp_ns)
            self._prune_capture_timestamps(timestamp_ns)

    def _record_dropped_frame(self, *, failed_read: bool = False) -> int:
        """Record a frame omitted from raw publication and return failures."""
        with self._stats_lock:
            self.dropped_frames += 1
            if failed_read:
                self._consecutive_failures += 1
            return self._consecutive_failures

    def _reset_consecutive_failures(self) -> None:
        """Clear the source-read failure streak after a successful recovery."""
        with self._stats_lock:
            self._consecutive_failures = 0

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
    def preview_enabled(self) -> bool:
        """bool: Whether JPEG encoding for preview clients is enabled."""
        return self._enable_preview

    def set_preview_enabled(self, enabled: bool) -> None:
        """Enable or disable JPEG preview without recreating capture.

        The raw-frame path is unaffected. When enabled during active capture,
        the next captured frame is encoded for preview clients; when disabled,
        the retained JPEG is discarded. Starting a camera with preview enabled
        still requires the JetPack-provided OpenCV runtime.

        Args:
            enabled: Whether JPEG encoding for browser and MJPEG clients is
                active.

        Raises:
            ValueError: If ``enabled`` is not a boolean.
        """
        if not isinstance(enabled, bool):
            raise CameraConfigurationError("enabled must be a boolean")

        with self._lifecycle_lock:
            if self._enable_preview == enabled:
                return

            self._enable_preview = enabled
            self._config = replace(self._config, enable_preview=enabled)

            if not enabled:
                self._publisher.clear()
                self._publisher.notify_waiters()

    @property
    def frame_number(self) -> int:
        """int: Monotonically increasing identifier of the latest JPEG frame."""
        return self._publisher.frame_number

    @property
    def jpeg(self) -> bytes | None:
        """bytes | None: Latest JPEG frame, or ``None`` when unavailable."""
        return self._publisher.jpeg

    @property
    def raw_frame_number(self) -> int:
        """int: Identifier of the newest processed raw BGR frame."""
        return self._raw_publisher.frame_number

    @property
    def raw_frame(self) -> object | None:
        """object | None: Newest processed raw BGR frame without a copy.

        Consumers must treat this payload as read-only. It is shared with JPEG
        encoding and external frame consumers to avoid needless copy overhead.
        """
        return self._raw_publisher.frame

    def latest_frame(self, copy: bool = True) -> Frame | None:
        """Return the newest raw frame immediately, without waiting.

        The camera retains a single raw BGR frame. As with :meth:`read`, the
        default returns an image copy owned by the caller. Set ``copy=False``
        to receive the shared read-only payload without a copy.

        Args:
            copy: Whether to return an independent image copy.

        Returns:
            The newest raw frame, or ``None`` before the first frame or after
            camera shutdown.

        Raises:
            ValueError: If ``copy`` is not a boolean or an image cannot be
                copied.
        """
        return self._prepare_frame(self._raw_publisher.latest_frame, copy)

    def subscribe_latest(self, name: str) -> LatestFrameSubscription[Frame]:
        """Create one non-blocking latest-frame slot for a named consumer.

        The slot receives the same read-only BGR/CPU :class:`Frame` published
        by the compatibility API. A slow consumer skips overwritten frames and
        never blocks capture or another subscriber. Camera shutdown closes all
        remaining subscriptions.
        """
        with self._lifecycle_lock:
            if self._frame_hub.closed:
                if self.running:
                    raise RuntimeError("camera frame subscriptions are closed")

                self._frame_hub = LatestFrameHub[Frame](self.record_consumer_drop)

            return self._frame_hub.subscribe(name)

    def latest_jpeg(self) -> bytes | None:
        """Return the newest preview JPEG without starting a capture pipeline.

        Returns:
            Latest encoded JPEG data, or ``None`` when preview is disabled or
            no frame has been encoded.
        """
        if not self._enable_preview:
            return None

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
            CameraDependencyError: If a required local runtime is unavailable.
            CameraOpenError: If a previous worker is still stopping or the
                configured camera cannot be opened.
        """
        if not GStreamerCaptureBackend.available() and not opencv_available():
            raise CameraDependencyError(
                "System OpenCV with GStreamer support is required."
            )

        if self._enable_preview and not opencv_available():
            raise CameraDependencyError(
                "System OpenCV with GStreamer support is required for JPEG "
                "preview. Disable preview with Camera(enable_preview=False) "
                "for raw-frame-only capture."
            )

        with self._lifecycle_lock:
            if self.running:
                return

            self._release_finished_capture()
            self._backend = self._create_backend()

            try:
                self._backend.open()

            except CameraError:
                raise

            except Exception as error:
                raise CameraOpenError(
                    f"Could not open the IMX camera backend: {error}"
                ) from error

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
            raise CameraOpenError("camera capture thread is still stopping")

        self._release_backend()
        self._thread = None

    def _release_backend(self) -> None:
        """Close and discard the active capture backend."""
        if self._backend is not None:
            self._backend.close()
        self._backend = None

    def _capture_loop(self) -> None:
        """Read, process, and publish frames until capture stops."""
        consecutive_read_failures = 0
        while self.running:
            try:
                backend = self._backend

                if backend is None:
                    break

                try:
                    success, frame = backend.read()

                except Exception:
                    self._record_dropped_frame(failed_read=True)
                    raise

                if not success:
                    consecutive_read_failures = self._record_dropped_frame(
                        failed_read=True
                    )
                    if (
                        consecutive_read_failures
                        >= self._recovery_policy.max_consecutive_read_failures
                    ):
                        raise CameraReadError("camera backend stopped producing frames")
                    time.sleep(0.02)
                    continue

                consecutive_read_failures = 0
                timestamp_ns = time.monotonic_ns()
                self._record_capture(timestamp_ns, backend.capture_timestamp_ns)

                if backend.transfer_duration_ns is not None:
                    self._metrics.record_stage(
                        PipelineStage.TRANSFER,
                        backend.transfer_duration_ns,
                    )

                processed_frame = self._process_frame(frame)

                if processed_frame is None:
                    self._record_dropped_frame()
                    continue

                self._publish_frame(
                    processed_frame,
                    timestamp_ns,
                    capture_timestamp_ns=backend.capture_timestamp_ns,
                )

            except Exception as error:
                self.last_error = error
                logger.exception("IMX camera capture failed")

                if not self._recover_backend():
                    if self.running:
                        self.last_error = CameraRecoveryError(
                            "camera recovery attempts were exhausted"
                        )

                    self._running.clear()

        self._publisher.notify_waiters()
        self._raw_publisher.notify_waiters()
        self._frame_hub.close()

    def _recover_backend(self) -> bool:
        """Reopen the capture backend after a transient capture failure.

        Returns:
            ``True`` when a new backend is opened and capture may continue.
        """
        for attempt in range(self._recovery_policy.max_attempts):
            if not self.running:
                return False

            if attempt:
                time.sleep(self._recovery_policy.initial_backoff * (2**attempt))

            with self._stats_lock:
                self.recovery_attempts += 1

            with self._lifecycle_lock:
                if not self.running:
                    return False
                try:
                    self._release_backend()
                    backend = self._create_backend()
                    backend.open()
                    self._backend = backend

                except Exception as error:
                    self.last_recovery_error = error
                    logger.warning(
                        "Camera recovery attempt %s/%s failed: %s",
                        attempt + 1,
                        self._recovery_policy.max_attempts,
                        error,
                    )
                    continue

            with self._stats_lock:
                self.recoveries += 1
            self._reset_consecutive_failures()
            self.last_recovery_error = None
            self.last_error = None
            logger.info("Camera capture recovered on attempt %s", attempt + 1)
            return True

        return False

    def _process_frame(self, frame: Any) -> Any | None:
        """Run the active optional image processor for one BGR frame."""
        with self._software_hdr_lock:
            if self._software_hdr_processor is None:
                return frame

            return self._software_hdr_processor.process(
                frame,
                self._v4l2_controls.set_exposure,
            )

    def _publish_frame(
        self,
        frame: object,
        timestamp_ns: int,
        *,
        capture_timestamp_ns: int | None = None,
    ) -> None:
        """Publish a raw frame and optionally encode it for preview clients."""
        self._raw_publisher.publish(
            frame,
            width=self._config.output_width,
            height=self._config.output_height,
            timestamp_ns=timestamp_ns,
            capture_timestamp_ns=capture_timestamp_ns,
        )
        published_frame = self._raw_publisher.latest_frame
        if published_frame is not None:
            self._frame_hub.publish(published_frame)

        if self._enable_preview:
            encoder_started_ns = time.monotonic_ns()
            encoded = self._publisher.publish(frame)
            encoder_duration_ns = time.monotonic_ns() - encoder_started_ns

            if encoded:
                self._metrics.record_stage(
                    PipelineStage.ENCODER,
                    encoder_duration_ns,
                )
            else:
                self._metrics.record_consumer_drop("preview")

        self._metrics.record_stage(
            PipelineStage.END_TO_END,
            max(time.monotonic_ns() - timestamp_ns, 0),
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
            raise CameraConfigurationError(
                "timeout must be greater than or equal to zero"
            )

        return self._publisher.wait_for_jpeg(
            previous_frame_number,
            timeout,
            lambda: self.running,
        )

    def wait_for_raw_frame(
        self,
        previous_frame_number: int,
        timeout: float = 2.0,
    ) -> tuple[int, object | None]:
        """Wait for a processed BGR frame newer than a known frame number.

        The returned payload is not copied. Treat it as read-only; the camera
        and Vision Pipeline retain only the most recent raw frame reference.

        Args:
            previous_frame_number: Frame number already consumed by the caller.
            timeout: Maximum time to wait in seconds.

        Returns:
            Newest raw frame number and BGR payload, when available.

        Raises:
            ValueError: If ``timeout`` is negative.
        """
        if timeout < 0:
            raise CameraConfigurationError(
                "timeout must be greater than or equal to zero"
            )

        return self._raw_publisher.wait_for_frame(
            previous_frame_number,
            timeout,
            lambda: self.running,
        )

    def read(self, timeout: float = 2.0, copy: bool = True) -> Frame | None:
        """Return the newest available processed BGR frame.

        The camera retains exactly one raw frame. This method never creates a
        consumer queue, so a caller may receive a newer frame than one observed
        by another consumer and older frames may be skipped. It neither encodes
        JPEG data nor invokes image inference.

        By default, the BGR image is copied and the caller exclusively owns the
        returned image buffer. With ``copy=False``, the returned image is the
        publisher's shared BGR payload and must be treated as read-only. The
        shared form avoids only the additional Python API copy. Capture has
        already converted NV12/NVMM to BGR and copied the frame into host RAM;
        this option is not a GPU zero-copy guarantee.

        Args:
            timeout: Maximum wait for the first available raw frame, in seconds.
            copy: Whether to return an independent image copy.

        Returns:
            The newest available raw frame, or ``None`` when no frame arrives
            before the timeout or capture stops while waiting.

        Raises:
            RuntimeError: If the camera is not running.
            ValueError: If ``timeout`` or ``copy`` is invalid, or the image
                payload cannot be copied.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise CameraConfigurationError(
                "timeout must be a finite non-negative number"
            )

        if not isfinite(timeout) or timeout < 0:
            raise CameraConfigurationError(
                "timeout must be a finite non-negative number"
            )

        self._validate_copy(copy)

        if not self.running:
            raise CameraReadError("camera is not running; call start() before read()")

        frame = self._raw_publisher.wait_for_camera_frame(
            previous_frame_number=-1,
            timeout=timeout,
            is_running=lambda: self.running,
        )

        return self._prepare_frame(frame, copy)

    @staticmethod
    def _validate_copy(copy: bool) -> None:
        """Validate a raw-frame image ownership option."""
        if not isinstance(copy, bool):
            raise CameraConfigurationError("copy must be a boolean")

    @staticmethod
    def _prepare_frame(frame: Frame | None, copy: bool) -> Frame | None:
        """Return a copied or shared frame according to caller ownership."""
        Camera._validate_copy(copy)

        if frame is None or not copy:
            return frame

        try:
            image = copy_image(frame.image)

        except Exception as error:
            raise CameraReadError(
                "camera frame image cannot be copied; use read(copy=False) "
                "only when the consumer can treat the shared image as read-only"
            ) from error

        return Frame(
            image=image,
            sequence=frame.sequence,
            timestamp_ns=frame.timestamp_ns,
            capture_timestamp_ns=frame.capture_timestamp_ns,
            width=frame.width,
            height=frame.height,
            format=frame.format,
        )

    def read_image(self, timeout: float = 2.0, copy: bool = True) -> object | None:
        """Return only the newest BGR image for compatibility-oriented callers.

        This is equivalent to ``camera.read(timeout=timeout, copy=copy)``
        followed by access to ``Frame.image``. Prefer :meth:`read` in new
        integrations to retain timestamps, dimensions, pixel format, and frame
        sequence information.

        Args:
            timeout: Maximum wait for a raw frame, in seconds.
            copy: Whether to return an independent image copy.

        Returns:
            Raw BGR image payload, or ``None`` when no frame is available.
        """
        frame = self.read(timeout=timeout, copy=copy)
        return frame.image if frame is not None else None

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
                    raise CameraConfigurationError(
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
        max_exposure_us = max(100, 1_000_000 // self._config.fps)
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
                    raise CameraRecoveryError("camera capture thread did not stop")

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
            self._raw_publisher.notify_waiters()
            self._frame_hub.close()
            thread = self._thread

            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.0)

                if thread.is_alive():
                    logger.warning("IMX camera thread did not stop within 3 seconds")
                    return

            self._release_backend()
            self._thread = None
            self._publisher.clear()
            self._raw_publisher.clear()

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
    "CameraConfigurationError",
    "CameraDependencyError",
    "CameraError",
    "CameraFrame",
    "CameraOpenError",
    "CameraReadError",
    "Frame",
    "FrameFormat",
    "MemoryType",
    "PipelineMetrics",
    "PipelineStage",
    "CameraConfig",
    "CameraRecoveryPolicy",
    "CameraRecoveryError",
    "CameraStats",
    "CameraTimeoutError",
    "DEFAULT_CAMERA_CONFIG",
    "SoftwareHDRSettings",
    "build_gstreamer_pipeline",
    "get_camera",
    "load_camera_config",
]
