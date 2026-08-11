"""High-level NVMM camera with a borrowed latest-frame contract."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from math import isfinite
from pathlib import Path
from typing import Protocol

from packages.consumers.latest import LatestFrameHub, LatestFrameSubscription

from .backends import GpuGStreamerCaptureBackend
from .camera import CameraRecoveryPolicy
from .config import CameraConfig, load_camera_config
from .errors import (
    CameraConfigurationError,
    CameraDependencyError,
    CameraError,
    CameraOpenError,
    CameraReadError,
    CameraRecoveryError,
)
from .models import (
    CameraStats,
    EncodedStreamDescription,
    EncodedVideoFrame,
    FrameFormat,
    GpuFrame,
    HardwareVideoConfig,
    MemoryType,
    MetricsRecorder,
    PipelineMetrics,
    PipelineStage,
    VideoCodec,
    VideoEncoderBackend,
    VideoEncoderPipeline,
    VideoEncoderPipelineFactory,
    VideoEncodeStats,
    VideoOverlayRenderer,
)
from .pipeline import (
    build_gpu_gstreamer_pipeline,
    build_video_encoder_pipeline,
    normalize_argus_properties,
)
from .publishing import (
    EncodedJPEGPublisher,
    EncodedVideoPublisher,
    GpuFramePublisher,
)

logger = logging.getLogger(__name__)


class _GpuBackend(Protocol):
    """Backend operations required by :class:`GpuCamera`."""

    @property
    def backend_name(self) -> str:
        """Stable backend identifier."""
        ...

    def open(self) -> None:
        """Open both pipeline branches."""
        ...

    def read(self) -> tuple[bool, GpuFrame | None]:
        """Pull one borrowed NVMM frame."""
        ...

    def read_preview(self) -> bytes | None:
        """Pull one independently encoded preview frame."""
        ...

    def read_video(self) -> EncodedVideoFrame | None:
        """Pull one independently encoded access unit."""
        ...

    def close(self) -> None:
        """Close both branches as one pipeline."""
        ...


class GpuCamera:
    """Experimentally capture newest NV12 frames without CPU conversion.

    ``GpuCamera`` is an explicit opt-in API. It never changes ``Camera`` or its
    BGR/NumPy ``raw_frame`` behavior. The inference branch yields borrowed
    :class:`GpuFrame` leases, while the optional preview branch is isolated by
    a leaky GStreamer queue and NVIDIA JPEG encoder.
    """

    STATS_WINDOW_NS = 1_000_000_000
    ERROR_LOG_INTERVAL_SECONDS = 5.0

    def __init__(
        self,
        config: CameraConfig | None = None,
        *,
        config_path: str | Path | None = None,
        recovery_policy: CameraRecoveryPolicy | None = None,
        enable_preview: bool | None = None,
        video_config: HardwareVideoConfig | None = None,
        video_overlay: VideoOverlayRenderer | None = None,
        overlay_error_policy: str = "fail-open",
        encoder_pipeline_factory: VideoEncoderPipelineFactory | None = None,
        argus_properties: tuple[str, ...] = (),
        experimental: bool = False,
    ) -> None:
        """Initialize an NVMM pipeline without opening the camera."""
        if not isinstance(experimental, bool):
            raise CameraConfigurationError("experimental must be a boolean")

        if not experimental:
            raise CameraConfigurationError(
                "GpuCamera is an experimental API; pass experimental=True "
                "after reviewing the GPU compatibility guide"
            )

        if config is not None and config_path is not None:
            raise CameraConfigurationError(
                "config and config_path cannot be used together"
            )

        if enable_preview is not None and not isinstance(enable_preview, bool):
            raise CameraConfigurationError("enable_preview must be a boolean")

        if video_config is not None and not isinstance(
            video_config, HardwareVideoConfig
        ):
            raise CameraConfigurationError(
                "video_config must be a HardwareVideoConfig or None"
            )
        if overlay_error_policy not in {"fail-open", "fail-closed"}:
            raise CameraConfigurationError(
                "overlay_error_policy must be fail-open or fail-closed"
            )
        if encoder_pipeline_factory is not None and not callable(
            encoder_pipeline_factory
        ):
            raise CameraConfigurationError(
                "encoder_pipeline_factory must be callable or None"
            )
        if encoder_pipeline_factory is not None and video_config is None:
            raise CameraConfigurationError(
                "encoder_pipeline_factory requires video_config"
            )
        if video_overlay is not None:
            if video_config is None:
                raise CameraConfigurationError(
                    "video_overlay requires a video encoder configuration"
                )
            if not isinstance(video_overlay, VideoOverlayRenderer):
                raise CameraConfigurationError(
                    "video_overlay must implement VideoOverlayRenderer"
                )
            if video_overlay.memory_type is not MemoryType.NVMM:
                raise CameraConfigurationError(
                    "production video overlays must operate on NVMM; use "
                    "InferencePreviewSource for the CPU fallback"
                )

        base_config = config or load_camera_config(config_path)
        resolved_preview = (
            base_config.enable_preview if enable_preview is None else enable_preview
        )
        self._config = CameraConfig(
            sensor_id=base_config.sensor_id,
            capture_width=base_config.capture_width,
            capture_height=base_config.capture_height,
            output_width=base_config.output_width,
            output_height=base_config.output_height,
            fps=base_config.fps,
            flip_method=base_config.flip_method,
            sensor_mode=base_config.sensor_mode,
            enable_preview=resolved_preview,
            quality=base_config.quality,
            max_fps=base_config.max_fps,
            output_format=FrameFormat.NV12_NVMM,
        )
        config_properties = (
            ()
            if self._config.sensor_mode is None
            else (f"sensor-mode={self._config.sensor_mode}",)
        )
        self._argus_properties = normalize_argus_properties(
            (*config_properties, *argus_properties)
        )
        self._video_config = video_config
        self._video_overlay = video_overlay
        self._overlay_error_policy = overlay_error_policy
        self._encoder_pipeline_factory = encoder_pipeline_factory
        self._resolved_video_encoder_backend: str | None = None
        self._encoded_stream_description: EncodedStreamDescription | None = None
        self._pipeline = ""
        self._rebuild_pipeline()
        self._recovery_policy = recovery_policy or CameraRecoveryPolicy()
        self._gpu_publisher = GpuFramePublisher()
        self._frame_hub = self._new_frame_hub()
        self._preview_publisher = EncodedJPEGPublisher(self._config.preview_fps)
        self._video_publisher = EncodedVideoPublisher()
        self._video_hub = LatestFrameHub[EncodedVideoFrame](self.record_consumer_drop)
        self._metrics = MetricsRecorder()
        self._backend: _GpuBackend | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._stats_lock = threading.Lock()
        self._capture_timestamps_ns: deque[int] = deque()
        self._sequence = 0
        self._consecutive_recovery_attempts = 0
        self._last_capture_error_log_at = 0.0
        self._suppressed_capture_error_logs = 0

        self.frames_captured = 0
        self.dropped_frames = 0
        self._last_frame_timestamp_ns: int | None = None
        self._last_capture_timestamp_ns: int | None = None
        self._consecutive_failures = 0
        self.last_error: Exception | None = None
        self.recovery_attempts = 0
        self.recoveries = 0
        self.last_recovery_error: Exception | None = None

    @property
    def api_stability(self) -> str:
        """Release status of the explicitly enabled GPU capture contract."""
        return "experimental"

    @property
    def config(self) -> CameraConfig:
        """Resolved NV12/NVMM camera configuration."""
        return self._config

    @property
    def pipeline(self) -> str:
        """GStreamer pipeline used for the next open operation."""
        return self._pipeline

    @property
    def running(self) -> bool:
        """Whether the capture worker is active."""
        return self._running.is_set()

    @property
    def active_backend(self) -> str | None:
        """Name of the active NVMM backend."""
        with self._lifecycle_lock:
            if not self.running or self._backend is None:
                return None

            return self._backend.backend_name

    @property
    def frame_format(self) -> FrameFormat:
        """Explicit GPU output format."""
        return FrameFormat.NV12_NVMM

    @property
    def memory_type(self) -> MemoryType:
        """NVMM memory domain retained through the inference appsink."""
        return MemoryType.NVMM

    @property
    def frame_resolution(self) -> tuple[int, int]:
        """GPU frame width and height."""
        return self._config.output_width, self._config.output_height

    @property
    def preview_enabled(self) -> bool:
        """Whether the independent JPEG branch is part of the pipeline."""
        return self._config.enable_preview

    @property
    def video_config(self) -> HardwareVideoConfig | None:
        """Optional production video-encoder configuration."""
        return self._video_config

    @property
    def video_encoder_backend(self) -> str | None:
        """Resolved encoder backend, or the requested policy before startup."""
        if self._video_config is None:
            return None
        return self._resolved_video_encoder_backend or self._video_config.backend.value

    @property
    def encoded_stream_description(self) -> EncodedStreamDescription | None:
        """Newest negotiated encoded caps and codec parameter sets."""
        with self._lifecycle_lock:
            return self._encoded_stream_description

    @property
    def video_enabled(self) -> bool:
        """Whether the independent H.264/H.265 branch is configured."""
        return self._video_config is not None

    @property
    def overlay_error_policy(self) -> str:
        """Whether one overlay error preserves or fails production video."""
        return self._overlay_error_policy

    def overlay_diagnostics(self) -> dict[str, object]:
        """Return capture-side overlay policy and failure counters."""
        with self._lifecycle_lock:
            backend = self._backend
            failure = (
                None
                if backend is None
                else getattr(backend, "overlay_last_error", None)
            )
            return {
                "policy": self._overlay_error_policy,
                "failed_frames": (
                    0
                    if backend is None
                    else int(getattr(backend, "overlay_failed_frames", 0))
                ),
                "last_error": None if failure is None else str(failure),
            }

    @property
    def video_stats(self) -> VideoEncodeStats:
        """Recent video encode FPS/bitrate and cumulative output counts."""
        return self._video_publisher.stats(time.monotonic_ns())

    def set_video_overlay(
        self,
        renderer: VideoOverlayRenderer | None,
    ) -> None:
        """Select an NVMM overlay renderer before opening the camera pipeline.

        This permits constructing an ``InferenceConsumer`` from a camera
        subscription first, then wiring its results into a GPU renderer. The
        renderer lifecycle remains caller-owned.
        """
        with self._lifecycle_lock:
            if self.running:
                raise CameraConfigurationError(
                    "video overlay can be changed only while camera is stopped"
                )
            if renderer is not None:
                if self._video_config is None:
                    raise CameraConfigurationError(
                        "video overlay requires a hardware video or general "
                        "video encoder configuration"
                    )
                if not isinstance(renderer, VideoOverlayRenderer):
                    raise CameraConfigurationError(
                        "renderer must implement VideoOverlayRenderer"
                    )
                if renderer.memory_type is not MemoryType.NVMM:
                    raise CameraConfigurationError(
                        "production video overlay renderer must use NVMM"
                    )
            self._video_overlay = renderer
            self._rebuild_pipeline()

    def _encoder_pipeline(
        self,
        backend: VideoEncoderBackend | None = None,
    ) -> VideoEncoderPipeline | None:
        """Build the selected public encoder definition."""
        if self._video_config is None:
            return None
        if self._encoder_pipeline_factory is not None:
            definition = self._encoder_pipeline_factory(
                self._video_config,
                self._config.output_width,
                self._config.output_height,
                self._config.fps,
            )
            if not isinstance(definition, VideoEncoderPipeline):
                raise CameraConfigurationError(
                    "encoder_pipeline_factory must return VideoEncoderPipeline"
                )
            return definition
        return build_video_encoder_pipeline(
            self._video_config,
            self._config.output_width,
            self._config.output_height,
            self._config.fps,
            backend=backend,
        )

    def _rebuild_pipeline(
        self,
        backend: VideoEncoderBackend | None = None,
    ) -> None:
        """Regenerate the camera graph from public configuration only."""
        definition = self._encoder_pipeline(backend)
        factory = None if definition is None else lambda *_: definition
        self._pipeline = build_gpu_gstreamer_pipeline(
            sensor_id=self._config.sensor_id,
            capture_width=self._config.capture_width,
            capture_height=self._config.capture_height,
            output_width=self._config.output_width,
            output_height=self._config.output_height,
            framerate=self._config.fps,
            flip_method=self._config.flip_method,
            enable_preview=self._config.enable_preview,
            jpeg_quality=self._config.quality,
            video_config=self._video_config,
            enable_video_overlay=self._video_overlay is not None,
            encoder_pipeline_factory=factory,
            argus_properties=self._argus_properties,
        )
        if definition is not None:
            if (
                self._encoder_pipeline_factory is not None
                or backend is not None
                or self._video_config is not None
                and self._video_config.backend is not VideoEncoderBackend.AUTO
            ):
                self._resolved_video_encoder_backend = definition.backend
            else:
                self._resolved_video_encoder_backend = None
            self._encoded_stream_description = definition.description

    @property
    def gpu_frame_number(self) -> int:
        """Sequence of the newest borrowed GPU frame."""
        return self._gpu_publisher.frame_number

    @property
    def frame_available(self) -> bool:
        """Whether a GPU frame has been published."""
        return self._gpu_publisher.latest_frame is not None

    @property
    def frame_number(self) -> int:
        """Sequence of the newest JPEG preview frame."""
        return self._preview_publisher.frame_number

    @property
    def jpeg(self) -> bytes | None:
        """Newest independently encoded preview JPEG."""
        return self._preview_publisher.jpeg

    def latest_jpeg(self) -> bytes | None:
        """Return newest preview JPEG without affecting the GPU branch."""
        return self._preview_publisher.jpeg

    @property
    def frames_encoded(self) -> int:
        """Number of published encoded preview frames."""
        return self._preview_publisher.frames_encoded

    @property
    def last_frame_time(self) -> float | None:
        """Wall-clock time of the newest preview JPEG."""
        return self._preview_publisher.last_frame_time

    @property
    def pipeline_metrics(self) -> PipelineMetrics:
        """Immutable per-stage timing aggregates."""
        return self._metrics.snapshot()

    @property
    def consumer_dropped_frames(self) -> dict[str, int]:
        """Latest-frame drops reported per consumer."""
        return dict(self._metrics.consumer_drops())

    def record_stage_latency(
        self,
        stage: PipelineStage | str,
        duration_ns: int,
    ) -> None:
        """Record latency owned by an external inference consumer."""
        self._metrics.record_stage(stage, duration_ns)

    def record_consumer_drop(self, consumer: str, count: int = 1) -> None:
        """Record frames skipped by a named latest-frame consumer."""
        self._metrics.record_consumer_drop(consumer, count)

    def stats(self) -> CameraStats:
        """Return an immutable NVMM capture diagnostics snapshot."""
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

    def start(self) -> None:
        """Open both pipeline branches and start the latest-frame worker."""
        if not self._backend_available():
            raise CameraDependencyError(
                "GpuCamera requires PyGObject GStreamer on NVIDIA Jetson"
            )

        with self._lifecycle_lock:
            if self.running:
                return

            self._release_finished_capture()
            self._prepare_pipeline()
            backend = self._create_backend()

            try:
                backend.open()

            except CameraError:
                backend.close()
                raise

            except Exception as error:
                backend.close()
                raise CameraOpenError(
                    f"Could not open the NVMM camera backend: {error}"
                ) from error

            self._backend = backend
            self.last_error = None
            self.last_recovery_error = None
            self._consecutive_recovery_attempts = 0
            self._running.set()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="imx-gpu-camera-capture",
                daemon=True,
            )
            self._thread.start()

    def _backend_available(self) -> bool:
        """Whether the platform can create the configured GPU backend."""
        return GpuGStreamerCaptureBackend.available()

    def _create_backend(self) -> _GpuBackend:
        """Create one backend owning inference and preview branches."""
        return GpuGStreamerCaptureBackend(
            self._pipeline,
            self._config.output_width,
            self._config.output_height,
            enable_preview=self._config.enable_preview,
            video_config=self._video_config,
            video_overlay=self._video_overlay,
            stream_description=self._encoded_stream_description,
            video_encoder_backend=self._resolved_video_encoder_backend,
            overlay_error_policy=self._overlay_error_policy,
        )

    def _prepare_pipeline(self) -> None:
        """Resolve AUTO and report every missing plugin before parse_launch."""
        if not GpuGStreamerCaptureBackend.available():
            # Deterministic backend subclasses used by host tests do not need
            # a system GStreamer registry.
            return

        selected: VideoEncoderBackend | None = None
        definition: VideoEncoderPipeline | None = None

        if self._video_config is not None:
            if self._encoder_pipeline_factory is not None:
                definition = self._encoder_pipeline()

            else:
                selected = self._resolve_encoder_backend()
                definition = self._encoder_pipeline(selected)

        required = [
            "nvarguscamerasrc",
            "nvvidconv",
            "tee",
            "queue",
            "capsfilter",
            "appsink",
        ]

        if self._config.enable_preview:
            required.append("nvjpegenc")
        if self._video_overlay is not None:
            required.append("identity")

        if definition is not None:
            required.extend(definition.required_elements)

        missing = GpuGStreamerCaptureBackend.missing_elements(
            tuple(dict.fromkeys(required))
        )
        if missing:
            backend_name = "camera"

            if definition is not None:
                backend_name = f"encoder backend {definition.backend}"

            hint = (
                "; use backend=x264 or install the NVIDIA encoder plugin"
                if definition is not None and definition.backend == "nvenc"
                else ""
            )

            raise CameraDependencyError(
                f"{backend_name} unavailable; missing GStreamer element(s): "
                f"{', '.join(missing)}{hint}"
            )

        self._rebuild_pipeline(selected)

    def _resolve_encoder_backend(self) -> VideoEncoderBackend:
        """Resolve the configured backend from actual GStreamer factories."""
        if self._video_config is None:
            raise CameraConfigurationError("video encoder is disabled")

        requested = self._video_config.backend
        nvenc_element = (
            "nvv4l2h264enc"
            if self._video_config.codec is VideoCodec.H264
            else "nvv4l2h265enc"
        )
        nvenc_available = GpuGStreamerCaptureBackend.element_available(nvenc_element)
        x264_available = GpuGStreamerCaptureBackend.element_available("x264enc")

        if requested is VideoEncoderBackend.NVENC:
            if not nvenc_available:
                raise CameraDependencyError(
                    "encoder backend nvenc unavailable; use backend=x264 or "
                    f"install {nvenc_element}"
                )
            return requested

        if requested is VideoEncoderBackend.X264:
            if not x264_available:
                raise CameraDependencyError(
                    "encoder backend x264 unavailable; install x264enc"
                )

            return requested

        if nvenc_available:
            return VideoEncoderBackend.NVENC

        if self._video_config.codec is VideoCodec.H264 and x264_available:
            logger.warning(
                "%s is unavailable; production preview is using CPU x264",
                nvenc_element,
            )
            return VideoEncoderBackend.X264

        alternatives = (
            " and x264enc" if self._video_config.codec is VideoCodec.H264 else ""
        )
        raise CameraDependencyError(
            "encoder backend auto unavailable; missing "
            f"{nvenc_element}{alternatives}"
        )

    def _capture_loop(self) -> None:
        """Publish newest GPU and preview frames with bounded buffering."""
        consecutive_read_failures = 0
        while self.running:
            try:
                backend = self._backend

                if backend is None:
                    break

                success, backend_frame = backend.read()

                if not success or backend_frame is None:
                    consecutive_read_failures = self._record_drop(failed_read=True)
                    if (
                        consecutive_read_failures
                        >= self._recovery_policy.max_consecutive_read_failures
                    ):
                        raise CameraReadError(
                            "NVMM backend stopped producing GPU frames"
                        )
                    time.sleep(0.02)
                    continue

                consecutive_read_failures = 0
                frame = self._assign_sequence(backend_frame)

                self._record_capture(frame)
                self._gpu_publisher.publish(frame)
                self._frame_hub.publish(frame)

                if self._config.enable_preview:
                    encoder_started_ns = time.monotonic_ns()
                    jpeg = backend.read_preview()

                    if jpeg is not None:
                        published = self._preview_publisher.publish(jpeg)

                        if published:
                            self._metrics.record_stage(
                                PipelineStage.ENCODER,
                                time.monotonic_ns() - encoder_started_ns,
                            )

                        else:
                            self._metrics.record_consumer_drop("preview")

                if self._video_config is not None:
                    encoded_video = backend.read_video()

                    if encoded_video is not None:
                        if encoded_video.stream_description is not None:
                            with self._lifecycle_lock:
                                self._encoded_stream_description = (
                                    encoded_video.stream_description
                                )

                        self._video_publisher.publish(encoded_video)
                        self._video_hub.publish(encoded_video)

                self._metrics.record_stage(
                    PipelineStage.END_TO_END,
                    max(time.monotonic_ns() - frame.timestamp_ns, 0),
                )

            except Exception as error:
                self.last_error = error
                self._log_capture_failure(error)

                if _is_already_allocated_error(error):
                    self.last_recovery_error = error
                    self._running.clear()
                    break

                if not self._recover_backend():
                    if self.running:
                        recovery_error = self.last_recovery_error
                        self.last_error = (
                            recovery_error
                            if recovery_error is not None
                            and _is_already_allocated_error(recovery_error)
                            else CameraRecoveryError(
                                "NVMM camera recovery attempts were exhausted"
                            )
                        )

                    self._running.clear()

        self._gpu_publisher.notify_waiters()
        self._preview_publisher.notify_waiters()
        self._frame_hub.close()
        self._video_hub.close()

    def _assign_sequence(self, frame: GpuFrame) -> GpuFrame:
        """Assign a camera-lifetime sequence that survives backend recovery."""
        self._sequence += 1
        assigned = frame.retain(sequence=self._sequence)
        frame.release()
        return assigned

    def _new_frame_hub(self) -> LatestFrameHub[GpuFrame]:
        """Create subscriber slots with independent ref-counted GPU leases."""
        return LatestFrameHub[GpuFrame](
            self.record_consumer_drop,
            retain=lambda frame: frame.retain(),
            release=lambda frame: frame.release(),
        )

    def _record_capture(self, frame: GpuFrame) -> None:
        """Record one successful NVMM source read."""
        with self._stats_lock:
            self.frames_captured += 1
            self._last_frame_timestamp_ns = frame.timestamp_ns
            self._last_capture_timestamp_ns = frame.capture_timestamp_ns
            self._consecutive_failures = 0
            self._consecutive_recovery_attempts = 0
            self._capture_timestamps_ns.append(frame.timestamp_ns)
            self._prune_capture_timestamps(frame.timestamp_ns)
        self.last_error = None
        self.last_recovery_error = None
        self._last_capture_error_log_at = 0.0
        self._suppressed_capture_error_logs = 0

    def _record_drop(self, *, failed_read: bool = False) -> int:
        """Record one omitted frame or failed read."""
        with self._stats_lock:
            self.dropped_frames += 1

            if failed_read:
                self._consecutive_failures += 1

            return self._consecutive_failures

    def _prune_capture_timestamps(self, now_ns: int) -> None:
        """Discard capture timestamps older than the FPS window."""
        oldest_ns = now_ns - self.STATS_WINDOW_NS

        while (
            self._capture_timestamps_ns and self._capture_timestamps_ns[0] < oldest_ns
        ):
            self._capture_timestamps_ns.popleft()

    def _capture_fps(self) -> float:
        """Calculate recent successful GPU capture rate."""
        if len(self._capture_timestamps_ns) < 2:
            return 0.0

        elapsed_ns = self._capture_timestamps_ns[-1] - self._capture_timestamps_ns[0]

        if elapsed_ns <= 0:
            return 0.0

        return (len(self._capture_timestamps_ns) - 1) * 1_000_000_000 / elapsed_ns

    def _recover_backend(self) -> bool:
        """Recreate the full tee pipeline after an error in either branch."""
        while self.running:
            with self._stats_lock:
                if (
                    self._consecutive_recovery_attempts
                    >= self._recovery_policy.max_attempts
                ):
                    return False

                self._consecutive_recovery_attempts += 1
                attempt = self._consecutive_recovery_attempts
                self.recovery_attempts += 1

            if not self.running:
                return False

            delay = self._recovery_policy.initial_backoff * (2 ** (attempt - 1))
            if delay:
                time.sleep(delay)

            with self._lifecycle_lock:
                if not self.running:
                    return False

                backend: _GpuBackend | None = None
                try:
                    self._release_backend()
                    backend = self._create_backend()
                    backend.open()
                    self._backend = backend

                except Exception as error:
                    if backend is not None:
                        backend.close()
                    self.last_recovery_error = error
                    logger.warning(
                        "NVMM recovery attempt %s/%s failed: %s",
                        attempt,
                        self._recovery_policy.max_attempts,
                        error,
                    )
                    if _is_already_allocated_error(error):
                        return False
                    continue

            with self._stats_lock:
                self.recoveries += 1

            return True

        return False

    def _log_capture_failure(self, error: Exception) -> None:
        """Emit capture tracebacks at a bounded rate during recovery."""
        now = time.monotonic()
        if (
            self._last_capture_error_log_at != 0.0
            and now - self._last_capture_error_log_at < self.ERROR_LOG_INTERVAL_SECONDS
        ):
            self._suppressed_capture_error_logs += 1
            return

        suffix = (
            ""
            if self._suppressed_capture_error_logs == 0
            else " (%s similar errors suppressed)"
        )
        arguments: tuple[object, ...] = (
            ()
            if self._suppressed_capture_error_logs == 0
            else (self._suppressed_capture_error_logs,)
        )
        logger.error(
            "NVMM camera capture failed" + suffix,
            *arguments,
            exc_info=(type(error), error, error.__traceback__),
        )
        self._last_capture_error_log_at = now
        self._suppressed_capture_error_logs = 0

    def read(self, timeout: float | None = None) -> GpuFrame | None:
        """Return the newest borrowed NVMM frame without CPU conversion."""
        resolved_timeout = 2.0 if timeout is None else timeout
        if (
            isinstance(resolved_timeout, bool)
            or not isinstance(resolved_timeout, (int, float))
            or not isfinite(resolved_timeout)
            or resolved_timeout < 0
        ):
            raise CameraConfigurationError(
                "timeout must be a finite non-negative number"
            )

        if not self.running:
            raise CameraReadError("GPU camera is not running; call start() first")

        return self._gpu_publisher.wait_for_frame(
            -1,
            float(resolved_timeout),
            lambda: self.running,
        )

    def latest_frame(self) -> GpuFrame | None:
        """Return the newest borrowed NVMM frame immediately."""
        return self._gpu_publisher.latest_frame

    def subscribe_latest(self, name: str) -> LatestFrameSubscription[GpuFrame]:
        """Create one non-blocking borrowed NVMM slot for a named consumer.

        Each subscriber owns only a single latest-frame slot and an independent
        reference-counted lease. A slow GPU consumer skips unread frames but
        may finish processing the frame it already received. Direct consumers
        must call :meth:`GpuFrame.release` after processing; ``FrameConsumer``
        and ``InferenceConsumer`` release leases automatically.
        """
        with self._lifecycle_lock:
            if self._frame_hub.closed:
                if self.running:
                    raise RuntimeError("GPU frame subscriptions are closed")

                self._frame_hub = self._new_frame_hub()

            return self._frame_hub.subscribe(name)

    def subscribe_video(
        self,
        name: str,
    ) -> LatestFrameSubscription[EncodedVideoFrame]:
        """Create one latest encoded-access-unit slot for a transport worker."""
        if self._video_config is None:
            raise CameraConfigurationError(
                "video encoding is disabled; provide video_config to GpuCamera"
            )
        with self._lifecycle_lock:
            if self._video_hub.closed:
                if self.running:
                    raise RuntimeError("video subscriptions are closed")
                self._video_hub = LatestFrameHub[EncodedVideoFrame](
                    self.record_consumer_drop
                )
            return self._video_hub.subscribe(name)

    def wait_for_jpeg(
        self,
        previous_frame_number: int,
        timeout: float = 2.0,
    ) -> tuple[int, bytes | None]:
        """Wait for a JPEG from the independent hardware preview branch."""
        if timeout < 0:
            raise CameraConfigurationError("timeout must be non-negative")

        return self._preview_publisher.wait_for_jpeg(
            previous_frame_number,
            timeout,
            lambda: self.running,
        )

    def _release_finished_capture(self) -> None:
        """Release resources left by a completed worker."""
        if self._thread is not None and self._thread.is_alive():
            raise CameraOpenError("GPU camera capture thread is still stopping")

        self._release_backend()
        self._thread = None

    def _release_backend(self) -> None:
        """Close the one pipeline that owns both branches."""
        if self._backend is not None:
            self._backend.close()

        self._backend = None

    def stop(self) -> None:
        """Stop and close inference and preview branches together."""
        with self._lifecycle_lock:
            self._running.clear()
            self._gpu_publisher.notify_waiters()
            self._preview_publisher.notify_waiters()
            self._frame_hub.close()
            self._video_hub.close()
            thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)

            if thread.is_alive():
                logger.warning("GPU camera thread did not stop within 3 seconds")
                return

        with self._lifecycle_lock:
            self._release_backend()
            self._thread = None
            self._gpu_publisher.clear()
            self._preview_publisher.clear()
            self._video_publisher.clear()

    def __enter__(self) -> GpuCamera:
        """Start capture and return this camera."""
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Close both branches on context-manager exit."""
        self.stop()


def _is_already_allocated_error(error: BaseException) -> bool:
    """Recognize an Argus resource conflict through wrapped camera errors."""
    current: BaseException | None = error
    while current is not None:
        normalized = "".join(
            character for character in str(current).lower() if character.isalnum()
        )
        if "alreadyallocated" in normalized:
            return True
        current = current.__cause__ or current.__context__
    return False


__all__ = ["GpuCamera"]
