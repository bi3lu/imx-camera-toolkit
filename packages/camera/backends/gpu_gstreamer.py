"""NVMM-preserving GStreamer backend for borrowed GPU frames."""

from __future__ import annotations

import importlib
import time
from typing import Any

from ..errors import (
    CameraDependencyError,
    CameraOpenError,
    CameraReadError,
)
from ..models import (
    EncodedVideoFrame,
    FrameFormat,
    GpuBufferHandle,
    GpuFrame,
    HardwareVideoConfig,
    MemoryType,
    VideoOverlayRenderer,
)

Gst: Any | None

try:
    gi_module = importlib.import_module("gi")
    gi_module.require_version("Gst", "1.0")
    Gst = importlib.import_module("gi.repository.Gst")
    Gst.init(None)

except (ImportError, ValueError):
    Gst = None


class GpuGStreamerCaptureBackend:
    """Pull borrowed NV12/NVMM buffers without mapping them into host memory."""

    def __init__(
        self,
        pipeline: str,
        output_width: int,
        output_height: int,
        *,
        enable_preview: bool,
        video_config: HardwareVideoConfig | None = None,
        video_overlay: VideoOverlayRenderer | None = None,
    ) -> None:
        """Initialize an unopened pipeline backend."""
        self._pipeline_description = pipeline
        self._output_width = output_width
        self._output_height = output_height
        self._enable_preview = enable_preview
        self._video_config = video_config
        self._video_overlay = video_overlay
        self._pipeline: Any | None = None
        self._source: Any | None = None
        self._gpu_sink: Any | None = None
        self._preview_sink: Any | None = None
        self._video_sink: Any | None = None
        self._overlay_pad: Any | None = None
        self._overlay_probe_id: int | None = None
        self._overlay_error: Exception | None = None
        self._sequence = 0
        self._video_sequence = 0
        self._overlay_sequence = 0

    @classmethod
    def available(cls) -> bool:
        """Whether PyGObject GStreamer is importable without requiring NumPy."""
        return Gst is not None

    @property
    def backend_name(self) -> str:
        """Stable backend name reported through health diagnostics."""
        return "gstreamer-nvmm"

    @property
    def argus_source(self) -> Any | None:
        """Return the live ``nvarguscamerasrc`` element."""
        return self._source

    def open(self) -> None:
        """Parse and start both branches as one recoverable pipeline."""
        if Gst is None:
            raise CameraDependencyError("PyGObject GStreamer is unavailable")

        pipeline: Any | None = None
        try:
            pipeline = Gst.parse_launch(self._pipeline_description)
            source = pipeline.get_by_name("argus_source")
            gpu_sink = pipeline.get_by_name("gpu_sink")
            preview_sink = pipeline.get_by_name("preview_sink")
            video_sink = pipeline.get_by_name("video_sink")
            overlay_hook = pipeline.get_by_name("video_overlay_hook")

            if source is None or gpu_sink is None:
                pipeline.set_state(Gst.State.NULL)
                raise CameraOpenError(
                    "GPU pipeline is missing its Argus source or NVMM appsink"
                )

            if self._enable_preview and preview_sink is None:
                pipeline.set_state(Gst.State.NULL)
                raise CameraOpenError("GPU pipeline is missing its preview appsink")

            if self._video_config is not None and video_sink is None:
                pipeline.set_state(Gst.State.NULL)
                raise CameraOpenError("GPU pipeline is missing its video appsink")

            if self._video_overlay is not None:
                if overlay_hook is None:
                    pipeline.set_state(Gst.State.NULL)
                    raise CameraOpenError("GPU pipeline is missing its overlay hook")

                overlay_pad = overlay_hook.get_static_pad("src")

                if overlay_pad is None:
                    pipeline.set_state(Gst.State.NULL)
                    raise CameraOpenError("GPU overlay hook has no source pad")

                self._overlay_pad = overlay_pad
                self._overlay_probe_id = overlay_pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    self._render_video_overlay,
                )

            state_change = pipeline.set_state(Gst.State.PLAYING)

            if state_change == Gst.StateChangeReturn.FAILURE:
                pipeline.set_state(Gst.State.NULL)
                raise CameraOpenError("Could not start the NVMM camera pipeline")

            _, state, _ = pipeline.get_state(5 * Gst.SECOND)

            if state != Gst.State.PLAYING:
                pipeline.set_state(Gst.State.NULL)
                raise CameraOpenError(
                    "NVMM camera pipeline did not enter the playing state"
                )

        except Exception as error:
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)

            if isinstance(error, (CameraDependencyError, CameraOpenError)):
                raise

            raise CameraOpenError(
                f"Could not create the NVMM GStreamer pipeline: {error}"
            ) from error

        self._pipeline = pipeline
        self._source = source
        self._gpu_sink = gpu_sink
        self._preview_sink = preview_sink
        self._video_sink = video_sink

    def read(self) -> tuple[bool, GpuFrame | None]:
        """Pull one NVMM sample and return its checked borrowed Gst buffer."""
        if self._gpu_sink is None or Gst is None:
            return False, None

        sample = self._gpu_sink.emit("try-pull-sample", Gst.SECOND // 5)
        if sample is None:
            self._raise_pipeline_error()
            return False, None

        caps = sample.get_caps()
        features = caps.get_features(0) if caps is not None else None

        if features is None or not features.contains("memory:NVMM"):
            raise CameraReadError("GPU appsink produced a non-NVMM buffer")

        structure = caps.get_structure(0)
        pixel_format = structure.get_string("format")

        if pixel_format != "NV12":
            raise CameraReadError("GPU appsink produced a non-NV12 buffer")

        buffer = sample.get_buffer()

        if buffer is None:
            raise CameraReadError("GPU appsink sample has no Gst.Buffer")

        capture_timestamp_ns = (
            None if buffer.pts == Gst.CLOCK_TIME_NONE else int(buffer.pts)
        )
        self._sequence += 1
        return True, GpuFrame(
            sequence=self._sequence,
            timestamp_ns=time.monotonic_ns(),
            capture_timestamp_ns=capture_timestamp_ns,
            width=self._output_width,
            height=self._output_height,
            format=FrameFormat.NV12_NVMM,
            memory_type=MemoryType.NVMM,
            buffer=GpuBufferHandle(buffer, owner=sample),
        )

    def read_preview(self) -> bytes | None:
        """Pull newest encoded JPEG bytes from the independent preview branch."""
        if self._preview_sink is None or Gst is None:
            return None

        sample = self._preview_sink.emit("try-pull-sample", 0)

        if sample is None:
            self._raise_pipeline_error()
            return None

        buffer = sample.get_buffer()

        if buffer is None:
            raise CameraReadError("preview appsink sample has no Gst.Buffer")

        return bytes(buffer.extract_dup(0, buffer.get_size()))

    def read_video(self) -> EncodedVideoFrame | None:
        """Pull one newest hardware-encoded access unit without raw pixels."""
        if self._overlay_error is not None:
            error = self._overlay_error
            self._overlay_error = None
            raise CameraReadError(f"GPU video overlay failed: {error}") from error

        if self._video_sink is None or Gst is None or self._video_config is None:
            return None

        sample = self._video_sink.emit("try-pull-sample", 0)

        if sample is None:
            self._raise_pipeline_error()
            return None

        buffer = sample.get_buffer()

        if buffer is None:
            raise CameraReadError("video appsink sample has no Gst.Buffer")

        self._video_sequence += 1
        pts_ns = None if buffer.pts == Gst.CLOCK_TIME_NONE else int(buffer.pts)
        duration_ns = (
            None if buffer.duration == Gst.CLOCK_TIME_NONE else int(buffer.duration)
        )
        return EncodedVideoFrame(
            sequence=self._video_sequence,
            timestamp_ns=time.monotonic_ns(),
            codec=self._video_config.codec,
            data=bytes(buffer.extract_dup(0, buffer.get_size())),
            keyframe=not buffer.has_flags(Gst.BufferFlags.DELTA_UNIT),
            pts_ns=pts_ns,
            duration_ns=duration_ns,
        )

    def _render_video_overlay(self, _: object, info: object) -> object:
        """Run an injected GPU renderer on the branch-isolated NVMM surface."""
        if Gst is None or self._video_overlay is None:
            return Gst.PadProbeReturn.OK if Gst is not None else 0

        buffer = info.get_buffer()  # type: ignore[attr-defined]

        if buffer is None:
            return Gst.PadProbeReturn.DROP

        self._overlay_sequence += 1
        frame = GpuFrame(
            sequence=self._overlay_sequence,
            timestamp_ns=time.monotonic_ns(),
            capture_timestamp_ns=(
                None if buffer.pts == Gst.CLOCK_TIME_NONE else int(buffer.pts)
            ),
            width=self._output_width,
            height=self._output_height,
            format=FrameFormat.NV12_NVMM,
            memory_type=MemoryType.NVMM,
            buffer=GpuBufferHandle(buffer),
        )
        try:
            self._video_overlay.render(frame)

        except Exception as error:
            self._overlay_error = error
            return Gst.PadProbeReturn.DROP

        finally:
            frame.invalidate()

        return Gst.PadProbeReturn.OK

    def _raise_pipeline_error(self) -> None:
        """Raise a read error when either pipeline branch reports ERROR or EOS."""
        if self._pipeline is None or Gst is None:
            return

        bus = self._pipeline.get_bus()
        message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)

        if message is None:
            return

        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            detail = debug or str(error)
            raise CameraReadError(f"NVMM pipeline failed: {detail}")

        raise CameraReadError("NVMM pipeline reached end of stream")

    def close(self) -> None:
        """Stop the shared pipeline, closing inference and preview together."""
        if self._pipeline is not None and Gst is not None:
            self._pipeline.set_state(Gst.State.NULL)

        if self._overlay_pad is not None and self._overlay_probe_id is not None:
            self._overlay_pad.remove_probe(self._overlay_probe_id)

        self._pipeline = None
        self._source = None
        self._gpu_sink = None
        self._preview_sink = None
        self._video_sink = None
        self._overlay_pad = None
        self._overlay_probe_id = None
        self._overlay_error = None
