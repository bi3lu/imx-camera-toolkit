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
from ..models import FrameFormat, GpuBufferHandle, GpuFrame, MemoryType

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
    ) -> None:
        """Initialize an unopened pipeline backend."""
        self._pipeline_description = pipeline
        self._output_width = output_width
        self._output_height = output_height
        self._enable_preview = enable_preview
        self._pipeline: Any | None = None
        self._source: Any | None = None
        self._gpu_sink: Any | None = None
        self._preview_sink: Any | None = None
        self._sequence = 0

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

            if source is None or gpu_sink is None:
                pipeline.set_state(Gst.State.NULL)
                raise CameraOpenError(
                    "GPU pipeline is missing its Argus source or NVMM appsink"
                )

            if self._enable_preview and preview_sink is None:
                pipeline.set_state(Gst.State.NULL)
                raise CameraOpenError("GPU pipeline is missing its preview appsink")

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

        self._pipeline = None
        self._source = None
        self._gpu_sink = None
        self._preview_sink = None
