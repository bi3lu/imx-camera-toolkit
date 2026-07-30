"""PyGObject GStreamer backend with live Argus element access."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from ..errors import CameraDependencyError, CameraOpenError
from .base import CaptureBackend

logger = logging.getLogger(__name__)

Gst: Any | None
np_module: Any | None

try:
    gi_module = importlib.import_module("gi")
    gi_module.require_version("Gst", "1.0")
    Gst = importlib.import_module("gi.repository.Gst")
    np_module = importlib.import_module("numpy")
    Gst.init(None)

except (ImportError, ValueError):
    Gst = None
    np_module = None


class GStreamerCaptureBackend(CaptureBackend):
    """Capture BGR frames while retaining the live Argus source element."""

    def __init__(self, pipeline: str, output_width: int, output_height: int) -> None:
        """Initialize a backend without parsing or opening the pipeline."""
        self._pipeline_description = pipeline
        self._output_width = output_width
        self._output_height = output_height
        self._pipeline: Any | None = None
        self._source: Any | None = None
        self._sink: Any | None = None

    @classmethod
    def available(cls) -> bool:
        """Return whether PyGObject GStreamer and NumPy are available."""
        return Gst is not None and np_module is not None

    @property
    def argus_source(self) -> Any | None:
        """Return the retained ``nvarguscamerasrc`` GObject."""
        return self._source

    def open(self) -> None:
        """Parse, start, and validate the configured GStreamer pipeline."""
        if Gst is None or np_module is None:
            raise CameraDependencyError("PyGObject GStreamer and NumPy are unavailable")

        try:
            pipeline = Gst.parse_launch(self._pipeline_description)
            source = pipeline.get_by_name("argus_source")
            sink = pipeline.get_by_name("camera_sink")

            if source is None or sink is None:
                pipeline.set_state(Gst.State.NULL)
                raise CameraOpenError(
                    "GStreamer pipeline is missing a named camera element"
                )

            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                pipeline.set_state(Gst.State.NULL)
                raise CameraOpenError("Could not start the IMX GStreamer pipeline")

            _, state, _ = pipeline.get_state(5 * Gst.SECOND)

            if state != Gst.State.PLAYING:
                pipeline.set_state(Gst.State.NULL)
                raise CameraOpenError(
                    "IMX GStreamer pipeline did not enter the playing state"
                )

        except Exception as error:
            if isinstance(error, (CameraDependencyError, CameraOpenError)):
                raise

            raise CameraOpenError(
                f"Could not create the IMX GStreamer pipeline: {error}"
            ) from error

        self._pipeline = pipeline
        self._source = source
        self._sink = sink

    def read(self) -> tuple[bool, Any | None]:
        """Pull one owned BGR frame from the configured appsink."""
        if self._sink is None or Gst is None or np_module is None:
            return False, None

        sample = self._sink.emit("try-pull-sample", Gst.SECOND // 5)

        if sample is None:
            return False, None

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)

        if not success:
            return False, None

        try:
            frame_size = self._output_width * self._output_height * 3

            if map_info.size < frame_size:
                logger.warning("GStreamer camera frame is smaller than expected")
                return False, None

            frame = np_module.frombuffer(
                map_info.data,
                dtype=np_module.uint8,
                count=frame_size,
            )
            return True, frame.reshape(
                self._output_height,
                self._output_width,
                3,
            ).copy()

        finally:
            buffer.unmap(map_info)

    def close(self) -> None:
        """Stop the pipeline and discard retained GStreamer elements."""
        if self._pipeline is not None and Gst is not None:
            self._pipeline.set_state(Gst.State.NULL)

        self._pipeline = None
        self._source = None
        self._sink = None
