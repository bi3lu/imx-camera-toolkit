"""CUDA-side rectangle overlays for isolated production preview surfaces."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite

from packages.camera.models import GpuFrame, MemoryType
from packages.consumers import InferenceResultSource
from packages.inference import InferenceResult
from packages.inference.interop import CudaStream, NativeCudaInterop


@dataclass(frozen=True, slots=True)
class OverlayRectangle:
    """Model-neutral rectangle rendered into NV12 by a CUDA kernel."""

    left: int
    top: int
    width: int
    height: int
    color_rgb: tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2

    def __post_init__(self) -> None:
        """Validate geometry and byte RGB components."""
        for name in ("left", "top", "width", "height", "thickness"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.left < 0 or self.top < 0:
            raise ValueError("rectangle origin must be non-negative")
        if self.width <= 0 or self.height <= 0 or self.thickness <= 0:
            raise ValueError("rectangle size and thickness must be positive")
        if len(self.color_rgb) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in self.color_rgb
        ):
            raise ValueError("color_rgb must contain three byte values")


RectangleMapper = Callable[[InferenceResult], Sequence[OverlayRectangle]]


def _default_mapper(result: InferenceResult) -> tuple[OverlayRectangle, ...]:
    """Select already-normalized rectangles from opaque result overlays."""
    return tuple(
        overlay for overlay in result.overlays if isinstance(overlay, OverlayRectangle)
    )


def _rgb_to_limited_bt709(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert one RGB color to limited-range BT.709 NV12 values."""
    red, green, blue = color
    y = 16 + 0.183 * red + 0.614 * green + 0.062 * blue
    u = 128 - 0.101 * red - 0.339 * green + 0.439 * blue
    v = 128 + 0.439 * red - 0.399 * green - 0.040 * blue
    return tuple(max(0, min(round(value), 255)) for value in (y, u, v))  # type: ignore[return-value]


class CudaOverlayRenderer:
    """Draw latest inference rectangles on a dedicated CUDA stream.

    ``GpuCamera`` inserts a device-to-device ``nvvidconv`` before invoking this
    renderer, so the in-place overlay surface is isolated from the TensorRT
    inference branch and remains NVMM until any encoder-specific conversion.
    """

    def __init__(
        self,
        inference: InferenceResultSource,
        mapper: RectangleMapper | None = None,
        *,
        max_result_age_seconds: float = 1.0,
        interop: NativeCudaInterop | None = None,
    ) -> None:
        """Create a CUDA stream without prescribing a model output schema."""
        if mapper is not None and not callable(mapper):
            raise TypeError("mapper must be callable or None")
        if (
            isinstance(max_result_age_seconds, bool)
            or not isinstance(max_result_age_seconds, (int, float))
            or not isfinite(max_result_age_seconds)
            or max_result_age_seconds <= 0
        ):
            raise ValueError("max_result_age_seconds must be finite and positive")
        self._inference = inference
        self._mapper = mapper or _default_mapper
        self._max_age_ns = int(max_result_age_seconds * 1_000_000_000)
        self._interop = interop or NativeCudaInterop()
        self._stream: CudaStream | None = self._interop.create_stream()
        self._metrics_lock = threading.Lock()
        self._rendered_frames = 0
        self._empty_results = 0
        self._stale_results = 0
        self._failed_frames = 0
        self._last_error: Exception | None = None

    @property
    def memory_type(self) -> MemoryType:
        """The renderer never requests a host BGR image."""
        return MemoryType.NVMM

    @property
    def rendered_frames(self) -> int:
        """Frames on which at least one overlay was drawn."""
        with self._metrics_lock:
            return self._rendered_frames

    @property
    def empty_results(self) -> int:
        """Frames without a current result or drawable overlay."""
        with self._metrics_lock:
            return self._empty_results

    @property
    def stale_results(self) -> int:
        """Frames skipped because the newest inference result was too old."""
        with self._metrics_lock:
            return self._stale_results

    @property
    def failed_frames(self) -> int:
        """Frames whose mapper, CUDA import, draw, or synchronization failed."""
        with self._metrics_lock:
            return self._failed_frames

    @property
    def last_error(self) -> Exception | None:
        """Most recent historical renderer failure."""
        with self._metrics_lock:
            return self._last_error

    def health(self) -> dict[str, object]:
        """Return renderer metrics suitable for a preview health provider."""
        with self._metrics_lock:
            return {
                "healthy": self._last_error is None,
                "rendered_frames": self._rendered_frames,
                "empty_results": self._empty_results,
                "stale_results": self._stale_results,
                "failed_frames": self._failed_frames,
                "last_error": (
                    None if self._last_error is None else str(self._last_error)
                ),
            }

    def render(self, frame: GpuFrame) -> None:
        """Draw the newest non-stale result and synchronize before encode."""
        try:
            result = self._inference.latest_result
            stream = self._stream

            if result is None:
                with self._metrics_lock:
                    self._empty_results += 1

                return

            if stream is None:
                raise RuntimeError("CUDA overlay renderer is closed")

            if time.monotonic_ns() - result.frame_timestamp_ns > self._max_age_ns:
                with self._metrics_lock:
                    self._stale_results += 1

                return

            rectangles = tuple(self._mapper(result))

            if not rectangles:
                with self._metrics_lock:
                    self._empty_results += 1

                return

            surface = self._interop.import_frame(frame)

            for rectangle in rectangles:
                if not isinstance(rectangle, OverlayRectangle):
                    raise TypeError("mapper must return OverlayRectangle values")

                self._interop.draw_nv12_rectangle(
                    surface,
                    left=rectangle.left,
                    top=rectangle.top,
                    width=rectangle.width,
                    height=rectangle.height,
                    thickness=rectangle.thickness,
                    yuv=_rgb_to_limited_bt709(rectangle.color_rgb),
                    stream=stream,
                )
            stream.synchronize()

        except Exception as error:
            with self._metrics_lock:
                self._failed_frames += 1
                self._last_error = error

            raise

        with self._metrics_lock:
            self._rendered_frames += 1

    def close(self) -> None:
        """Synchronize and release the renderer-owned CUDA stream."""
        if self._stream is not None:
            self._stream.synchronize()

        self._stream = None


__all__ = ["CudaOverlayRenderer", "OverlayRectangle", "RectangleMapper"]
