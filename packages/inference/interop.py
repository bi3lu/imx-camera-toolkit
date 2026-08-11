"""Checked Python boundary for the optional CUDA interoperability module."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Protocol

from packages.camera.models import GpuFrame

from .errors import CudaInteropError, InferenceDependencyError


class CudaBuffer(Protocol):
    """Runner-owned device allocation exposed to TensorRT by pointer."""

    @property
    def pointer(self) -> int:
        """CUDA device address."""
        ...

    @property
    def size(self) -> int:
        """Allocation size in bytes."""
        ...

    def copy_to_host(self, stream: CudaStream) -> bytes:
        """Copy the allocation after queued GPU work completes."""
        ...


class CudaStream(Protocol):
    """Non-blocking CUDA stream shared with TensorRT."""

    @property
    def handle(self) -> int:
        """Native ``cudaStream_t`` address."""
        ...

    def synchronize(self) -> None:
        """Wait for all work submitted to this stream."""
        ...


class NvmmSurface(Protocol):
    """Scoped CUDA registration of one borrowed NvBufSurface."""


class InteropRuntime(Protocol):
    """Operations required by :class:`TensorRTRunner`."""

    def compute_capability(self) -> tuple[int, int]:
        """Return active CUDA device major/minor capability."""
        ...

    def create_stream(self) -> CudaStream:
        """Create a non-blocking CUDA stream."""
        ...

    def allocate(self, size: int) -> CudaBuffer:
        """Allocate runner-owned device memory."""
        ...

    def import_frame(self, frame: GpuFrame) -> NvmmSurface:
        """Register a valid borrowed NVMM frame with CUDA."""
        ...

    def preprocess_nv12(
        self,
        surface: NvmmSurface,
        destination: CudaBuffer,
        *,
        width: int,
        height: int,
        channel_order: str,
        scale: float,
        mean: tuple[float, float, float],
        standard_deviation: tuple[float, float, float],
        resize_mode: str,
        padding_value: tuple[float, float, float],
        stream: CudaStream,
    ) -> None:
        """Convert and resize NV12 directly into a device input tensor."""
        ...


class NativeCudaInterop:
    """Thin checked facade over the pybind11 NvBufSurface/CUDA extension."""

    def __init__(
        self,
        native_module: ModuleType | None = None,
        *,
        gstreamer_module: ModuleType | None = None,
    ) -> None:
        """Load the optional module only when TensorRT capture is requested."""
        if native_module is None:
            try:
                native_module = importlib.import_module(
                    "packages.inference._cuda_interop"
                )

            except ImportError as error:
                raise InferenceDependencyError(
                    "CUDA interop extension is unavailable; install the tensorrt "
                    "extra and build native/CMakeLists.txt on the target Jetson"
                ) from error

        self._native = native_module
        self._gst = gstreamer_module

    def _load_gstreamer(self) -> Any:
        """Load the canonical Gst namespace used by PyGObject overrides."""
        if self._gst is None:
            try:
                gi_module = importlib.import_module("gi")
                gi_module.require_version("Gst", "1.0")
                self._gst = importlib.import_module("gi.repository.Gst")

            except (ImportError, ValueError) as error:
                raise InferenceDependencyError(
                    "PyGObject GStreamer bindings are unavailable"
                ) from error

        return self._gst

    def compute_capability(self) -> tuple[int, int]:
        """Return active CUDA device capability from the runtime API."""
        major, minor = self._native.compute_capability()
        return int(major), int(minor)

    def create_stream(self) -> CudaStream:
        """Create the stream later passed to ``execute_async_v3``."""
        return self._native.CudaStream()  # type: ignore[no-any-return]

    def allocate(self, size: int) -> CudaBuffer:
        """Allocate a device buffer without creating a host image."""
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise CudaInteropError("device allocation size must be positive")

        return self._native.DeviceBuffer(size)  # type: ignore[no-any-return]

    def import_frame(self, frame: GpuFrame) -> NvmmSurface:
        """Retain and register the opaque Gst.Buffer backing ``frame``."""
        if not isinstance(frame, GpuFrame):
            raise TypeError("frame must be a GpuFrame")

        payload = frame.payload()
        gst = self._load_gstreamer()

        if not isinstance(payload, gst.Buffer):
            raise CudaInteropError(
                "CUDA interop requires GpuFrame backed by Gst.Buffer"
            )

        try:
            return self._native.NvmmSurface(  # type: ignore[no-any-return]
                payload,
                frame.width,
                frame.height,
            )

        except (RuntimeError, ValueError) as error:
            raise CudaInteropError(f"could not import NVMM frame: {error}") from error

    def preprocess_nv12(
        self,
        surface: NvmmSurface,
        destination: CudaBuffer,
        *,
        width: int,
        height: int,
        channel_order: str,
        scale: float,
        mean: tuple[float, float, float],
        standard_deviation: tuple[float, float, float],
        resize_mode: str = "stretch",
        padding_value: tuple[float, float, float] = (114.0, 114.0, 114.0),
        stream: CudaStream,
    ) -> None:
        """Launch NV12-to-NCHW conversion on the TensorRT CUDA stream."""
        try:
            self._native.preprocess_nv12(
                surface,
                destination,
                width,
                height,
                channel_order,
                scale,
                mean,
                standard_deviation,
                resize_mode,
                padding_value,
                stream,
            )

        except (RuntimeError, ValueError) as error:
            raise CudaInteropError(f"NV12 preprocessing failed: {error}") from error

    def draw_nv12_rectangle(
        self,
        surface: NvmmSurface,
        *,
        left: int,
        top: int,
        width: int,
        height: int,
        thickness: int,
        yuv: tuple[int, int, int],
        stream: CudaStream,
    ) -> None:
        """Draw one rectangle directly into an isolated NVMM surface."""
        try:
            self._native.draw_nv12_rectangle(
                surface,
                left,
                top,
                width,
                height,
                thickness,
                yuv[0],
                yuv[1],
                yuv[2],
                stream,
            )

        except (RuntimeError, ValueError) as error:
            raise CudaInteropError(f"NV12 overlay failed: {error}") from error
