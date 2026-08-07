"""Contract tests for model-agnostic CPU and GPU frame consumers."""

from __future__ import annotations

import pytest

from imx_camera_toolkit import (
    Frame,
    FrameFormat,
    GpuBufferHandle,
    GpuFrame,
    GpuFrameExpiredError,
    MemoryType,
)
from imx_camera_toolkit.frames import CaptureFrame
from imx_camera_toolkit.testing import (
    MockFrameSource,
    mock_cpu_frame,
    mock_gpu_frame,
)


def _consume(frame: CaptureFrame) -> tuple[FrameFormat, object]:
    """Example consumer using public contracts only."""
    if isinstance(frame, GpuFrame):
        return frame.format, frame.payload()
    return frame.output_format, frame.image


def test_cpu_frame_keeps_legacy_bgr_payload_and_adds_explicit_format() -> None:
    """The existing Frame contract must remain BGR/CPU compatible."""
    image = bytearray(b"bgr")
    frame = mock_cpu_frame(image, width=2, height=1, timestamp_ns=10)

    assert isinstance(frame, Frame)
    assert frame.image is image
    assert frame.format == "BGR"
    assert frame.output_format is FrameFormat.BGR_CPU
    assert frame.memory_type is MemoryType.CPU
    assert _consume(frame) == (FrameFormat.BGR_CPU, image)


def test_gpu_frame_exposes_only_a_borrowed_nvmm_handle() -> None:
    """GPU consumers receive no implicit NumPy/CPU image payload."""
    handle = object()
    frame = mock_gpu_frame(handle, width=1920, height=1080, timestamp_ns=20)

    assert frame.format is FrameFormat.NV12_NVMM
    assert frame.output_format is FrameFormat.NV12_NVMM
    assert frame.memory_type is MemoryType.NVMM
    assert not hasattr(frame, "image")
    assert frame.payload() is handle
    assert _consume(frame) == (FrameFormat.NV12_NVMM, handle)


def test_gpu_frame_requires_exactly_one_supported_payload() -> None:
    """Ambiguous or missing GPU payload representations are rejected."""
    with pytest.raises(ValueError, match="exactly one"):
        GpuFrame(
            sequence=1,
            timestamp_ns=1,
            width=1,
            height=1,
            format=FrameFormat.NV12_NVMM,
            memory_type=MemoryType.NVMM,
        )

    with pytest.raises(ValueError, match="exactly one"):
        GpuFrame(
            sequence=1,
            timestamp_ns=1,
            width=1,
            height=1,
            format=FrameFormat.NV12_NVMM,
            memory_type=MemoryType.NVMM,
            dmabuf_fd=3,
            buffer=GpuBufferHandle(object()),
        )


def test_latest_gpu_frame_invalidates_the_previous_borrowed_lease() -> None:
    """The mock source enforces latest-frame lifetime without a queue."""
    source = MockFrameSource()
    first = mock_gpu_frame(object(), sequence=1)
    second = mock_gpu_frame(object(), sequence=2)

    source.publish(first)
    assert source.read(timeout=0) is first
    source.publish(second)

    assert first.valid is False

    with pytest.raises(GpuFrameExpiredError, match="expired"):
        first.payload()

    assert source.read(timeout=0) is second
    assert second.valid is True
