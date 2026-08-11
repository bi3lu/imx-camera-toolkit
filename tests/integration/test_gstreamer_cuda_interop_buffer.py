"""Regression coverage for JetPack's overridden Gst.Buffer Python type."""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

from imx_camera_toolkit._internal.inference.interop import NativeCudaInterop
from imx_camera_toolkit.testing import mock_gpu_frame


@pytest.mark.integration
def test_native_interop_accepts_real_overridden_gst_buffer() -> None:
    """JetPack's gi.overrides.Gst.Buffer must pass the Python type boundary."""
    try:
        gi = importlib.import_module("gi")
        gi.require_version("Gst", "1.0")
        gst = importlib.import_module("gi.repository.Gst")

    except (ImportError, ValueError):
        pytest.skip("system GStreamer PyGObject runtime is unavailable")

    gst.init(None)
    payload = gst.Buffer.new()
    calls: list[object] = []
    native = ModuleType("fake_cuda_interop")
    native.NvmmSurface = lambda buffer, width, height: calls.append(buffer)  # type: ignore[attr-defined]

    NativeCudaInterop(native).import_frame(mock_gpu_frame(payload))

    assert type(payload).__module__ == "gi.overrides.Gst"
    assert calls == [payload]
