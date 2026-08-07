"""Real GStreamer H.264/RTP/decode smoke coverage when plugins are installed."""

from __future__ import annotations

import importlib
from typing import Any

import pytest


def _runtime() -> Any:
    """Load the system runtime or skip hosts without PyGObject."""
    try:
        gi = importlib.import_module("gi")
        gi.require_version("Gst", "1.0")
        gst = importlib.import_module("gi.repository.Gst")
    except (ImportError, ValueError):
        pytest.skip("system GStreamer PyGObject runtime is unavailable")

    gst.init(None)
    required = (
        "videotestsrc",
        "x264enc",
        "h264parse",
        "rtph264pay",
        "rtph264depay",
        "avdec_h264",
        "videoconvert",
        "appsink",
    )
    missing = [name for name in required if gst.ElementFactory.find(name) is None]
    if missing:
        pytest.skip(f"GStreamer roundtrip plugins unavailable: {', '.join(missing)}")
    return gst


@pytest.mark.integration
def test_real_x264_rtp_roundtrip_decodes_five_nonblack_frames() -> None:
    """Exercise real parser/payloader negotiation through an H.264 decoder."""
    gst = _runtime()
    pipeline = gst.parse_launch(
        "videotestsrc num-buffers=12 pattern=ball ! "
        "video/x-raw,width=320,height=240,framerate=30/1 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=5 "
        "byte-stream=true ! "
        "h264parse config-interval=-1 ! "
        "rtph264pay config-interval=-1 pt=96 ! "
        "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,"
        "payload=96,packetization-mode=(string)1 ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=RGB ! "
        "appsink name=decoded_sink sync=false"
    )
    sink = pipeline.get_by_name("decoded_sink")
    assert sink is not None
    assert pipeline.set_state(gst.State.PLAYING) != gst.StateChangeReturn.FAILURE

    decoded = 0
    try:
        for _ in range(12):
            sample = sink.emit("try-pull-sample", 2 * gst.SECOND)
            if sample is None:
                break
            buffer = sample.get_buffer()
            assert buffer is not None
            pixels = bytes(buffer.extract_dup(0, buffer.get_size()))
            assert any(pixels)
            decoded += 1
            if decoded >= 5:
                break
    finally:
        pipeline.set_state(gst.State.NULL)

    assert decoded >= 5
