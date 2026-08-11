"""End-to-end WebRTC coverage using two real GStreamer peers."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from imx_camera_toolkit import (
    EncodedStreamDescription,
    EncodedVideoFrame,
    VideoCodec,
    VideoEncoderConfig,
)
from imx_camera_toolkit._internal.camera.backends.gpu_gstreamer import (
    _h264_parameter_sets,
)
from imx_camera_toolkit._internal.production_preview.errors import (
    ProductionPreviewDependencyError,
)
from imx_camera_toolkit._internal.production_preview.metrics import (
    ClientMetricsRegistry,
)
from imx_camera_toolkit._internal.production_preview.runtime import (
    load_gstreamer_runtime,
)
from imx_camera_toolkit._internal.production_preview.transport import WebRTCPeer
from imx_camera_toolkit.consumers import LatestFrameHub
from imx_camera_toolkit.production_preview import ProductionPreviewConfig

HUGE_PTS_OFFSET_NS = 7 * 24 * 60 * 60 * 1_000_000_000


def _runtime() -> Any:
    """Load WebRTC and decoder plugins or skip an unsupported host."""
    try:
        runtime = load_gstreamer_runtime(
            webrtc=True,
            required_elements=(
                "videotestsrc",
                "x264enc",
                "rtph264depay",
                "avdec_h264",
                "videoconvert",
                "appsink",
            ),
        )
    except ProductionPreviewDependencyError as error:
        pytest.skip(str(error))
    return runtime


def _encoded_test_frames(
    runtime: Any,
) -> tuple[tuple[EncodedVideoFrame, ...], EncodedStreamDescription]:
    """Encode deterministic access units before creating the WebRTC peers."""
    gst = runtime.Gst
    pipeline = gst.parse_launch(
        "videotestsrc num-buffers=30 pattern=ball ! "
        "video/x-raw,width=320,height=240,framerate=30/1 ! "
        "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=10 "
        "byte-stream=true aud=true ! "
        "h264parse config-interval=-1 ! "
        "video/x-h264,stream-format=byte-stream,alignment=au ! "
        "appsink name=encoded_sink sync=false"
    )
    sink = pipeline.get_by_name("encoded_sink")
    assert sink is not None
    assert pipeline.set_state(gst.State.PLAYING) != gst.StateChangeReturn.FAILURE

    frames: list[EncodedVideoFrame] = []
    sps: bytes | None = None
    pps: bytes | None = None
    profile: str | None = None
    level: str | None = None
    try:
        for sequence in range(1, 31):
            sample = sink.emit("try-pull-sample", 2 * gst.SECOND)
            assert sample is not None
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            profile = structure.get_string("profile") or profile
            level = structure.get_string("level") or level
            buffer = sample.get_buffer()
            assert buffer is not None
            data = bytes(buffer.extract_dup(0, buffer.get_size()))
            found_sps, found_pps = _h264_parameter_sets(data)
            sps = found_sps or sps
            pps = found_pps or pps
            pts_ns = int(buffer.pts)
            dts_ns = pts_ns if buffer.dts == gst.CLOCK_TIME_NONE else int(buffer.dts)
            duration_ns = (
                None if buffer.duration == gst.CLOCK_TIME_NONE else int(buffer.duration)
            )
            frames.append(
                EncodedVideoFrame(
                    sequence=sequence,
                    timestamp_ns=time.monotonic_ns(),
                    codec=VideoCodec.H264,
                    data=data,
                    keyframe=not buffer.has_flags(gst.BufferFlags.DELTA_UNIT),
                    pts_ns=HUGE_PTS_OFFSET_NS + pts_ns,
                    dts_ns=HUGE_PTS_OFFSET_NS + dts_ns,
                    duration_ns=duration_ns,
                )
            )
    finally:
        pipeline.set_state(gst.State.NULL)

    assert sps is not None
    assert pps is not None
    return tuple(frames), EncodedStreamDescription(
        codec=VideoCodec.H264,
        profile=profile,
        level=level,
        width=320,
        height=240,
        fps=30,
        sps=sps,
        pps=pps,
    )


def _set_remote_offer_and_create_answer(
    runtime: Any,
    receiver: Any,
    offer_sdp: str,
) -> str:
    """Install the sender offer and return the receiver answer SDP."""
    gst = runtime.Gst
    result, message = runtime.GstSdp.sdp_message_new_from_text(offer_sdp)
    assert result == runtime.GstSdp.SDPResult.OK
    assert message is not None
    offer = runtime.GstWebRTC.WebRTCSessionDescription.new(
        runtime.GstWebRTC.WebRTCSDPType.OFFER,
        message,
    )
    promise = gst.Promise.new()
    receiver.emit("set-remote-description", offer, promise)
    assert promise.wait() == gst.PromiseResult.REPLIED

    promise = gst.Promise.new()
    receiver.emit("create-answer", None, promise)
    assert promise.wait() == gst.PromiseResult.REPLIED
    reply = promise.get_reply()
    answer = None if reply is None else reply.get_value("answer")
    assert answer is not None

    promise = gst.Promise.new()
    receiver.emit("set-local-description", answer, promise)
    assert promise.wait() == gst.PromiseResult.REPLIED
    return str(answer.sdp.as_text())


def _exchange_candidates(
    sender: WebRTCPeer,
    receiver: Any,
    receiver_candidates: list[tuple[int, str]],
    receiver_candidates_lock: threading.Lock,
    timeout: float = 10.0,
) -> None:
    """Trickle candidates in both directions until both peers connect."""
    sender_cursor = 0
    receiver_cursor = 0
    no_candidate_deadline = time.monotonic() + 2.0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        local_candidates = sender.local_candidates(sender_cursor)
        for sender_candidate in local_candidates:
            receiver.emit(
                "add-ice-candidate",
                sender_candidate["sdpMLineIndex"],
                sender_candidate["candidate"],
            )
        sender_cursor += len(local_candidates)

        with receiver_candidates_lock:
            remote_candidates = tuple(receiver_candidates[receiver_cursor:])
        for mline_index, receiver_candidate in remote_candidates:
            sender.add_remote_candidate(mline_index, receiver_candidate)
        receiver_cursor += len(remote_candidates)

        sender_webrtc = sender._webrtc
        assert sender_webrtc is not None
        sender_state = sender_webrtc.get_property("connection-state")
        receiver_state = receiver.get_property("connection-state")
        if (
            sender_state.value_nick == "connected"
            and receiver_state.value_nick == "connected"
        ):
            return
        if (
            sender_cursor == 0
            and receiver_cursor == 0
            and time.monotonic() >= no_candidate_deadline
        ):
            pytest.skip("WebRTC runtime cannot discover network interfaces for ICE")
        time.sleep(0.02)

    pytest.fail(
        "WebRTC peers did not connect: "
        f"sender={sender_state.value_nick}, receiver={receiver_state.value_nick}"
    )


@pytest.mark.integration
def test_webrtc_peer_roundtrip_handles_delta_start_and_large_pts() -> None:
    """Decode real WebRTC video after a P-frame start and week-long source PTS."""
    runtime = _runtime()
    gst = runtime.Gst
    frames, description = _encoded_test_frames(runtime)
    keyframe_index = next(index for index, frame in enumerate(frames) if frame.keyframe)
    delta_frame = next(
        frame for frame in frames[keyframe_index + 1 :] if not frame.keyframe
    )
    keyframe_and_gop = frames[keyframe_index : keyframe_index + 10]
    assert keyframe_and_gop[0].keyframe

    hub = LatestFrameHub[EncodedVideoFrame]()
    hub.publish(delta_frame)
    subscription = hub.subscribe("webrtc-e2e")
    metrics = ClientMetricsRegistry(timeout_seconds=30.0, max_clients=1)
    metrics.connect("sender", ProductionPreviewConfig().transport)
    sender = WebRTCPeer(
        "sender",
        subscription,
        VideoEncoderConfig(),
        ProductionPreviewConfig(),
        metrics,
        stream_description=description,
    )
    receiver_pipeline = gst.parse_launch(
        "webrtcbin name=receiver bundle-policy=max-bundle latency=0 "
        "receiver. ! application/x-rtp,media=video,encoding-name=H264,"
        "clock-rate=90000,payload=96 ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=RGB ! "
        "appsink name=decoded_sink sync=false"
    )
    receiver = receiver_pipeline.get_by_name("receiver")
    decoded_sink = receiver_pipeline.get_by_name("decoded_sink")
    assert receiver is not None
    assert decoded_sink is not None
    receiver_candidates: list[tuple[int, str]] = []
    receiver_candidates_lock = threading.Lock()

    def on_receiver_candidate(
        _: object,
        mline_index: int,
        candidate: str,
    ) -> None:
        with receiver_candidates_lock:
            receiver_candidates.append((int(mline_index), str(candidate)))

    receiver.connect("on-ice-candidate", on_receiver_candidate)
    assert (
        receiver_pipeline.set_state(gst.State.PLAYING) != gst.StateChangeReturn.FAILURE
    )

    pushed_pts: list[int] = []
    publisher: threading.Thread | None = None
    try:
        sender.start()
        deadline = time.monotonic() + 2.0
        while sender._worker.processed_frames < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sender._worker.processed_frames >= 1
        assert metrics.snapshot()[0].frames_pushed == 0

        sender_appsrc = sender._appsrc
        assert sender_appsrc is not None
        appsrc_pad = sender_appsrc.get_static_pad("src")
        assert appsrc_pad is not None

        def record_pts(_: object, info: object) -> object:
            buffer = info.get_buffer()  # type: ignore[attr-defined]
            if buffer is not None:
                pushed_pts.append(int(buffer.pts))
            return gst.PadProbeReturn.OK

        appsrc_pad.add_probe(gst.PadProbeType.BUFFER, record_pts)
        offer_sdp = sender.create_offer()
        profile_level_id = description.profile_level_id
        assert profile_level_id is not None
        assert "profile-level-id=" + profile_level_id in offer_sdp
        answer_sdp = _set_remote_offer_and_create_answer(
            runtime,
            receiver,
            offer_sdp,
        )
        sender.set_answer(answer_sdp)
        _exchange_candidates(
            sender,
            receiver,
            receiver_candidates,
            receiver_candidates_lock,
        )

        def publish() -> None:
            for frame in keyframe_and_gop:
                hub.publish(frame)
                time.sleep(1 / 30)

        publisher = threading.Thread(target=publish, daemon=True)
        started_at = time.monotonic()
        publisher.start()
        decoded = 0
        for _ in keyframe_and_gop:
            sample = decoded_sink.emit("try-pull-sample", 2 * gst.SECOND)
            if sample is None:
                break
            buffer = sample.get_buffer()
            assert buffer is not None
            pixels = bytes(buffer.extract_dup(0, buffer.get_size()))
            assert any(pixels)
            decoded += 1
            if decoded >= 5:
                break

        assert decoded >= 5
        assert time.monotonic() - started_at < 5.0
        assert pushed_pts[0] == 0
        assert sender._timeline.base_pts_ns == keyframe_and_gop[0].pts_ns
        client = metrics.snapshot()[0]
        assert client.frames_pushed >= 5
        assert client.rtp_packets_sent > client.frames_pushed
        assert client.last_bus_error is None
    finally:
        if publisher is not None:
            publisher.join(timeout=2.0)
        sender.stop()
        receiver_pipeline.set_state(gst.State.NULL)
        hub.close()
