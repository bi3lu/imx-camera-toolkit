"""Tests for hardware video pipelines and production browser transports."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from imx_camera_toolkit import (
    CameraConfigurationError,
    EncodedStreamDescription,
    EncodedVideoFrame,
    GpuCamera,
    HardwareVideoConfig,
    VideoCodec,
    VideoEncoderBackend,
    VideoEncoderConfig,
    VideoEncoderPipeline,
    VideoEncodeStats,
    build_gpu_gstreamer_pipeline,
)
from imx_camera_toolkit._internal.api.security import (
    BrowserSessionOAuth2PasswordBearer,
    SecurityConfig,
    token_sha256,
)
from imx_camera_toolkit._internal.camera.backends.gpu_gstreamer import (
    GpuGStreamerCaptureBackend,
    _h264_parameter_sets,
)
from imx_camera_toolkit._internal.camera.models import MemoryType
from imx_camera_toolkit._internal.camera.publishing.video import EncodedVideoPublisher
from imx_camera_toolkit._internal.production_preview.api import _serialize_health
from imx_camera_toolkit._internal.production_preview.metrics import (
    ClientMetricsRegistry,
)
from imx_camera_toolkit._internal.production_preview.transport import (
    _push_encoded_frame,
    _PushTimeline,
)
from imx_camera_toolkit.consumers import LatestFrameHub
from imx_camera_toolkit.inference import InferenceResult
from imx_camera_toolkit.production_preview import (
    CudaOverlayRenderer,
    OverlayRectangle,
    PreviewTransport,
    ProductionPreviewConfig,
    ProductionPreviewConfigurationError,
    ProductionPreviewServer,
    build_hls_transport_pipeline,
    build_webrtc_peer_pipeline,
    create_production_preview_app,
)
from imx_camera_toolkit.testing import mock_gpu_frame


def _api_endpoint(application: Any, path: str) -> Callable[..., Any]:
    """Resolve a FastAPI endpoint for validation tests without a live portal."""
    for route in application.routes:
        if getattr(route, "path", None) == path:
            return cast(Callable[..., Any], route.endpoint)
    raise LookupError(path)


def test_hardware_video_pipeline_encodes_h264_directly_from_nvmm() -> None:
    """H.264 production encode must not introduce a host raw-video capsfilter."""
    config = HardwareVideoConfig(
        codec=VideoCodec.H264,
        bitrate_bps=4_000_000,
        keyframe_interval=30,
    )

    pipeline = build_gpu_gstreamer_pipeline(video_config=config)

    assert "queue name=video_queue" in pipeline
    assert "video/x-raw(memory:NVMM)" in pipeline
    assert "nvv4l2h264enc name=video_encoder" in pipeline
    assert "bitrate=4000000" in pipeline
    assert "iframeinterval=30" in pipeline
    assert "insert-sps-pps=1" in pipeline
    assert "h264parse config-interval=-1" in pipeline
    assert "appsink name=video_sink max-buffers=1 drop=true" in pipeline
    assert "videoconvert" not in pipeline


def test_x264_fallback_moves_only_encoder_branch_to_i420_cpu() -> None:
    """Orin Nano fallback must preserve NVMM capture/inference and overlay."""
    pipeline = build_gpu_gstreamer_pipeline(
        video_config=VideoEncoderConfig(backend=VideoEncoderBackend.X264),
        enable_video_overlay=True,
    )
    video_branch = pipeline.split("queue name=video_queue", maxsplit=1)[1]

    assert "identity name=video_overlay_hook" in video_branch
    assert "nvvidconv ! video/x-raw" in video_branch
    assert "format=(string)I420" in video_branch
    assert "x264enc name=video_encoder" in video_branch
    assert "key-int-max=30" in video_branch
    assert "nvv4l2h264enc" not in video_branch
    assert "appsink name=gpu_sink" in pipeline


def test_auto_backend_selects_x264_when_orin_nano_has_no_nvenc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry-based AUTO selection must reflect the installed elements."""
    monkeypatch.setattr(
        GpuGStreamerCaptureBackend,
        "element_available",
        classmethod(lambda cls, name: name == "x264enc"),
    )
    camera = GpuCamera(
        video_config=VideoEncoderConfig(backend=VideoEncoderBackend.AUTO),
    )

    assert camera._resolve_encoder_backend() is VideoEncoderBackend.X264


def test_runtime_rebuild_preserves_the_resolved_orin_nano_x264_backend() -> None:
    """Controls and preview changes must not revert AUTO to absent NVENC."""
    camera = GpuCamera(
        video_config=VideoEncoderConfig(backend=VideoEncoderBackend.AUTO),
    )
    camera._resolved_video_encoder_backend = VideoEncoderBackend.X264.value

    camera.apply_argus_properties(("wbmode=daylight",))
    camera.set_preview_enabled(True)

    assert "x264enc name=video_encoder" in camera.pipeline
    assert "nvv4l2h264enc" not in camera.pipeline
    assert "appsink name=preview_sink" in camera.pipeline


def test_custom_public_encoder_factory_survives_overlay_rebuild() -> None:
    """Custom encoders must not require patching the private pipeline string."""
    calls = 0

    def factory(
        config: VideoEncoderConfig,
        width: int,
        height: int,
        framerate: int,
    ) -> VideoEncoderPipeline:
        nonlocal calls
        calls += 1
        return VideoEncoderPipeline(
            pipeline=(
                "identity name=custom_encoder ! "
                "appsink name=video_sink max-buffers=1 drop=true"
            ),
            backend="custom",
            description=EncodedStreamDescription(
                codec=config.codec,
                width=width,
                height=height,
                fps=framerate,
            ),
            required_elements=("identity", "appsink"),
        )

    camera = GpuCamera(
        video_config=VideoEncoderConfig(),
        encoder_pipeline_factory=factory,
    )
    camera.set_video_overlay(_FakeNvmmOverlay())

    assert calls >= 2
    assert "identity name=custom_encoder" in camera.pipeline
    assert camera.video_encoder_backend == "custom"


def test_real_sps_drives_rfc6184_profile_level_id() -> None:
    """Constrained-baseline level 3.1 must be 42c01f, never guessed."""
    access_unit = (
        b"\x00\x00\x00\x01\x67\x42\xc0\x1f\xaa"
        b"\x00\x00\x01\x68\xbb"
        b"\x00\x00\x01\x65\xcc"
    )
    sps, pps = _h264_parameter_sets(access_unit)
    description = EncodedStreamDescription(
        codec=VideoCodec.H264,
        profile="constrained-baseline",
        level="3.1",
        sps=sps,
        pps=pps,
    )
    pipeline = build_webrtc_peer_pipeline(
        description,
        ProductionPreviewConfig(),
    )

    assert description.profile_level_id == "42c01f"
    assert "packetization-mode=(string)1" in pipeline
    assert "profile-level-id=(string)42c01f" in pipeline
    assert "42e01f" not in pipeline


def test_hardware_video_pipeline_supports_h265_and_isolated_gpu_overlay() -> None:
    """Overlay mutation must occur only after a device-side branch copy."""
    config = HardwareVideoConfig(codec=VideoCodec.H265)

    pipeline = build_gpu_gstreamer_pipeline(
        video_config=config,
        enable_video_overlay=True,
    )

    video_branch = pipeline.split("queue name=video_queue", maxsplit=1)[1]
    assert "nvvidconv" in video_branch
    assert "identity name=video_overlay_hook" in video_branch
    assert "nvv4l2h265enc name=video_encoder" in video_branch
    assert "h265parse config-interval=-1" in video_branch
    assert "video/x-h265" in video_branch
    assert "videoconvert" not in video_branch


def test_debug_mjpeg_can_coexist_with_optional_production_video() -> None:
    """Adding production transport must not replace the simple JPEG path."""
    pipeline = build_gpu_gstreamer_pipeline(
        enable_preview=True,
        video_config=HardwareVideoConfig(),
    )

    assert "nvjpegenc" in pipeline
    assert "appsink name=preview_sink" in pipeline
    assert "nvv4l2h264enc" in pipeline
    assert "appsink name=video_sink" in pipeline


class _FakeNvmmOverlay:
    """Structurally valid caller-owned GPU renderer."""

    memory_type = MemoryType.NVMM

    def render(self, frame: object) -> None:
        """Accept a borrowed frame without retaining it."""

    def close(self) -> None:
        """Release no resources in this test double."""


def test_gpu_camera_can_wire_overlay_after_inference_subscription() -> None:
    """An application may assemble inference before finalizing the pipeline."""
    camera = GpuCamera(
        video_config=HardwareVideoConfig(),
    )

    camera.set_video_overlay(_FakeNvmmOverlay())

    assert "identity name=video_overlay_hook" in camera.pipeline
    camera.set_video_overlay(None)
    assert "identity name=video_overlay_hook" not in camera.pipeline


def test_gpu_camera_rejects_overlay_without_hardware_video() -> None:
    """A renderer must not silently activate a host or unencoded branch."""
    camera = GpuCamera()

    with pytest.raises(CameraConfigurationError, match="hardware video"):
        camera.set_video_overlay(_FakeNvmmOverlay())


def test_encoded_video_metrics_report_recent_fps_and_actual_bitrate() -> None:
    """Encoder metrics must derive rates from emitted access-unit sizes."""
    publisher = EncodedVideoPublisher()
    for sequence in range(1, 31):
        publisher.publish(
            EncodedVideoFrame(
                sequence=sequence,
                timestamp_ns=(sequence - 1) * 1_000_000_000 // 30,
                codec=VideoCodec.H264,
                data=b"x" * 10_000,
                keyframe=sequence == 1,
            )
        )

    stats = publisher.stats(29 * 1_000_000_000 // 30)

    assert stats.encoded_frames == 30
    assert stats.encoded_bytes == 300_000
    assert stats.encode_fps == pytest.approx(30.0, rel=0.01)
    assert stats.bitrate_bps == pytest.approx(2_400_000, rel=0.01)


class _FakeBuffer:
    """Minimal writable Gst.Buffer used to inspect pushed timestamps."""

    def __init__(self, size: int) -> None:
        self.data = bytes(size)
        self.pts: int | None = None
        self.dts: int | None = None
        self.duration: int | None = None
        self.delta = False

    @classmethod
    def new_allocate(cls, _: object, size: int, __: object) -> _FakeBuffer:
        """Match Gst.Buffer.new_allocate."""
        return cls(size)

    def fill(self, _: int, data: bytes) -> None:
        """Retain encoded bytes."""
        self.data = data

    def set_flags(self, _: object) -> None:
        """Record a delta-unit flag."""
        self.delta = True


class _FakeFlowValue:
    """Comparable Gst.FlowReturn test value."""

    value_nick = "ok"


class _FakeGst:
    """Small GStreamer namespace used by the push helper."""

    Buffer = _FakeBuffer

    class BufferFlags:
        DELTA_UNIT = object()

    class FlowReturn:
        OK = _FakeFlowValue()


class _FakeAppSrc:
    """Collect buffers accepted by appsrc."""

    def __init__(self) -> None:
        self.buffers: list[_FakeBuffer] = []

    def emit(self, _: str, buffer: _FakeBuffer) -> _FakeFlowValue:
        """Accept and retain one buffer."""
        self.buffers.append(buffer)
        return _FakeGst.FlowReturn.OK


def test_peer_waits_for_idr_and_rebases_long_uptime_timestamps() -> None:
    """A new peer must begin on a keyframe at PTS/DTS zero."""
    runtime = cast(Any, type("Runtime", (), {"Gst": _FakeGst})())
    appsrc = _FakeAppSrc()
    timeline = _PushTimeline()
    delta = EncodedVideoFrame(
        sequence=1,
        timestamp_ns=1,
        codec=VideoCodec.H264,
        data=b"delta",
        keyframe=False,
        pts_ns=36_000_000_000_000,
    )
    keyframe = EncodedVideoFrame(
        sequence=2,
        timestamp_ns=2,
        codec=VideoCodec.H264,
        data=b"idr",
        keyframe=True,
        pts_ns=36_000_033_333_333,
        dts_ns=36_000_030_000_000,
        duration_ns=33_333_333,
    )
    following = EncodedVideoFrame(
        sequence=3,
        timestamp_ns=3,
        codec=VideoCodec.H264,
        data=b"p",
        keyframe=False,
        pts_ns=36_000_066_666_666,
        dts_ns=36_000_063_333_333,
    )

    assert _push_encoded_frame(runtime, appsrc, delta, timeline) is False
    assert _push_encoded_frame(runtime, appsrc, keyframe, timeline) is True
    assert _push_encoded_frame(runtime, appsrc, following, timeline) is True
    assert len(appsrc.buffers) == 2
    assert appsrc.buffers[0].pts == 0
    assert appsrc.buffers[0].dts == 0
    assert appsrc.buffers[0].duration == 33_333_333
    assert appsrc.buffers[1].pts == 33_333_333
    assert appsrc.buffers[1].dts == 33_333_333


def test_transport_builders_reuse_encoded_stream_without_software_encoder(
    tmp_path: Path,
) -> None:
    """WebRTC peers and HLS packaging must never create another encoder."""
    webrtc_config = ProductionPreviewConfig()
    hls_config = ProductionPreviewConfig(
        transport=PreviewTransport.HLS,
        hls_directory=tmp_path,
    )

    webrtc = build_webrtc_peer_pipeline(VideoCodec.H264, webrtc_config)
    hls = build_hls_transport_pipeline(VideoCodec.H265, hls_config)

    assert "rtph264pay" in webrtc
    assert "webrtcbin name=webrtc" in webrtc
    assert "h265parse" in hls
    assert "hlssink2 name=hls_sink" in hls
    for pipeline in (webrtc, hls):
        assert "nvv4l2" not in pipeline
        assert "x264enc" not in pipeline
        assert "x265enc" not in pipeline


def test_webrtc_rejects_h265_for_browser_compatibility() -> None:
    """H.265 remains available through HLS rather than unreliable WebRTC SDP."""
    with pytest.raises(
        ProductionPreviewConfigurationError,
        match="requires H.264",
    ):
        ProductionPreviewConfig().validate_codec(VideoCodec.H265)


def test_client_metrics_include_count_and_per_client_drop_rate() -> None:
    """Every browser must expose bounded sent/drop counters independently."""
    registry = ClientMetricsRegistry(timeout_seconds=30.0, max_clients=2)
    registry.connect("one", PreviewTransport.WEBRTC)
    registry.connect("two", PreviewTransport.WEBRTC)
    registry.record_sent("one", 1_000, count=8)
    registry.record_drop("one", 2)

    clients = registry.snapshot()

    assert len(clients) == 2
    assert clients[0].client_id == "one"
    assert clients[0].drop_rate == pytest.approx(0.2)
    assert clients[1].drop_rate == 0.0
    with pytest.raises(RuntimeError, match="limit"):
        registry.connect("three", PreviewTransport.WEBRTC)


def test_appsrc_acceptance_is_not_reported_as_rtp_delivery() -> None:
    """Pushed access units and packetized RTP must remain separate counters."""
    registry = ClientMetricsRegistry(timeout_seconds=30.0, max_clients=1)
    registry.connect("peer", PreviewTransport.WEBRTC)
    registry.record_pushed("peer", 4_000)
    registry.record_bus_message("peer", error="h264parse: not-negotiated")

    client = registry.snapshot()[0]

    assert client.frames_pushed == 1
    assert client.rtp_packets_sent == 0
    assert client.rtp_bytes_sent == 0
    with pytest.warns(DeprecationWarning, match="frames_sent is deprecated"):
        assert client.frames_sent == 1
    assert client.media_status == "failed"
    assert client.last_bus_error == "h264parse: not-negotiated"


def test_deprecated_frames_sent_keeps_frame_units() -> None:
    """Legacy frame counts must never expose packet fragmentation as frames."""
    registry = ClientMetricsRegistry(timeout_seconds=30.0, max_clients=1)
    registry.connect("peer", PreviewTransport.WEBRTC)
    registry.record_pushed("peer", 4_000)
    for _ in range(9):
        registry.record_rtp("peer", 1_200)

    client = registry.snapshot()[0]

    assert client.frames_pushed == 1
    assert client.rtp_packets_sent == 9
    with pytest.warns(DeprecationWarning, match="frames_sent is deprecated"):
        assert client.frames_sent == 1


def test_browser_view_handles_streamless_tracks_and_reconnects() -> None:
    """Bundled JS must tolerate missing msid and expired server sessions."""
    html = (Path(__file__).parents[2] / "view" / "production.html").read_text("utf-8")

    assert "event.streams[0] ?? new MediaStream([event.track])" in html
    assert "video.play().catch" in html
    assert "error.statusCode === 404 || error.statusCode === 410" in html
    assert "activePeer.getStats()" in html
    assert "framesDecoded" in html
    assert 'fetch("/auth/session"' in html
    assert 'type="password"' in html


class _FakeEncodedSource:
    """Hardware-video source with no physical camera or GStreamer runtime."""

    def __init__(self, codec: VideoCodec = VideoCodec.H264) -> None:
        self.running = True
        self.video_config = HardwareVideoConfig(codec=codec)
        self.video_encoder_backend = "x264"
        self.encoded_stream_description = EncodedStreamDescription(
            codec=codec,
            width=1280,
            height=720,
            fps=30,
            sps=(b"\x67\x42\xc0\x1f" if codec is VideoCodec.H264 else None),
            pps=(b"\x68\x00" if codec is VideoCodec.H264 else None),
        )
        self.video_stats = VideoEncodeStats(
            encoded_frames=30,
            encoded_bytes=300_000,
            encode_fps=30.0,
            bitrate_bps=2_400_000.0,
        )
        self.hub = LatestFrameHub[EncodedVideoFrame]()

    def subscribe_video(self, name: str) -> Any:
        """Return one test latest-frame slot."""
        return self.hub.subscribe(name)


def test_field_browser_exchanges_bearer_for_hls_capable_session_cookie() -> None:
    """Public shell login must authenticate headerless browser media requests."""
    security = SecurityConfig(
        field_mode=True,
        token_grants=(
            (token_sha256("stream-token"), frozenset({"stream:read", "admin"})),
        ),
        allowed_hosts=("camera.example",),
        require_https=True,
    )
    server = ProductionPreviewServer(_FakeEncodedSource())
    application = create_production_preview_app(
        server,
        manage_server=False,
        security_config=security,
    )

    root = _api_endpoint(application, "/")()
    login_request = Request(
        {
            "type": "http",
            "headers": [(b"authorization", b"Bearer stream-token")],
        }
    )
    login = _api_endpoint(application, "/auth/session")(login_request)
    cookie_request = Request(
        {
            "type": "http",
            "headers": [(b"cookie", b"imx_camera_session=stream-token")],
        }
    )
    extractor = BrowserSessionOAuth2PasswordBearer(
        tokenUrl="/auth/session",
        auto_error=False,
    )

    assert root.status_code == 200
    assert login.status_code == 204
    assert "HttpOnly" in login.headers["set-cookie"]
    assert "SameSite=strict" in login.headers["set-cookie"]
    assert "Secure" in login.headers["set-cookie"]
    assert asyncio.run(extractor(cookie_request)) == "stream-token"


def test_hls_api_serves_safe_assets_and_reports_client_segment_drops(
    tmp_path: Path,
) -> None:
    """HLS delivery must track segment gaps without accepting path traversal."""
    (tmp_path / "playlist.m3u8").write_text("#EXTM3U", "utf-8")
    (tmp_path / "segment00001.ts").write_bytes(b"one")
    (tmp_path / "segment00003.ts").write_bytes(b"three")
    server = ProductionPreviewServer(
        _FakeEncodedSource(),
        ProductionPreviewConfig(
            transport=PreviewTransport.HLS,
            hls_directory=tmp_path,
        ),
        health_providers={"inference": lambda: {"processed_frames": 12}},
    )
    server._running = True
    application = create_production_preview_app(server, manage_server=False)

    client_id = server.create_hls_session()
    playlist = server.hls_asset(client_id, "playlist.m3u8")
    first = server.hls_asset(client_id, "segment00001.ts")
    third = server.hls_asset(client_id, "segment00003.ts")
    with pytest.raises(FileNotFoundError):
        server.hls_asset(client_id, "../secret")
    health = _serialize_health(server)

    assert playlist.name == "playlist.m3u8"
    assert first.read_bytes() == b"one"
    assert third.read_bytes() == b"three"
    assert health["encode_fps"] == 30.0
    assert health["bitrate_bps"] == 2_400_000.0
    assert health["encoder_backend"] == "x264"
    assert cast(dict[str, Any], health["stream"])["profile_level_id"] == "42c01f"
    assert health["active_clients"] == 1
    assert health["components"] == {"inference": {"processed_frames": 12}}
    clients = cast(list[dict[str, Any]], health["clients"])
    assert clients[0]["dropped_frames"] == 1
    assert clients[0]["drop_rate"] == pytest.approx(1 / 3)
    paths = {getattr(route, "path", None) for route in application.routes}
    assert "/api/preview/session" in paths
    assert "/api/preview/hls/{client_id}/{asset}" in paths


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_sdp_bytes", 1024 * 1024 + 1),
        ("max_ice_candidate_bytes", 64 * 1024 + 1),
        ("max_ice_candidates_per_session", 4097),
        ("max_new_sessions_per_second", 1001),
        ("client_timeout_seconds", float("inf")),
    ],
)
def test_production_preview_rejects_unbounded_network_limits(
    field: str,
    value: int | float,
) -> None:
    """Every signaling limit must have a finite lower and upper bound."""
    with pytest.raises(ProductionPreviewConfigurationError):
        ProductionPreviewConfig(**{field: value})  # type: ignore[arg-type]


def test_production_preview_bounds_signaling_payloads_and_session_rate(
    tmp_path: Path,
) -> None:
    """Oversized signaling and bursts fail before allocating peer resources."""
    server = ProductionPreviewServer(
        _FakeEncodedSource(),
        ProductionPreviewConfig(
            transport=PreviewTransport.HLS,
            hls_directory=tmp_path,
            max_clients=10,
            max_sdp_bytes=8,
            max_ice_candidate_bytes=8,
            max_new_sessions_per_second=2,
        ),
    )
    server._running = True

    with pytest.raises(ValueError, match="max_sdp_bytes"):
        server.set_webrtc_answer("missing", "012345678")
    with pytest.raises(ValueError, match="max_ice_candidate_bytes"):
        server.add_webrtc_candidate("missing", 0, "012345678")

    server.create_hls_session()
    server.create_hls_session()
    with pytest.raises(RuntimeError, match="session rate limit"):
        server.create_hls_session()


def test_production_feedback_rejects_nan_and_absurd_counters() -> None:
    """Browser telemetry must not admit non-finite or unbounded numbers."""
    server = ProductionPreviewServer(_FakeEncodedSource())
    application = create_production_preview_app(server, manage_server=False)
    feedback = _api_endpoint(
        application,
        "/api/preview/webrtc/{client_id}/feedback",
    )
    base = {
        "packets_received": 1,
        "bytes_received": 2,
        "frames_received": 3,
        "frames_decoded": 3,
        "packets_lost": 0,
        "jitter_ms": 1.0,
        "rtt_ms": 2.0,
    }

    with pytest.raises(HTTPException) as nan_error:
        feedback("peer", {**base, "rtt_ms": float("nan")})
    assert nan_error.value.status_code == 422

    with pytest.raises(HTTPException) as counter_error:
        feedback("peer", {**base, "bytes_received": 1 << 64})
    assert counter_error.value.status_code == 422


class _ResultSource:
    """Mutable latest inference result used by the CUDA overlay test."""

    latest_result: InferenceResult | None = None


class _FakeStream:
    """CUDA stream test double."""

    handle = 1

    def __init__(self) -> None:
        self.synchronizations = 0

    def synchronize(self) -> None:
        """Record ordering before the video encoder consumes the surface."""
        self.synchronizations += 1


class _FakeOverlayInterop:
    """Record direct NVMM drawing calls without loading CUDA."""

    def __init__(self) -> None:
        self.stream = _FakeStream()
        self.draws: list[dict[str, object]] = []
        self.activations = 0

    def activate(self) -> Any:
        """Record each context scope entered by the renderer."""
        self.activations += 1
        return nullcontext()

    def create_stream(self) -> _FakeStream:
        """Return the renderer-owned test stream."""
        return self.stream

    def import_frame(self, frame: object) -> object:
        """Return an opaque imported surface."""
        return frame

    def draw_nv12_rectangle(self, surface: object, **values: object) -> None:
        """Record geometry and converted NV12 color."""
        self.draws.append({"surface": surface, **values})


def test_cuda_overlay_draws_latest_result_without_cpu_image_payload() -> None:
    """Production overlays must launch against NVMM and synchronize GPU work."""
    source = _ResultSource()
    source.latest_result = InferenceResult(
        frame_sequence=1,
        frame_timestamp_ns=time.monotonic_ns(),
        inference_time_ns=1,
        outputs=(),
        overlays=(OverlayRectangle(10, 20, 30, 40),),
    )
    interop = _FakeOverlayInterop()
    renderer = CudaOverlayRenderer(
        source,
        interop=cast(Any, interop),
    )
    frame = mock_gpu_frame(object(), width=1280, height=720)

    renderer.render(frame)

    assert renderer.memory_type is MemoryType.NVMM
    assert len(interop.draws) == 1
    assert interop.draws[0]["left"] == 10
    assert interop.draws[0]["yuv"] == (173, 42, 26)
    assert interop.stream.synchronizations == 1
    assert renderer.rendered_frames == 1
    assert renderer.empty_results == 0
    assert renderer.stale_results == 0
    assert renderer.failed_frames == 0
    assert renderer.last_error is None
    assert interop.activations == 2

    renderer.close()

    assert interop.stream.synchronizations == 2
    assert interop.activations == 3


def test_cuda_overlay_reports_empty_stale_and_failed_frames() -> None:
    """Overlay diagnostics must distinguish absence, age, and CUDA errors."""
    source = _ResultSource()
    interop = _FakeOverlayInterop()
    renderer = CudaOverlayRenderer(
        source,
        mapper=lambda result: cast(Any, result.overlays),
        interop=cast(Any, interop),
    )
    frame = mock_gpu_frame(object())

    renderer.render(frame)
    source.latest_result = InferenceResult(
        frame_sequence=1,
        frame_timestamp_ns=0,
        inference_time_ns=1,
        outputs=(),
        overlays=(OverlayRectangle(1, 1, 2, 2),),
    )
    renderer.render(frame)
    source.latest_result = InferenceResult(
        frame_sequence=2,
        frame_timestamp_ns=time.monotonic_ns(),
        inference_time_ns=1,
        outputs=(),
        overlays=(object(),),
    )
    with pytest.raises(TypeError, match="OverlayRectangle"):
        renderer.render(frame)

    assert renderer.empty_results == 1
    assert renderer.stale_results == 1
    assert renderer.failed_frames == 1
    assert isinstance(renderer.last_error, TypeError)
    assert renderer.health()["healthy"] is False
    renderer.close()


def test_overlay_health_provider_failure_does_not_break_debug_health() -> None:
    """An application diagnostics bug must be isolated in its component entry."""

    def fail() -> dict[str, object]:
        raise RuntimeError("health unavailable")

    server = ProductionPreviewServer(
        _FakeEncodedSource(),
        health_providers={"overlay": fail},
    )

    assert server.health_diagnostics() == {
        "overlay": {"healthy": False, "provider_error": "health unavailable"}
    }
