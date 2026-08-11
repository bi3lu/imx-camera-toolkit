"""Shared-encoder HLS packaging and per-client WebRTC delivery."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from packages.camera.models import (
    EncodedStreamDescription,
    EncodedVideoFrame,
    HardwareVideoConfig,
    VideoCodec,
    VideoEncodeStats,
)
from packages.consumers import FrameConsumer, LatestFrameSubscription

from .config import PreviewTransport, ProductionPreviewConfig
from .errors import ProductionPreviewDependencyError, ProductionPreviewError
from .metrics import (
    ClientMetricsRegistry,
    ProductionPreviewStats,
)
from .pipeline import build_hls_transport_pipeline, build_webrtc_peer_pipeline
from .runtime import GStreamerRuntime, load_gstreamer_runtime

HealthProvider = Callable[[], Mapping[str, object]]


class EncodedVideoSource(Protocol):
    """Encoded source required by production transports."""

    @property
    def running(self) -> bool:
        """Whether the shared camera pipeline is active."""
        ...

    @property
    def video_config(self) -> HardwareVideoConfig | None:
        """Video encoder settings used by access units."""
        ...

    @property
    def video_stats(self) -> VideoEncodeStats:
        """Recent video encode throughput."""
        ...

    def subscribe_video(
        self,
        name: str,
    ) -> LatestFrameSubscription[EncodedVideoFrame]:
        """Create one latest encoded-frame slot for a transport worker."""
        ...


class DescribedEncodedVideoSource(EncodedVideoSource, Protocol):
    """Extended source contract used for safe cross-encoder negotiation."""

    @property
    def video_encoder_backend(self) -> str | None:
        """Resolved encoder backend used by the source."""
        ...

    @property
    def encoded_stream_description(self) -> EncodedStreamDescription | None:
        """Negotiated codec caps and parameter sets."""
        ...


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PushTimeline:
    """Per-subscriber GOP gate and zero-based timestamp state."""

    waiting_for_keyframe: bool = True
    base_pts_ns: int | None = None
    base_dts_ns: int | None = None
    last_pts_ns: int | None = None
    last_dts_ns: int | None = None


def _push_encoded_frame(
    runtime: GStreamerRuntime,
    appsrc: Any,
    frame: EncodedVideoFrame,
    timeline: _PushTimeline | None = None,
) -> bool:
    """Push one access unit with a peer-local timeline, beginning at an IDR."""
    if timeline is not None:
        if timeline.waiting_for_keyframe and not frame.keyframe:
            return False
        timeline.waiting_for_keyframe = False

    gst = runtime.Gst
    buffer = gst.Buffer.new_allocate(None, len(frame.data), None)
    buffer.fill(0, frame.data)

    if frame.pts_ns is not None:
        pts_ns = frame.pts_ns

        if timeline is not None:
            if timeline.base_pts_ns is None:
                timeline.base_pts_ns = pts_ns

            pts_ns = max(pts_ns - timeline.base_pts_ns, 0)

            if timeline.last_pts_ns is not None:
                pts_ns = max(pts_ns, timeline.last_pts_ns + 1)

            timeline.last_pts_ns = pts_ns

        buffer.pts = pts_ns

    dts_ns = frame.dts_ns if frame.dts_ns is not None else frame.pts_ns

    if dts_ns is not None:
        if timeline is not None:
            if timeline.base_dts_ns is None:
                timeline.base_dts_ns = dts_ns

            dts_ns = max(dts_ns - timeline.base_dts_ns, 0)

            if timeline.last_dts_ns is not None:
                dts_ns = max(dts_ns, timeline.last_dts_ns + 1)

            timeline.last_dts_ns = dts_ns

        buffer.dts = dts_ns

    if frame.duration_ns is not None:
        buffer.duration = frame.duration_ns

    if not frame.keyframe:
        buffer.set_flags(gst.BufferFlags.DELTA_UNIT)

    result = appsrc.emit("push-buffer", buffer)

    if result != gst.FlowReturn.OK:
        raise ProductionPreviewError(f"encoded appsrc returned {result.value_nick}")
    return True


class HLSPackager:
    """Package the shared encoded stream into a short rolling HLS window."""

    def __init__(
        self,
        subscription: LatestFrameSubscription[EncodedVideoFrame],
        video_config: HardwareVideoConfig,
        config: ProductionPreviewConfig,
    ) -> None:
        """Configure a packager without creating files or GStreamer objects."""
        self._subscription = subscription
        self._video_config = video_config
        self._config = config
        self._runtime: GStreamerRuntime | None = None
        self._pipeline: Any | None = None
        self._appsrc: Any | None = None
        self._timeline = _PushTimeline()
        self._worker = FrameConsumer(
            subscription,
            self._push,
            thread_name="imx-hls-packager",
        )

    @property
    def running(self) -> bool:
        """Whether the HLS ingestion worker is active."""
        return self._worker.running

    @property
    def dropped_frames(self) -> int:
        """Encoded access units skipped before the HLS appsrc."""
        return self._worker.dropped_frames

    @property
    def last_error(self) -> Exception | None:
        """Newest HLS ingestion error."""
        return self._worker.last_error

    def start(self) -> None:
        """Create the rolling output directory, pipeline, and ingestion worker."""
        directory = self._config.hls_directory

        if directory is None:
            raise ProductionPreviewError("HLS directory is not configured")

        directory.mkdir(parents=True, exist_ok=True)
        parser = (
            "h264parse" if self._video_config.codec is VideoCodec.H264 else "h265parse"
        )
        runtime = load_gstreamer_runtime(required_elements=(parser,))
        pipeline_description = build_hls_transport_pipeline(
            self._video_config.codec,
            self._config,
        )

        try:
            pipeline = runtime.Gst.parse_launch(pipeline_description)
            appsrc = pipeline.get_by_name("encoded_source")

            if appsrc is None:
                raise ProductionPreviewDependencyError(
                    "HLS pipeline has no encoded appsrc"
                )

            result = pipeline.set_state(runtime.Gst.State.PLAYING)

            if result == runtime.Gst.StateChangeReturn.FAILURE:
                raise ProductionPreviewDependencyError(
                    "HLS pipeline could not enter PLAYING"
                )

        except Exception:
            if "pipeline" in locals():
                pipeline.set_state(runtime.Gst.State.NULL)

            raise

        self._runtime = runtime
        self._pipeline = pipeline
        self._appsrc = appsrc
        self._worker.start()

    def stop(self) -> None:
        """Stop ingestion and finalize the current rolling playlist."""
        self._worker.stop()

        if self._appsrc is not None:
            self._appsrc.emit("end-of-stream")

        if self._pipeline is not None and self._runtime is not None:
            self._pipeline.set_state(self._runtime.Gst.State.NULL)

        self._pipeline = None
        self._appsrc = None
        self._runtime = None

    def _push(self, frame: EncodedVideoFrame) -> None:
        """Forward one newest encoded access unit to hlssink2."""
        if self._runtime is None or self._appsrc is None:
            raise ProductionPreviewError("HLS packager is not running")

        _push_encoded_frame(self._runtime, self._appsrc, frame, self._timeline)


class WebRTCPeer:
    """One browser peer fed from its own bounded shared-encoder slot."""

    def __init__(
        self,
        client_id: str,
        subscription: LatestFrameSubscription[EncodedVideoFrame],
        video_config: HardwareVideoConfig,
        config: ProductionPreviewConfig,
        metrics: ClientMetricsRegistry,
        stream_description: EncodedStreamDescription | None = None,
    ) -> None:
        """Configure a peer without starting transport or negotiation."""
        self.client_id = client_id
        self._subscription = subscription
        self._video_config = video_config
        self._stream_description = stream_description or EncodedStreamDescription(
            codec=video_config.codec
        )
        self._config = config
        self._metrics = metrics
        self._runtime: GStreamerRuntime | None = None
        self._pipeline: Any | None = None
        self._appsrc: Any | None = None
        self._webrtc: Any | None = None
        self._timeline = _PushTimeline()
        self._bus_running = threading.Event()
        self._bus_thread: threading.Thread | None = None
        self._bus_error: Exception | None = None
        self._candidate_lock = threading.Lock()
        self._local_candidates: list[dict[str, object]] = []
        self._last_reported_drops = 0
        self._worker = FrameConsumer(
            subscription,
            self._push,
            thread_name=f"imx-webrtc-{client_id}",
        )

    @property
    def last_error(self) -> Exception | None:
        """Newest peer media-delivery error."""
        return self._bus_error or self._worker.last_error

    def start(self) -> None:
        """Create the RTP/webrtcbin pipeline and media worker."""
        runtime = load_gstreamer_runtime(webrtc=True)
        description = build_webrtc_peer_pipeline(
            self._stream_description,
            self._config,
        )
        pipeline = runtime.Gst.parse_launch(description)
        appsrc = pipeline.get_by_name("encoded_source")
        webrtc = pipeline.get_by_name("webrtc")
        peer_queue = pipeline.get_by_name("peer_queue")
        parser = pipeline.get_by_name("peer_parser")
        payloader = pipeline.get_by_name("peer_payloader")

        if any(
            item is None for item in (appsrc, webrtc, peer_queue, parser, payloader)
        ):
            pipeline.set_state(runtime.Gst.State.NULL)
            raise ProductionPreviewDependencyError(
                "WebRTC pipeline is missing appsrc, queue, or webrtcbin"
            )

        if self._config.stun_server is not None:
            webrtc.set_property("stun-server", self._config.stun_server)

        if self._config.turn_server is not None:
            webrtc.set_property("turn-server", self._config.turn_server)

        webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        peer_queue.connect("overrun", self._on_queue_overrun)
        parser_src = parser.get_static_pad("src")
        payloader_src = payloader.get_static_pad("src")
        if parser_src is None or payloader_src is None:
            pipeline.set_state(runtime.Gst.State.NULL)
            raise ProductionPreviewDependencyError(
                "WebRTC parser or payloader has no source pad"
            )
        parser_src.add_probe(
            runtime.Gst.PadProbeType.BUFFER,
            self._on_parser_buffer,
        )
        payloader_src.add_probe(
            runtime.Gst.PadProbeType.BUFFER,
            self._on_rtp_packet,
        )
        result = pipeline.set_state(runtime.Gst.State.PLAYING)

        if result == runtime.Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(runtime.Gst.State.NULL)
            raise ProductionPreviewDependencyError(
                "WebRTC pipeline could not enter PLAYING"
            )

        self._runtime = runtime
        self._pipeline = pipeline
        self._appsrc = appsrc
        self._webrtc = webrtc
        profile_level_id = self._stream_description.profile_level_id
        fmtp = "packetization-mode=1"
        if profile_level_id is not None:
            fmtp += f";profile-level-id={profile_level_id}"
        self._metrics.record_negotiated_media(
            self.client_id,
            self._video_config.codec.value,
            fmtp,
        )
        self._bus_running.set()
        self._bus_thread = threading.Thread(
            target=self._monitor_bus,
            name=f"imx-webrtc-bus-{self.client_id}",
            daemon=True,
        )
        self._bus_thread.start()
        self._worker.start()

    def create_offer(self) -> str:
        """Create and install one server-side SDP offer."""
        runtime, webrtc = self._require_runtime()
        promise = runtime.Gst.Promise.new()
        webrtc.emit("create-offer", None, promise)

        if promise.wait() != runtime.Gst.PromiseResult.REPLIED:
            raise ProductionPreviewError("WebRTC offer promise was not replied")

        reply = promise.get_reply()
        offer = None if reply is None else reply.get_value("offer")

        if offer is None:
            raise ProductionPreviewError("webrtcbin returned no SDP offer")

        local_promise = runtime.Gst.Promise.new()
        webrtc.emit("set-local-description", offer, local_promise)
        local_promise.wait()
        self._metrics.touch(self.client_id)
        return str(offer.sdp.as_text())

    def set_answer(self, sdp: str) -> None:
        """Parse and install the browser's SDP answer."""
        if not isinstance(sdp, str) or not sdp.strip():
            raise ValueError("answer SDP must be a non-empty string")

        runtime, webrtc = self._require_runtime()

        if runtime.GstSdp is None or runtime.GstWebRTC is None:
            raise ProductionPreviewDependencyError("WebRTC namespaces unavailable")

        result, message = runtime.GstSdp.sdp_message_new_from_text(sdp)

        if result != runtime.GstSdp.SDPResult.OK or message is None:
            raise ValueError("invalid WebRTC answer SDP")

        answer = runtime.GstWebRTC.WebRTCSessionDescription.new(
            runtime.GstWebRTC.WebRTCSDPType.ANSWER,
            message,
        )
        promise = runtime.Gst.Promise.new()
        webrtc.emit("set-remote-description", answer, promise)
        promise.wait()
        self._metrics.touch(self.client_id)

    def add_remote_candidate(self, mline_index: int, candidate: str) -> None:
        """Forward one trickled browser ICE candidate to webrtcbin."""
        if mline_index < 0 or not isinstance(candidate, str) or not candidate:
            raise ValueError("invalid ICE candidate")

        _, webrtc = self._require_runtime()
        webrtc.emit("add-ice-candidate", mline_index, candidate)
        self._metrics.touch(self.client_id)

    def local_candidates(self, after: int = 0) -> tuple[dict[str, object], ...]:
        """Return server ICE candidates after a caller-owned cursor."""
        if after < 0:
            raise ValueError("candidate cursor must be non-negative")

        with self._candidate_lock:
            candidates = tuple(self._local_candidates[after:])

        self._metrics.touch(self.client_id)
        return candidates

    def stop(self) -> None:
        """Stop media delivery and close this peer pipeline."""
        self._bus_running.clear()
        self._worker.stop()

        bus_thread = self._bus_thread
        if bus_thread is not None and bus_thread is not threading.current_thread():
            bus_thread.join(timeout=1.0)

        if self._appsrc is not None:
            self._appsrc.emit("end-of-stream")

        if self._pipeline is not None and self._runtime is not None:
            self._pipeline.set_state(self._runtime.Gst.State.NULL)

        self._pipeline = None
        self._appsrc = None
        self._webrtc = None
        self._runtime = None
        self._bus_thread = None

    def _push(self, frame: EncodedVideoFrame) -> None:
        """Push newest access units and account for this peer's skipped slot."""
        if self._runtime is None or self._appsrc is None:
            raise ProductionPreviewError("WebRTC peer is not running")

        drops = self._subscription.dropped_frames

        if drops > self._last_reported_drops:
            self._metrics.record_drop(
                self.client_id,
                drops - self._last_reported_drops,
            )
            self._last_reported_drops = drops

        pushed = _push_encoded_frame(
            self._runtime,
            self._appsrc,
            frame,
            self._timeline,
        )

        if pushed:
            self._metrics.record_pushed(self.client_id, len(frame.data))

    def _on_parser_buffer(self, _: object, __: object) -> object:
        """Mark successful access-unit flow beyond h264parse."""
        if self._runtime is None:
            return 0

        self._metrics.record_flow(self.client_id, parser="ok")
        return self._runtime.Gst.PadProbeReturn.OK

    def _on_rtp_packet(self, _: object, info: object) -> object:
        """Count actual buffers emitted by rtph264pay."""
        if self._runtime is None:
            return 0
        buffer = info.get_buffer()  # type: ignore[attr-defined]

        if buffer is not None:
            self._metrics.record_rtp(self.client_id, int(buffer.get_size()))

        return self._runtime.Gst.PadProbeReturn.OK

    def _monitor_bus(self) -> None:
        """Continuously surface asynchronous parser/RTP/WebRTC failures."""
        runtime = self._runtime
        pipeline = self._pipeline

        if runtime is None or pipeline is None:
            return

        gst = runtime.Gst
        bus = pipeline.get_bus()
        mask = gst.MessageType.ERROR | gst.MessageType.WARNING | gst.MessageType.EOS

        while self._bus_running.is_set():
            message = bus.timed_pop_filtered(gst.SECOND // 10, mask)
            self._record_webrtc_states()

            if message is None:
                continue

            source = getattr(message, "src", None)
            source_name = (
                source.get_name()
                if source is not None and hasattr(source, "get_name")
                else "unknown"
            )

            if message.type == gst.MessageType.WARNING:
                warning, debug = message.parse_warning()
                detail = f"{source_name}: {warning}"

                if debug:
                    detail += f" ({debug})"

                logger.warning("WebRTC peer %s warning: %s", self.client_id, detail)
                self._metrics.record_bus_message(
                    self.client_id,
                    warning=detail,
                )
                continue

            if message.type == gst.MessageType.ERROR:
                error, debug = message.parse_error()
                detail = f"{source_name}: {error}"

                if debug:
                    detail += f" ({debug})"

            else:
                detail = f"{source_name}: end of stream"

            logger.error("WebRTC peer %s failed: %s", self.client_id, detail)
            self._bus_error = ProductionPreviewError(detail)
            self._metrics.record_bus_message(self.client_id, error=detail)

            if "parser" in source_name:
                self._metrics.record_flow(self.client_id, parser="failed")

            elif "payloader" in source_name:
                self._metrics.record_flow(self.client_id, payloader="failed")

            self._worker.stop(timeout=0)
            self._bus_running.clear()

    def _record_webrtc_states(self) -> None:
        """Sample stable state properties without depending on bus ordering."""
        webrtc = self._webrtc

        if webrtc is None:
            return

        def state(name: str) -> str | None:
            try:
                value = webrtc.get_property(name)

            except Exception:
                return None

            return str(getattr(value, "value_nick", value))

        self._metrics.record_states(
            self.client_id,
            signaling=state("signaling-state"),
            ice=state("ice-connection-state"),
            connection=state("connection-state"),
        )

    def _on_ice_candidate(
        self,
        _: object,
        mline_index: int,
        candidate: str,
    ) -> None:
        """Retain small signaling messages until the browser polls them."""
        with self._candidate_lock:
            self._local_candidates.append(
                {
                    "sdpMLineIndex": int(mline_index),
                    "candidate": str(candidate),
                }
            )

    def _on_queue_overrun(self, _: object) -> None:
        """Count downstream RTP/WebRTC queue drops for this client."""
        self._metrics.record_drop(self.client_id, 1)

    def _require_runtime(self) -> tuple[GStreamerRuntime, Any]:
        """Return live runtime and webrtcbin or fail clearly."""
        if self._runtime is None or self._webrtc is None:
            raise ProductionPreviewError("WebRTC peer is not running")

        return self._runtime, self._webrtc


class ProductionPreviewServer:
    """Coordinate one shared encoder with WebRTC peers or an HLS packager."""

    _HLS_SEGMENT = re.compile(r"segment(?P<number>[0-9]{5})[.]ts")

    def __init__(
        self,
        source: EncodedVideoSource,
        config: ProductionPreviewConfig | None = None,
        *,
        health_providers: Mapping[str, HealthProvider] | None = None,
    ) -> None:
        """Validate an encoded source without starting camera or transport."""
        video_config = source.video_config

        if video_config is None:
            raise ValueError("source must have hardware video enabled")

        self._source = source
        self._video_config = video_config
        self._stream_description = getattr(
            source,
            "encoded_stream_description",
            None,
        ) or EncodedStreamDescription(codec=video_config.codec)
        self._config = config or ProductionPreviewConfig()
        self._config.validate_codec(video_config.codec)
        providers = dict(health_providers or {})

        if any(not isinstance(name, str) or not name.strip() for name in providers):
            raise ValueError("health provider names must be non-empty strings")

        if any(not callable(provider) for provider in providers.values()):
            raise TypeError("health providers must be callable")

        self._health_providers = providers
        self._metrics = ClientMetricsRegistry(
            self._config.client_timeout_seconds,
            self._config.max_clients,
        )
        self._lock = threading.RLock()
        self._running = False
        self._hls: HLSPackager | None = None
        self._peers: dict[str, WebRTCPeer] = {}
        self._hls_last_segment: dict[str, int] = {}
        self._remote_candidate_counts: dict[str, int] = {}
        self._session_times: deque[float] = deque()

    @property
    def config(self) -> ProductionPreviewConfig:
        """Resolved production transport configuration."""
        return self._config

    @property
    def running(self) -> bool:
        """Whether this transport layer accepts browser clients."""
        with self._lock:
            return self._running

    def source_diagnostics(self) -> dict[str, object]:
        """Return generic capture diagnostics when the source exposes them."""
        last_error = getattr(self._source, "last_error", None)
        values: dict[str, object] = {
            "running": self._source.running,
            "backend": getattr(self._source, "active_backend", None),
            "last_error": None if last_error is None else str(last_error),
        }
        stats_method = getattr(self._source, "stats", None)

        if callable(stats_method):
            snapshot = stats_method()
            for output_name, attribute in (
                ("captured_frames", "captured_frames"),
                ("dropped_frames", "dropped_frames"),
                ("capture_fps", "capture_fps"),
                ("last_frame_timestamp_ns", "last_frame_timestamp_ns"),
            ):
                values[output_name] = getattr(snapshot, attribute, None)
        overlay_method = getattr(self._source, "overlay_diagnostics", None)

        if callable(overlay_method):
            values["overlay"] = overlay_method()

        return values

    def health_diagnostics(self) -> dict[str, dict[str, object]]:
        """Evaluate application-owned health providers without breaking health."""
        diagnostics: dict[str, dict[str, object]] = {}

        for name, provider in self._health_providers.items():
            try:
                values = provider()

                if not isinstance(values, Mapping):
                    raise TypeError("health provider must return a mapping")

                diagnostics[name] = dict(values)

            except Exception as error:
                diagnostics[name] = {
                    "healthy": False,
                    "provider_error": str(error),
                }

        return diagnostics

    def start(self) -> None:
        """Start HLS packaging or enable on-demand WebRTC peer creation."""
        with self._lock:
            if self._running:
                return

            if not self._source.running:
                raise RuntimeError("encoded camera source must be running")

            if self._config.transport is PreviewTransport.HLS:
                hls = HLSPackager(
                    self._source.subscribe_video("hls-packager"),
                    self._video_config,
                    self._config,
                )
                try:
                    hls.start()

                except Exception:
                    hls.stop()
                    raise

                self._hls = hls

            self._running = True

    def stop(self) -> None:
        """Close every client and transport without stopping the camera."""
        with self._lock:
            self._running = False
            peers = tuple(self._peers.values())
            self._peers.clear()
            self._remote_candidate_counts.clear()
            self._session_times.clear()
            hls = self._hls
            self._hls = None

        for peer in peers:
            peer.stop()
            self._metrics.disconnect(peer.client_id)

        if hls is not None:
            hls.stop()

    def create_webrtc_session(self) -> tuple[str, str]:
        """Create one peer and return its identifier and SDP offer."""
        self._expire_clients()
        self._consume_session_slot()
        stream_description = self._wait_for_stream_description()
        with self._lock:
            self._require_transport(PreviewTransport.WEBRTC)
            client_id = uuid4().hex
            self._metrics.connect(client_id, PreviewTransport.WEBRTC)
            peer = WebRTCPeer(
                client_id,
                self._source.subscribe_video(f"webrtc-{client_id}"),
                self._video_config,
                self._config,
                self._metrics,
                stream_description=stream_description,
            )

            try:
                peer.start()
                offer = peer.create_offer()

            except Exception:
                peer.stop()
                self._metrics.disconnect(client_id)
                raise

            self._peers[client_id] = peer
            self._remote_candidate_counts[client_id] = 0
            return client_id, offer

    def set_webrtc_answer(self, client_id: str, sdp: str) -> None:
        """Set an answer on an existing WebRTC peer."""
        if not isinstance(sdp, str) or not sdp.strip():
            raise ValueError("sdp must be a non-empty string")

        if len(sdp.encode("utf-8")) > self._config.max_sdp_bytes:
            raise ValueError("sdp exceeds max_sdp_bytes")

        self._peer(client_id).set_answer(sdp)

    def add_webrtc_candidate(
        self,
        client_id: str,
        mline_index: int,
        candidate: str,
    ) -> None:
        """Add a browser ICE candidate to an existing peer."""
        if (
            isinstance(mline_index, bool)
            or not isinstance(mline_index, int)
            or not 0 <= mline_index <= 65_535
        ):
            raise ValueError("mline_index must be between 0 and 65535")

        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("candidate must be a non-empty string")

        if len(candidate.encode("utf-8")) > self._config.max_ice_candidate_bytes:
            raise ValueError("candidate exceeds max_ice_candidate_bytes")

        with self._lock:
            peer = self._peer(client_id)
            count = self._remote_candidate_counts.get(client_id, 0)

            if count >= self._config.max_ice_candidates_per_session:
                raise ValueError("remote ICE candidate limit reached")

            peer.add_remote_candidate(mline_index, candidate)
            self._remote_candidate_counts[client_id] = count + 1

    def webrtc_candidates(
        self,
        client_id: str,
        after: int,
    ) -> tuple[dict[str, object], ...]:
        """Poll server candidates without retaining media frames."""
        return self._peer(client_id).local_candidates(after)

    def record_webrtc_feedback(
        self,
        client_id: str,
        *,
        packets_received: int,
        bytes_received: int,
        frames_received: int,
        frames_decoded: int,
        packets_lost: int | None = None,
        jitter_ms: float | None = None,
        rtt_ms: float | None = None,
    ) -> None:
        """Attach receiver-side browser stats to an existing peer."""
        self._peer(client_id)
        self._metrics.record_feedback(
            client_id,
            packets_received=packets_received,
            bytes_received=bytes_received,
            frames_received=frames_received,
            frames_decoded=frames_decoded,
            packets_lost=packets_lost,
            jitter_ms=jitter_ms,
            rtt_ms=rtt_ms,
        )

    def create_hls_session(self) -> str:
        """Register an HLS browser for client-count and segment-drop metrics."""
        self._expire_clients()
        self._consume_session_slot()

        with self._lock:
            self._require_transport(PreviewTransport.HLS)
            client_id = uuid4().hex
            self._metrics.connect(client_id, PreviewTransport.HLS)
            self._hls_last_segment[client_id] = -1
            return client_id

    def hls_asset(self, client_id: str, asset: str) -> Path:
        """Resolve a safe playlist/segment and record per-client segment gaps."""
        self._expire_clients()

        with self._lock:
            self._require_transport(PreviewTransport.HLS)
            directory = self._config.hls_directory

            if directory is None:
                raise ProductionPreviewError("HLS directory is unavailable")

            if client_id not in self._hls_last_segment:
                raise KeyError(client_id)

            if asset == "playlist.m3u8":
                self._metrics.touch(client_id)

            else:
                match = self._HLS_SEGMENT.fullmatch(asset)

                if match is None:
                    raise FileNotFoundError(asset)

                number = int(match.group("number"))
                previous = self._hls_last_segment.get(client_id)

                if previous is None:
                    raise KeyError(client_id)

                if previous >= 0 and number > previous + 1:
                    self._metrics.record_drop(client_id, number - previous - 1)

                self._hls_last_segment[client_id] = max(previous, number)

            path = directory / asset

            if not path.is_file():
                raise FileNotFoundError(asset)

            if asset != "playlist.m3u8":
                self._metrics.record_sent(client_id, path.stat().st_size)

            return path

    def disconnect(self, client_id: str) -> None:
        """Close one WebRTC peer or unregister one HLS browser."""
        with self._lock:
            peer = self._peers.pop(client_id, None)
            self._hls_last_segment.pop(client_id, None)
            self._remote_candidate_counts.pop(client_id, None)
            self._metrics.disconnect(client_id)

        if peer is not None:
            peer.stop()

    def _consume_session_slot(self) -> None:
        """Bound global session creation independently of active clients."""
        now = time.monotonic()

        with self._lock:
            while self._session_times and now - self._session_times[0] >= 1.0:
                self._session_times.popleft()

            if len(self._session_times) >= self._config.max_new_sessions_per_second:
                raise RuntimeError("production preview session rate limit reached")

            self._session_times.append(now)

    def stats(self) -> ProductionPreviewStats:
        """Return encode FPS/bitrate, client count, and per-client drops."""
        self._expire_clients()
        clients = self._metrics.snapshot()
        current_description = getattr(
            self._source,
            "encoded_stream_description",
            None,
        )
        if isinstance(current_description, EncodedStreamDescription):
            self._stream_description = current_description

        return ProductionPreviewStats(
            transport=self._config.transport,
            codec=self._video_config.codec,
            encoder_backend=getattr(
                self._source,
                "video_encoder_backend",
                None,
            ),
            stream=self._stream_description,
            encode=self._source.video_stats,
            active_clients=len(clients),
            clients=clients,
        )

    def _wait_for_stream_description(self) -> EncodedStreamDescription:
        """Wait briefly for real H.264 SPS instead of guessing SDP fmtp."""
        deadline = time.monotonic() + self._config.stream_description_timeout_seconds
        description = self._stream_description

        while True:
            latest = getattr(
                self._source,
                "encoded_stream_description",
                None,
            )

            if isinstance(latest, EncodedStreamDescription):
                description = latest
                self._stream_description = latest

            if (
                description.codec is not VideoCodec.H264
                or description.profile_level_id is not None
            ):
                return description

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "encoded H.264 stream description is not ready; no SPS "
                    "was received from the encoder"
                )
            time.sleep(0.01)

    def _expire_clients(self) -> None:
        """Close transport resources belonging to inactive browser sessions."""
        expired = self._metrics.expire()

        with self._lock:
            peers = tuple(
                peer
                for client_id in expired
                if (peer := self._peers.pop(client_id, None)) is not None
            )

            for client_id in expired:
                self._hls_last_segment.pop(client_id, None)
                self._remote_candidate_counts.pop(client_id, None)

        for peer in peers:
            peer.stop()

    def _peer(self, client_id: str) -> WebRTCPeer:
        """Resolve one peer or raise a stable lookup error."""
        with self._lock:
            self._require_transport(PreviewTransport.WEBRTC)

            try:
                return self._peers[client_id]

            except KeyError as error:
                raise KeyError(client_id) from error

    def _require_transport(self, transport: PreviewTransport) -> None:
        """Ensure the server is running in the requested transport mode."""
        if not self._running:
            raise RuntimeError("production preview server is not running")

        if self._config.transport is not transport:
            raise RuntimeError(f"{transport.value} transport is disabled")


__all__ = [
    "DescribedEncodedVideoSource",
    "EncodedVideoSource",
    "HealthProvider",
    "HLSPackager",
    "ProductionPreviewServer",
    "WebRTCPeer",
]
