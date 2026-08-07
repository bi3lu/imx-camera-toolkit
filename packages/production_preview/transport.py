"""Shared-encoder HLS packaging and per-client WebRTC delivery."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from packages.camera.models import (
    EncodedVideoFrame,
    HardwareVideoConfig,
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


class EncodedVideoSource(Protocol):
    """Hardware-encoded source required by production transports."""

    @property
    def running(self) -> bool:
        """Whether the shared camera pipeline is active."""
        ...

    @property
    def video_config(self) -> HardwareVideoConfig | None:
        """Hardware encoder settings used by access units."""
        ...

    @property
    def video_stats(self) -> VideoEncodeStats:
        """Recent hardware encode throughput."""
        ...

    def subscribe_video(
        self,
        name: str,
    ) -> LatestFrameSubscription[EncodedVideoFrame]:
        """Create one latest encoded-frame slot for a transport worker."""
        ...


def _push_encoded_frame(
    runtime: GStreamerRuntime,
    appsrc: Any,
    frame: EncodedVideoFrame,
) -> None:
    """Push one encoded access unit without touching raw camera pixels."""
    gst = runtime.Gst
    buffer = gst.Buffer.new_allocate(None, len(frame.data), None)
    buffer.fill(0, frame.data)

    if frame.pts_ns is not None:
        buffer.pts = frame.pts_ns
        buffer.dts = frame.pts_ns

    if frame.duration_ns is not None:
        buffer.duration = frame.duration_ns

    if not frame.keyframe:
        buffer.set_flags(gst.BufferFlags.DELTA_UNIT)

    result = appsrc.emit("push-buffer", buffer)

    if result != gst.FlowReturn.OK:
        raise ProductionPreviewError(f"encoded appsrc returned {result.value_nick}")


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
        runtime = load_gstreamer_runtime()
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
        """Forward one newest hardware-encoded access unit to hlssink2."""
        if self._runtime is None or self._appsrc is None:
            raise ProductionPreviewError("HLS packager is not running")

        _push_encoded_frame(self._runtime, self._appsrc, frame)


class WebRTCPeer:
    """One browser peer fed from its own bounded shared-encoder slot."""

    def __init__(
        self,
        client_id: str,
        subscription: LatestFrameSubscription[EncodedVideoFrame],
        video_config: HardwareVideoConfig,
        config: ProductionPreviewConfig,
        metrics: ClientMetricsRegistry,
    ) -> None:
        """Configure a peer without starting transport or negotiation."""
        self.client_id = client_id
        self._subscription = subscription
        self._video_config = video_config
        self._config = config
        self._metrics = metrics
        self._runtime: GStreamerRuntime | None = None
        self._pipeline: Any | None = None
        self._appsrc: Any | None = None
        self._webrtc: Any | None = None
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
        return self._worker.last_error

    def start(self) -> None:
        """Create the RTP/webrtcbin pipeline and media worker."""
        runtime = load_gstreamer_runtime(webrtc=True)
        description = build_webrtc_peer_pipeline(
            self._video_config.codec,
            self._config,
        )
        pipeline = runtime.Gst.parse_launch(description)
        appsrc = pipeline.get_by_name("encoded_source")
        webrtc = pipeline.get_by_name("webrtc")
        peer_queue = pipeline.get_by_name("peer_queue")

        if appsrc is None or webrtc is None or peer_queue is None:
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
        self._worker.stop()

        if self._appsrc is not None:
            self._appsrc.emit("end-of-stream")

        if self._pipeline is not None and self._runtime is not None:
            self._pipeline.set_state(self._runtime.Gst.State.NULL)

        self._pipeline = None
        self._appsrc = None
        self._webrtc = None
        self._runtime = None

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

        _push_encoded_frame(self._runtime, self._appsrc, frame)
        self._metrics.record_sent(self.client_id, len(frame.data))

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
    ) -> None:
        """Validate an encoded source without starting camera or transport."""
        video_config = source.video_config
        if video_config is None:
            raise ValueError("source must have hardware video enabled")

        self._source = source
        self._video_config = video_config
        self._config = config or ProductionPreviewConfig()
        self._config.validate_codec(video_config.codec)
        self._metrics = ClientMetricsRegistry(
            self._config.client_timeout_seconds,
            self._config.max_clients,
        )
        self._lock = threading.RLock()
        self._running = False
        self._hls: HLSPackager | None = None
        self._peers: dict[str, WebRTCPeer] = {}
        self._hls_last_segment: dict[str, int] = {}

    @property
    def config(self) -> ProductionPreviewConfig:
        """Resolved production transport configuration."""
        return self._config

    @property
    def running(self) -> bool:
        """Whether this transport layer accepts browser clients."""
        with self._lock:
            return self._running

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
            )

            try:
                peer.start()
                offer = peer.create_offer()

            except Exception:
                peer.stop()
                self._metrics.disconnect(client_id)
                raise

            self._peers[client_id] = peer
            return client_id, offer

    def set_webrtc_answer(self, client_id: str, sdp: str) -> None:
        """Set an answer on an existing WebRTC peer."""
        self._peer(client_id).set_answer(sdp)

    def add_webrtc_candidate(
        self,
        client_id: str,
        mline_index: int,
        candidate: str,
    ) -> None:
        """Add a browser ICE candidate to an existing peer."""
        self._peer(client_id).add_remote_candidate(mline_index, candidate)

    def webrtc_candidates(
        self,
        client_id: str,
        after: int,
    ) -> tuple[dict[str, object], ...]:
        """Poll server candidates without retaining media frames."""
        return self._peer(client_id).local_candidates(after)

    def create_hls_session(self) -> str:
        """Register an HLS browser for client-count and segment-drop metrics."""
        self._expire_clients()
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
            self._metrics.disconnect(client_id)

        if peer is not None:
            peer.stop()

    def stats(self) -> ProductionPreviewStats:
        """Return encode FPS/bitrate, client count, and per-client drops."""
        self._expire_clients()
        clients = self._metrics.snapshot()
        return ProductionPreviewStats(
            transport=self._config.transport,
            codec=self._video_config.codec,
            encode=self._source.video_stats,
            active_clients=len(clients),
            clients=clients,
        )

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
    "EncodedVideoSource",
    "HLSPackager",
    "ProductionPreviewServer",
    "WebRTCPeer",
]
