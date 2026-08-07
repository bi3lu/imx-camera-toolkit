"""Thread-safe encode and per-client production preview diagnostics."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from packages.camera.models import (
    EncodedStreamDescription,
    VideoCodec,
    VideoEncodeStats,
)

from .config import PreviewTransport


@dataclass(frozen=True, slots=True)
class PreviewClientStats:
    """Immutable delivery and media-pipeline metrics for one client."""

    client_id: str
    transport: PreviewTransport
    connected_at_ns: int
    last_seen_ns: int
    frames_pushed: int
    pushed_bytes: int
    rtp_packets_sent: int
    rtp_bytes_sent: int
    dropped_frames: int
    last_rtp_packet_ns: int | None
    media_status: str
    signaling_state: str | None
    ice_connection_state: str | None
    connection_state: str | None
    parser_flow: str | None
    payloader_flow: str | None
    negotiated_codec: str | None
    negotiated_fmtp: str | None
    last_bus_error: str | None
    last_bus_warning: str | None
    rtt_ms: float | None
    jitter_ms: float | None
    packets_lost: int | None
    packets_received: int | None
    bytes_received: int | None
    frames_received: int | None
    frames_decoded: int | None

    @property
    def frames_sent(self) -> int:
        """Compatibility alias using real RTP for WebRTC delivery."""
        if self.transport is PreviewTransport.WEBRTC:
            return self.rtp_packets_sent

        return self.frames_pushed

    @property
    def bytes_sent(self) -> int:
        """Compatibility alias using real RTP bytes for WebRTC delivery."""
        if self.transport is PreviewTransport.WEBRTC:
            return self.rtp_bytes_sent

        return self.pushed_bytes

    @property
    def drop_rate(self) -> float:
        """Fraction of source access units or segments skipped."""
        total = self.frames_pushed + self.dropped_frames
        return 0.0 if total == 0 else self.dropped_frames / total


@dataclass(frozen=True, slots=True)
class ProductionPreviewStats:
    """Encoder identity and browser delivery metrics in one snapshot."""

    transport: PreviewTransport
    codec: VideoCodec
    encoder_backend: str | None
    stream: EncodedStreamDescription
    encode: VideoEncodeStats
    active_clients: int
    clients: tuple[PreviewClientStats, ...]


@dataclass(slots=True)
class _MutableClientStats:
    """Internal counters guarded by the registry lock."""

    client_id: str
    transport: PreviewTransport
    connected_at_ns: int
    last_seen_ns: int
    frames_pushed: int = 0
    pushed_bytes: int = 0
    rtp_packets_sent: int = 0
    rtp_bytes_sent: int = 0
    dropped_frames: int = 0
    last_rtp_packet_ns: int | None = None
    signaling_state: str | None = None
    ice_connection_state: str | None = None
    connection_state: str | None = None
    parser_flow: str | None = None
    payloader_flow: str | None = None
    negotiated_codec: str | None = None
    negotiated_fmtp: str | None = None
    last_bus_error: str | None = None
    last_bus_warning: str | None = None
    rtt_ms: float | None = None
    jitter_ms: float | None = None
    packets_lost: int | None = None
    packets_received: int | None = None
    bytes_received: int | None = None
    frames_received: int | None = None
    frames_decoded: int | None = None


class ClientMetricsRegistry:
    """Track bounded per-client delivery counters and inactive sessions."""

    MEDIA_STALL_NS = 3_000_000_000

    def __init__(self, timeout_seconds: float, max_clients: int) -> None:
        """Initialize an empty registry with validated config-owned limits."""
        self._timeout_ns = int(timeout_seconds * 1_000_000_000)
        self._max_clients = max_clients
        self._lock = threading.Lock()
        self._clients: dict[str, _MutableClientStats] = {}

    def connect(self, client_id: str, transport: PreviewTransport) -> None:
        """Register a new unique client or reject the configured limit."""
        now_ns = time.monotonic_ns()
        with self._lock:
            if client_id in self._clients:
                raise ValueError(f"client {client_id!r} already exists")
            if len(self._clients) >= self._max_clients:
                raise RuntimeError("production preview client limit reached")
            self._clients[client_id] = _MutableClientStats(
                client_id=client_id,
                transport=transport,
                connected_at_ns=now_ns,
                last_seen_ns=now_ns,
            )

    def disconnect(self, client_id: str) -> None:
        """Forget one client without changing cumulative encoder metrics."""
        with self._lock:
            self._clients.pop(client_id, None)

    def touch(self, client_id: str) -> None:
        """Mark signaling or HLS activity for timeout accounting."""
        with self._lock:
            client = self._clients.get(client_id)
            if client is not None:
                client.last_seen_ns = time.monotonic_ns()

    def record_pushed(self, client_id: str, size: int, count: int = 1) -> None:
        """Record access units accepted by appsrc, not downstream delivery."""
        with self._lock:
            client = self._clients.get(client_id)
            if client is not None:
                client.frames_pushed += count
                client.pushed_bytes += size
                client.last_seen_ns = time.monotonic_ns()

    def record_sent(self, client_id: str, size: int, count: int = 1) -> None:
        """Compatibility helper for actually served HLS segments."""
        self.record_pushed(client_id, size, count)

    def record_rtp(self, client_id: str, size: int) -> None:
        """Record one packet observed after the RTP payloader."""
        now_ns = time.monotonic_ns()
        with self._lock:
            client = self._clients.get(client_id)
            if client is not None:
                client.rtp_packets_sent += 1
                client.rtp_bytes_sent += size
                client.last_rtp_packet_ns = now_ns
                client.last_seen_ns = now_ns
                client.payloader_flow = "ok"

    def record_drop(self, client_id: str, count: int) -> None:
        """Record newest-frame replacements or skipped HLS segments."""
        if count <= 0:
            return
        with self._lock:
            client = self._clients.get(client_id)
            if client is not None:
                client.dropped_frames += count
                client.last_seen_ns = time.monotonic_ns()

    def record_bus_message(
        self,
        client_id: str,
        *,
        error: str | None = None,
        warning: str | None = None,
    ) -> None:
        """Retain the newest asynchronous downstream diagnostic."""
        with self._lock:
            client = self._clients.get(client_id)

            if client is not None:
                if error is not None:
                    client.last_bus_error = error

                if warning is not None:
                    client.last_bus_warning = warning

    def record_states(
        self,
        client_id: str,
        *,
        signaling: str | None = None,
        ice: str | None = None,
        connection: str | None = None,
    ) -> None:
        """Update WebRTC signaling and connectivity states."""
        with self._lock:
            client = self._clients.get(client_id)
            if client is None:
                return

            if signaling is not None:
                client.signaling_state = signaling

            if ice is not None:
                client.ice_connection_state = ice

            if connection is not None:
                client.connection_state = connection

    def record_flow(
        self,
        client_id: str,
        *,
        parser: str | None = None,
        payloader: str | None = None,
    ) -> None:
        """Update the latest observed parser/payloader flow state."""
        with self._lock:
            client = self._clients.get(client_id)
            if client is not None:
                if parser is not None:
                    client.parser_flow = parser
                if payloader is not None:
                    client.payloader_flow = payloader

    def record_negotiated_media(
        self,
        client_id: str,
        codec: str,
        fmtp: str | None,
    ) -> None:
        """Store the codec/fmtp used to construct this peer."""
        with self._lock:
            client = self._clients.get(client_id)
            if client is not None:
                client.negotiated_codec = codec
                client.negotiated_fmtp = fmtp

    def record_feedback(
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
        """Record browser receiver statistics supplied by the bundled view."""
        with self._lock:
            client = self._clients.get(client_id)
            if client is None:
                return
            client.packets_received = packets_received
            client.bytes_received = bytes_received
            client.frames_received = frames_received
            client.frames_decoded = frames_decoded
            client.packets_lost = packets_lost
            client.jitter_ms = jitter_ms
            client.rtt_ms = rtt_ms
            client.last_seen_ns = time.monotonic_ns()

    def snapshot(self) -> tuple[PreviewClientStats, ...]:
        """Return active clients ordered by identifier."""
        now_ns = time.monotonic_ns()
        with self._lock:
            return tuple(
                self._freeze(client, now_ns)
                for client in sorted(
                    self._clients.values(), key=lambda item: item.client_id
                )
            )

    def _freeze(
        self,
        client: _MutableClientStats,
        now_ns: int,
    ) -> PreviewClientStats:
        """Derive a media state from data flow rather than ICE alone."""
        if client.last_bus_error is not None:
            media_status = "failed"

        elif client.transport is PreviewTransport.HLS:
            media_status = "active" if client.frames_pushed else "starting"

        elif client.last_rtp_packet_ns is not None:
            media_status = (
                "stalled"
                if now_ns - client.last_rtp_packet_ns > self.MEDIA_STALL_NS
                else "active"
            )

        elif (
            client.frames_pushed
            and now_ns - client.connected_at_ns > self.MEDIA_STALL_NS
        ):
            media_status = "failed"

        else:
            media_status = "starting"

        return PreviewClientStats(
            client_id=client.client_id,
            transport=client.transport,
            connected_at_ns=client.connected_at_ns,
            last_seen_ns=client.last_seen_ns,
            frames_pushed=client.frames_pushed,
            pushed_bytes=client.pushed_bytes,
            rtp_packets_sent=client.rtp_packets_sent,
            rtp_bytes_sent=client.rtp_bytes_sent,
            dropped_frames=client.dropped_frames,
            last_rtp_packet_ns=client.last_rtp_packet_ns,
            media_status=media_status,
            signaling_state=client.signaling_state,
            ice_connection_state=client.ice_connection_state,
            connection_state=client.connection_state,
            parser_flow=client.parser_flow,
            payloader_flow=client.payloader_flow,
            negotiated_codec=client.negotiated_codec,
            negotiated_fmtp=client.negotiated_fmtp,
            last_bus_error=client.last_bus_error,
            last_bus_warning=client.last_bus_warning,
            rtt_ms=client.rtt_ms,
            jitter_ms=client.jitter_ms,
            packets_lost=client.packets_lost,
            packets_received=client.packets_received,
            bytes_received=client.bytes_received,
            frames_received=client.frames_received,
            frames_decoded=client.frames_decoded,
        )

    def expire(self) -> tuple[str, ...]:
        """Remove inactive clients and return IDs for transport cleanup."""
        now_ns = time.monotonic_ns()

        with self._lock:
            expired = tuple(
                client_id
                for client_id, client in self._clients.items()
                if now_ns - client.last_seen_ns > self._timeout_ns
            )

            for client_id in expired:
                self._clients.pop(client_id, None)

            return expired


__all__ = [
    "ClientMetricsRegistry",
    "PreviewClientStats",
    "ProductionPreviewStats",
]
