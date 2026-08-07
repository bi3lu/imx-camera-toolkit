"""Thread-safe encode and per-client production preview metrics."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from packages.camera.models import VideoCodec, VideoEncodeStats

from .config import PreviewTransport


@dataclass(frozen=True, slots=True)
class PreviewClientStats:
    """Immutable delivery metrics for one active browser client."""

    client_id: str
    transport: PreviewTransport
    connected_at_ns: int
    last_seen_ns: int
    frames_sent: int
    bytes_sent: int
    dropped_frames: int

    @property
    def drop_rate(self) -> float:
        """Fraction of observed frames that could not be delivered."""
        total = self.frames_sent + self.dropped_frames
        return 0.0 if total == 0 else self.dropped_frames / total


@dataclass(frozen=True, slots=True)
class ProductionPreviewStats:
    """Hardware encode and browser delivery metrics in one snapshot."""

    transport: PreviewTransport
    codec: VideoCodec
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
    frames_sent: int = 0
    bytes_sent: int = 0
    dropped_frames: int = 0


class ClientMetricsRegistry:
    """Track bounded per-client delivery counters and inactive sessions."""

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

    def record_sent(self, client_id: str, size: int, count: int = 1) -> None:
        """Record frames or segments delivered to one client."""
        with self._lock:
            client = self._clients.get(client_id)

            if client is None:
                return

            client.frames_sent += count
            client.bytes_sent += size
            client.last_seen_ns = time.monotonic_ns()

    def record_drop(self, client_id: str, count: int) -> None:
        """Record newest-frame replacements or skipped HLS segments."""
        if count <= 0:
            return

        with self._lock:
            client = self._clients.get(client_id)

            if client is None:
                return

            client.dropped_frames += count
            client.last_seen_ns = time.monotonic_ns()

    def snapshot(self) -> tuple[PreviewClientStats, ...]:
        """Return active clients ordered by identifier."""
        with self._lock:
            return tuple(
                PreviewClientStats(
                    client_id=client.client_id,
                    transport=client.transport,
                    connected_at_ns=client.connected_at_ns,
                    last_seen_ns=client.last_seen_ns,
                    frames_sent=client.frames_sent,
                    bytes_sent=client.bytes_sent,
                    dropped_frames=client.dropped_frames,
                )

                for client in sorted(
                    self._clients.values(),
                    key=lambda item: item.client_id,
                )
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
