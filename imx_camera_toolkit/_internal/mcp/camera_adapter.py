"""Read camera status and cached capabilities without an MCP SDK."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Protocol

from imx_camera_toolkit import __version__

from ..camera.camera import Camera
from ..camera.gpu_camera import GpuCamera
from ..camera.models import CameraStats
from .contracts import (
    MAX_SAFE_INTEGER,
    CameraBackend,
    CameraCapabilities,
    CameraState,
    CameraStatus,
)
from .errors import map_tool_errors
from .validation import EMPTY_ARGUMENTS, validate_camera_input

_PUBLIC_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class CameraStatsSource(Protocol):
    """Borrowed source that exposes capture diagnostics without acquiring frames."""

    def stats(self) -> CameraStats:
        """Return a point-in-time sample without waiting for a frame."""
        ...


class SubscriberStatsSource(Protocol):
    """Drop diagnostics for one application-owned subscription instance."""

    @property
    def dropped_frames(self) -> int:
        """Return the subscription's current independent drop count."""
        ...


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC wall time."""
    return datetime.now(timezone.utc)


def _camera_state(stats: CameraStats) -> CameraState:
    """Apply lifecycle precedence before testing publication readiness."""
    if stats.state == "failed":
        return "error"
    if stats.state == "recovering":
        return "recovering"
    if stats.state == "starting":
        return "starting"
    if stats.state == "stopped" or not stats.running:
        return "unavailable"
    return "ready" if stats.has_published_frame else "starting"


class CameraReadAdapter:
    """Borrow the camera and expose read-only data for the primary camera.

    The application owns the source and subscriptions. Reads use their
    diagnostics only and never wait for media or acquire a frame lease.
    """

    def __init__(
        self,
        camera: CameraStatsSource | None = None,
        *,
        backend: CameraBackend | None = None,
        capability_source: Callable[[], CameraCapabilities | None] | None = None,
        subscribers: Mapping[str, SubscriberStatsSource] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Bind the application's existing read-only sources.

        Args:
            camera: Application-owned camera, or ``None`` if unregistered.
            backend: Explicit capture path for a custom source. Built-in
                cameras identify their path by type when this is omitted.
            capability_source: Read-only provider of approved, cached
                capabilities. Return ``None`` when discovery is unavailable.
            subscribers: Approved public aliases mapped to individual existing
                subscriptions. The mapping is copied at construction.
            clock: Wall clock returning an aware datetime for the observation.

        Raises:
            ValueError: If a backend or public alias is invalid.
        """
        if backend not in (None, "cpu", "gpu"):
            raise ValueError("unsupported camera backend")
        if backend is None:
            if isinstance(camera, Camera):
                backend = "cpu"
            elif isinstance(camera, GpuCamera):
                backend = "gpu"
        self._camera = camera
        self._backend = backend
        self._capability_source = capability_source
        self._subscribers = dict(subscribers) if subscribers is not None else {}
        if any(
            not isinstance(alias, str) or _PUBLIC_ALIAS.fullmatch(alias) is None
            for alias in self._subscribers
        ):
            raise ValueError("invalid public subscriber alias")
        self._clock = clock

    def status(self, arguments: object = EMPTY_ARGUMENTS) -> CameraStatus:
        """Read one camera statistics sample and separate subscription counts.

        Args:
            arguments: Decoded tool arguments from an authorized caller.

        Raises:
            MCPError: If validation, a diagnostics source or the clock fails.
        """
        with map_tool_errors():
            validate_camera_input(arguments)
            stats = self._camera.stats() if self._camera is not None else None
            observed_at = self._clock()
            if observed_at.utcoffset() is None:
                raise ValueError("observation clock must return an aware datetime")
            observed_at_utc = observed_at.astimezone(timezone.utc).isoformat()
            observed_at_utc = observed_at_utc.removesuffix("+00:00") + "Z"
            if stats is None:
                return CameraStatus("unavailable", None, observed_at_utc, __version__)

            subscriber_drops = {}
            for alias, subscription in self._subscribers.items():
                count = subscription.dropped_frames
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError("invalid subscriber drop count")
                if count <= MAX_SAFE_INTEGER:
                    subscriber_drops[alias] = count

            return CameraStatus(
                state=_camera_state(stats),
                backend=self._backend,
                observed_at_utc=observed_at_utc,
                toolkit_version=__version__,
                capture_fps=stats.capture_fps,
                capture_dropped_frames=(
                    stats.dropped_frames
                    if stats.dropped_frames <= MAX_SAFE_INTEGER
                    else None
                ),
                subscriber_dropped_frames=subscriber_drops,
            )

    def capabilities(self, arguments: object = EMPTY_ARGUMENTS) -> CameraCapabilities:
        """Read approved capability data without probing capture or controls.

        Args:
            arguments: Decoded tool arguments from an authorized caller.

        Raises:
            MCPError: If validation or the registered capability provider fails.
        """
        with map_tool_errors():
            validate_camera_input(arguments)

            if self._camera is None or self._capability_source is None:
                return CameraCapabilities()

            capabilities = self._capability_source()
            if capabilities is None:
                return CameraCapabilities()
            if not isinstance(capabilities, CameraCapabilities):
                raise TypeError("capability provider must return CameraCapabilities")
            return capabilities
