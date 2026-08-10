"""Generic image-preview transport with no model-specific concepts."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from packages.camera.models import Frame
from packages.camera.publishing import JPEGPublisher
from packages.stream.stream import MJPEGStream

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RawSourceFrame:
    """Raw-source result that carries a latest-frame sequence number."""

    frame_number: int
    image: object


class PreviewSource(Protocol):
    """Source that can provide the latest image without an unbounded queue."""

    def read(self, timeout: float = 2.0, copy: bool = False) -> Frame | object | None:
        """Return the latest available image or frame within ``timeout``."""
        ...


class PreviewServer:
    """Publish arbitrary images to browser snapshot and MJPEG endpoints.

    ``PreviewServer`` is intentionally model-agnostic. It accepts an opaque
    image payload through :meth:`publish`, or reads one from an optional source
    exposing the small :class:`PreviewSource` protocol. It does not interpret
    detections, boxes, labels, masks, tracking IDs, or inference results.

    Args:
        source: Optional latest-frame source. The source lifecycle remains
            owned by the calling application.
        quality: JPEG quality from 0 to 100.
        max_fps: Maximum JPEG encoding rate in frames per second.
        read_timeout: Maximum wait for one source frame, in seconds.
    """

    def __init__(
        self,
        source: PreviewSource | None = None,
        *,
        quality: int = 65,
        max_fps: float = 30.0,
        read_timeout: float = 0.5,
    ) -> None:
        """Initialize a generic preview transport without starting workers."""
        if isinstance(quality, bool) or not isinstance(quality, int):
            raise ValueError("quality must be an integer")

        if not 0 <= quality <= 100:
            raise ValueError("quality must be between 0 and 100")

        if (
            isinstance(max_fps, bool)
            or not isinstance(max_fps, (int, float))
            or not math.isfinite(max_fps)
            or not 0 < max_fps <= 1_000
        ):
            raise ValueError("max_fps must be finite and between 0 and 1000")

        if (
            isinstance(read_timeout, bool)
            or not isinstance(read_timeout, (int, float))
            or not math.isfinite(read_timeout)
            or not 0 < read_timeout <= 3_600
        ):
            raise ValueError("read_timeout must be finite and between 0 and 3600")

        self._source = source
        self._read_timeout = float(read_timeout)
        self._publisher = JPEGPublisher(quality, float(max_fps))
        self._running = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self.last_error: Exception | None = None

    @property
    def source(self) -> PreviewSource | None:
        """PreviewSource | None: Source shared by this preview, when supplied."""
        return self._source

    @property
    def running(self) -> bool:
        """bool: Whether this preview transport is accepting or reading frames."""
        return self._running.is_set()

    @property
    def frame_number(self) -> int:
        """int: Identifier of the newest preview JPEG frame."""
        return self._publisher.frame_number

    @property
    def jpeg(self) -> bytes | None:
        """bytes | None: Latest preview JPEG, when an image was published."""
        return self._publisher.jpeg

    def publish(self, frame: Frame | object) -> bool:
        """Encode and publish an arbitrary image for browser consumers.

        Args:
            frame: Raw image payload or a toolkit :class:`Frame`. For a
                ``Frame``, only its opaque ``image`` payload is used.

        Returns:
            ``True`` when a JPEG frame was encoded. ``False`` means the
            configured JPEG rate limit skipped this publication.
        """
        image = frame.image if isinstance(frame, Frame) else frame
        return self._publisher.publish(image)

    def start(self) -> None:
        """Start optional source forwarding without starting or closing source."""
        with self._lifecycle_lock:
            if self.running:
                return

            self.last_error = None
            self._running.set()
            if self._source is None:
                return

            self._thread = threading.Thread(
                target=self._source_loop,
                name="imx-preview-source",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop source forwarding while preserving ownership of the source."""
        with self._lifecycle_lock:
            self._running.clear()
            self._publisher.notify_waiters()
            thread = self._thread

            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.0)

            self._thread = None

    def wait_for_jpeg(
        self,
        previous_frame_number: int,
        timeout: float = 2.0,
    ) -> tuple[int, bytes | None]:
        """Wait for a JPEG publication newer than ``previous_frame_number``.

        Args:
            previous_frame_number: Frame identifier already consumed.
            timeout: Maximum wait time in seconds.

        Returns:
            Latest JPEG frame identifier and bytes, when available.

        Raises:
            ValueError: If ``timeout`` is negative.
        """
        if timeout < 0:
            raise ValueError("timeout must be greater than or equal to zero")

        return self._publisher.wait_for_jpeg(
            previous_frame_number,
            timeout,
            lambda: self.running,
        )

    def create_app(
        self,
        *,
        view_path: str | Path | None = None,
        security_config: Any | None = None,
    ) -> Any:
        """Create a FastAPI application that transports this preview's images.

        The application serves only preview transport endpoints. It has no
        camera-control or inference-model endpoints.

        Args:
            view_path: Optional custom browser-view template path.
            security_config: Optional shared HTTP deployment policy.

        Returns:
            FastAPI application serving a browser view, snapshots, and MJPEG.

        Raises:
            RuntimeError: If the optional preview dependencies are unavailable.
            ValueError: If the browser view is invalid.
        """
        try:
            from fastapi import FastAPI, HTTPException, Query, Security
            from fastapi.responses import HTMLResponse, Response, StreamingResponse

        except ImportError as error:
            raise RuntimeError(
                "Preview HTTP support is optional. Install "
                '"imx-camera-toolkit[preview]".'
            ) from error

        from packages.api.api import NO_CACHE_HEADERS, load_camera_view
        from packages.api.security import (
            SecurityConfig,
            apply_security_middleware,
            build_authorizer,
        )

        resolved_security = security_config or SecurityConfig()

        if not isinstance(resolved_security, SecurityConfig):
            raise TypeError("security_config must be a SecurityConfig")

        authorize = build_authorizer(resolved_security)

        @asynccontextmanager
        async def lifespan(_: Any) -> AsyncIterator[None]:
            """Run preview forwarding for the HTTP application lifespan."""
            self.start()

            try:
                yield

            finally:
                self.stop()

        application = FastAPI(
            title="IMX Image Preview",
            description="Generic JPEG snapshot and MJPEG image transport.",
            lifespan=lifespan,
            docs_url="/docs" if resolved_security.docs_enabled else None,
            redoc_url="/redoc" if resolved_security.docs_enabled else None,
            openapi_url=("/openapi.json" if resolved_security.docs_enabled else None),
        )
        apply_security_middleware(application, resolved_security)
        application.state.preview_server = self

        @application.get(
            "/",
            response_class=HTMLResponse,
            include_in_schema=False,
            dependencies=[Security(authorize, scopes=["stream:read"])],
        )
        def index() -> Any:
            """Return the customizable generic browser-preview view."""
            try:
                return HTMLResponse(
                    content=load_camera_view(view_path=view_path),
                    headers=NO_CACHE_HEADERS,
                )

            except (RuntimeError, ValueError) as error:
                raise HTTPException(status_code=500, detail=str(error)) from error

        @application.get("/healthz", include_in_schema=False)
        def healthz() -> dict[str, str]:
            """Return a non-diagnostic process liveness response."""
            return {"status": "ok"}

        @application.get(
            "/api/camera/snapshot",
            dependencies=[Security(authorize, scopes=["stream:read"])],
        )
        def snapshot(after: int = Query(default=-1, ge=-1)) -> Any:
            """Return the newest generic preview JPEG."""
            frame_number, jpeg = self.wait_for_jpeg(after, timeout=2.0)
            if jpeg is None:
                raise HTTPException(status_code=503, detail="Preview frame unavailable")

            if after >= 0 and frame_number == after:
                return Response(status_code=204, headers=NO_CACHE_HEADERS)

            return Response(
                content=jpeg,
                media_type="image/jpeg",
                headers={**NO_CACHE_HEADERS, "X-Frame-Number": str(frame_number)},
            )

        @application.get(
            "/api/camera/mjpeg",
            dependencies=[Security(authorize, scopes=["stream:read"])],
        )
        def mjpeg() -> Any:
            """Return the newest generic preview images as an MJPEG response."""
            stream = MJPEGStream(self)
            return StreamingResponse(
                stream,
                media_type=stream.content_type,
                headers=NO_CACHE_HEADERS,
            )

        return application

    def _source_loop(self) -> None:
        """Forward latest source images while preserving source ownership."""
        previous_frame_number = -1
        while self.running:
            source = self._source
            if source is None:
                return

            if not self._source_running(source):
                time.sleep(0.05)
                continue

            try:
                frame = self._read_source_frame(source, previous_frame_number)
                if frame is not None:
                    if isinstance(frame, _RawSourceFrame):
                        previous_frame_number = frame.frame_number
                        published = self.publish(frame.image)
                    else:
                        published = self.publish(frame)

                    if not published:
                        time.sleep(min(self._read_timeout, 0.01))

                else:
                    time.sleep(min(self._read_timeout, 0.01))

            except Exception as error:
                self.last_error = error
                logger.exception("Preview source read or JPEG publication failed")
                time.sleep(0.1)

    def _read_source_frame(
        self,
        source: PreviewSource,
        previous_frame_number: int,
    ) -> Frame | _RawSourceFrame | object | None:
        """Read a source frame without imposing a camera-specific interface."""
        raw_waiter = getattr(source, "wait_for_raw_frame", None)
        if callable(raw_waiter):
            frame_number, frame = raw_waiter(
                previous_frame_number,
                timeout=self._read_timeout,
            )
            if frame is None or frame_number == previous_frame_number:
                return None

            return _RawSourceFrame(frame_number, frame)

        return source.read(timeout=self._read_timeout, copy=False)

    @staticmethod
    def _source_running(source: PreviewSource) -> bool:
        """Return a source running state when it exposes one, else ``True``."""
        running = getattr(source, "running", True)
        return running if isinstance(running, bool) else True
