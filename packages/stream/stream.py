"""Framework-neutral MJPEG streaming primitives for camera JPEG frames."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


DEFAULT_MJPEG_BOUNDARY = "frame"


class JPEGCamera(Protocol):
    """Interface required by :class:`MJPEGStream`.

    The camera package's ``Camera`` class implements this protocol.
    """

    @property
    def running(self) -> bool:
        """bool: Whether the camera capture loop is active."""
        ...

    def wait_for_jpeg(
        self, previous_frame_number: int, timeout: float = 2.0
    ) -> tuple[int, bytes | None]:
        """Wait for a newer JPEG frame.

        Args:
            previous_frame_number: Identifier of the previously consumed frame.
            timeout: Maximum wait time, in seconds.

        Returns:
            The latest frame identifier and its JPEG bytes, if available.
        """
        ...


def _encode_boundary(boundary: str) -> bytes:
    """Validate and encode an MJPEG multipart boundary.

    Args:
        boundary: ASCII boundary token without whitespace or line breaks.

    Returns:
        ASCII-encoded boundary token.

    Raises:
        ValueError: If the boundary is empty or cannot be used in an HTTP
            multipart header.
    """
    if not boundary or any(character.isspace() for character in boundary):
        raise ValueError("boundary must be a non-empty token without whitespace")

    try:
        return boundary.encode("ascii")

    except UnicodeEncodeError as error:
        raise ValueError("boundary must contain only ASCII characters") from error


def build_mjpeg_part(
    jpeg: bytes, boundary: str = DEFAULT_MJPEG_BOUNDARY
) -> bytes:
    """Format one JPEG image as an MJPEG multipart body part.

    Args:
        jpeg: Encoded JPEG image bytes.
        boundary: Multipart boundary without the leading ``--``.

    Returns:
        A complete multipart body part, including boundary and HTTP headers.

    Raises:
        TypeError: If ``jpeg`` is not bytes.
        ValueError: If ``jpeg`` is empty or ``boundary`` is invalid.
    """
    if not isinstance(jpeg, bytes):
        raise TypeError("jpeg must be bytes")

    if not jpeg:
        raise ValueError("jpeg must not be empty")

    encoded_boundary = _encode_boundary(boundary)
    headers = (
        b"--"
        + encoded_boundary
        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(jpeg)).encode("ascii")
        + b"\r\n\r\n"
    )
    return headers + jpeg + b"\r\n"


class MJPEGStream:
    """Expose the newest camera frames as an MJPEG byte iterator.

    The stream does not start or stop the camera. This permits multiple stream
    consumers to share the same camera instance and leaves camera ownership to
    the application.

    Args:
        camera: Running camera that provides JPEG frames.
        boundary: Multipart boundary without the leading ``--``.
        timeout: Maximum wait time for each camera frame, in seconds.

    Attributes:
        frames_sent: Number of multipart JPEG parts yielded by the stream.
        last_frame_number: Identifier of the most recently yielded frame.
    """

    def __init__(
        self,
        camera: JPEGCamera,
        *,
        boundary: str = DEFAULT_MJPEG_BOUNDARY,
        timeout: float = 2.0,
    ) -> None:
        """Initialize an MJPEG stream without taking ownership of the camera.

        Raises:
            ValueError: If ``boundary`` is invalid or ``timeout`` is not
                greater than zero.
        """
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        _encode_boundary(boundary)

        self._camera = camera
        self._boundary = boundary
        self._timeout = timeout
        self.frames_sent = 0
        self.last_frame_number: int | None = None

    @property
    def content_type(self) -> str:
        """str: Content type for an HTTP response carrying this stream."""
        return f"multipart/x-mixed-replace; boundary={self._boundary}"

    def __iter__(self) -> Iterator[bytes]:
        """Return an iterator yielding multipart JPEG body parts.

        Returns:
            Iterator of encoded MJPEG multipart body parts.
        """
        return self.iter_parts()

    def iter_parts(self) -> Iterator[bytes]:
        """Yield the latest JPEG frame until the camera capture loop stops.

        Frames skipped by the camera or by a slow consumer are intentionally
        not replayed. Each yielded item can be written directly to an HTTP
        response body.

        Yields:
            A complete multipart body part containing one JPEG frame.
        """
        previous_frame_number = -1

        while self._camera.running:
            frame_number, jpeg = self._camera.wait_for_jpeg(
                previous_frame_number, timeout=self._timeout
            )

            if jpeg is None or frame_number == previous_frame_number:
                continue

            previous_frame_number = frame_number
            self.frames_sent += 1
            self.last_frame_number = frame_number
            yield build_mjpeg_part(jpeg, self._boundary)


def stream_mjpeg(
    camera: JPEGCamera,
    *,
    boundary: str = DEFAULT_MJPEG_BOUNDARY,
    timeout: float = 2.0,
) -> Iterator[bytes]:
    """Create a one-shot MJPEG iterator for a camera.

    Args:
        camera: Running camera that provides JPEG frames.
        boundary: Multipart boundary without the leading ``--``.
        timeout: Maximum wait time for each camera frame, in seconds.

    Returns:
        Iterator of encoded MJPEG multipart body parts.
    """
    return iter(MJPEGStream(camera, boundary=boundary, timeout=timeout))
