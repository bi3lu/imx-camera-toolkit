"""Framework-neutral MJPEG streaming primitives for camera JPEG frames."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MJPEG_BOUNDARY = "frame"
DEFAULT_STREAM_TIMEOUT = 2.0
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yml")


@dataclass(frozen=True)
class StreamConfig:
    """Validated settings used to create an MJPEG stream.

    Attributes:
        boundary: MIME multipart boundary without the leading ``--``.
        timeout: Maximum wait time for a newer camera frame, in seconds.
    """

    boundary: str = DEFAULT_MJPEG_BOUNDARY
    timeout: float = DEFAULT_STREAM_TIMEOUT


DEFAULT_STREAM_CONFIG = StreamConfig()

try:
    import yaml

except ImportError:
    yaml: Any | None = None


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


def _validate_stream_config(config: StreamConfig) -> None:
    """Validate values used by an MJPEG stream.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If the multipart boundary or frame timeout is invalid.
    """
    if not isinstance(config.boundary, str):
        raise ValueError("boundary must be a string")
    if isinstance(config.timeout, bool) or not isinstance(config.timeout, (int, float)):
        raise ValueError("timeout must be a number")
    if config.timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    _encode_boundary(config.boundary)


def _read_config_values(config_data: dict[str, Any]) -> StreamConfig:
    """Convert a parsed YAML mapping into a validated stream configuration.

    Args:
        config_data: Mapping stored under ``stream_config`` in the YAML file.

    Returns:
        Validated stream configuration.

    Raises:
        ValueError: If keys are unknown or values have invalid types or ranges.
    """
    valid_keys = set(DEFAULT_STREAM_CONFIG.__dataclass_fields__)
    unknown_keys = set(config_data) - valid_keys
    if unknown_keys:
        formatted_keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"unknown stream configuration key(s): {formatted_keys}")

    config = StreamConfig(
        boundary=config_data.get("boundary", DEFAULT_STREAM_CONFIG.boundary),
        timeout=config_data.get("timeout", DEFAULT_STREAM_CONFIG.timeout),
    )
    _validate_stream_config(config)
    return config


def load_stream_config(config_path: str | Path | None = None) -> StreamConfig:
    """Load stream settings from YAML, falling back to built-in defaults.

    Args:
        config_path: Path to a YAML file. When omitted, uses the ``config.yml``
            located next to this module.

    Returns:
        A validated configuration. Built-in defaults are returned when the file
        is missing, cannot be read, is malformed, or contains invalid values.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    try:
        raw_config = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return DEFAULT_STREAM_CONFIG
    except OSError as error:
        logger.warning("Could not read stream configuration %s: %s", path, error)
        return DEFAULT_STREAM_CONFIG

    if yaml is None:
        logger.warning(
            "PyYAML is unavailable; using built-in stream configuration defaults"
        )
        return DEFAULT_STREAM_CONFIG

    try:
        parsed_config = yaml.safe_load(raw_config)
        if not isinstance(parsed_config, dict):
            raise ValueError("the YAML document must be a mapping")
        config_data = parsed_config.get("stream_config")
        if not isinstance(config_data, dict):
            raise ValueError("stream_config must be a mapping")
        return _read_config_values(config_data)
    except (ValueError, yaml.YAMLError) as error:
        logger.warning("Invalid stream configuration %s: %s", path, error)
        return DEFAULT_STREAM_CONFIG


class MJPEGStream:
    """Expose the newest camera frames as an MJPEG byte iterator.

    The stream does not start or stop the camera. This permits multiple stream
    consumers to share the same camera instance and leaves camera ownership to
    the application.

    Args:
        camera: Running camera that provides JPEG frames.
        boundary: Multipart boundary without the leading ``--``. Overrides
            ``config.yml``.
        timeout: Maximum wait time for each camera frame, in seconds. Overrides
            ``config.yml``.
        config_path: Optional path to a YAML configuration file.

    Attributes:
        frames_sent: Number of multipart JPEG parts yielded by the stream.
        last_frame_number: Identifier of the most recently yielded frame.
    """

    def __init__(
        self,
        camera: JPEGCamera,
        *,
        boundary: str | None = None,
        timeout: float | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """Initialize an MJPEG stream without taking ownership of the camera.

        Raises:
            ValueError: If ``boundary`` is invalid or ``timeout`` is not
                greater than zero.
        """
        loaded_config = load_stream_config(config_path)
        config = StreamConfig(
            boundary=loaded_config.boundary if boundary is None else boundary,
            timeout=loaded_config.timeout if timeout is None else timeout,
        )
        _validate_stream_config(config)

        self._camera = camera
        self._config = config
        self._boundary = config.boundary
        self._timeout = config.timeout
        self.frames_sent = 0
        self.last_frame_number: int | None = None

    @property
    def content_type(self) -> str:
        """str: Content type for an HTTP response carrying this stream."""
        return f"multipart/x-mixed-replace; boundary={self._boundary}"

    @property
    def config(self) -> StreamConfig:
        """StreamConfig: Resolved configuration used by this stream."""
        return self._config

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
    boundary: str | None = None,
    timeout: float | None = None,
    config_path: str | Path | None = None,
) -> Iterator[bytes]:
    """Create a one-shot MJPEG iterator for a camera.

    Args:
        camera: Running camera that provides JPEG frames.
        boundary: Multipart boundary without the leading ``--``. Overrides
            ``config.yml``.
        timeout: Maximum wait time for each camera frame, in seconds. Overrides
            ``config.yml``.
        config_path: Optional path to a YAML configuration file.

    Returns:
        Iterator of encoded MJPEG multipart body parts.
    """
    return iter(
        MJPEGStream(
            camera,
            boundary=boundary,
            timeout=timeout,
            config_path=config_path,
        )
    )
