"""Borrowed GPU camera-frame contract for zero-copy consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from .formats import FrameFormat, MemoryType


class GpuFrameExpiredError(RuntimeError):
    """Raised when a consumer accesses a GPU frame after its lease expired."""


class GpuBufferHandle:
    """Checked borrowed wrapper around a ``Gst.Buffer`` or ``NvBufSurface``.

    The wrapper drops its reference when invalidated. Access through
    :meth:`get` therefore cannot silently reuse a buffer after capture has
    advanced to the next frame.
    """

    def __init__(self, resource: object, *, owner: object | None = None) -> None:
        """Wrap one non-null GPU resource and retain its lifetime owner.

        ``owner`` is commonly the ``Gst.Sample`` from which a ``Gst.Buffer``
        was borrowed. Keeping both references until invalidation prevents the
        buffer pool from recycling the payload while a consumer is using it.
        """
        if resource is None:
            raise ValueError("resource must not be None")

        self._resource: object | None = resource
        self._owner: object | None = owner
        self._lock = Lock()

    @property
    def valid(self) -> bool:
        """Whether the wrapped resource remains available."""
        with self._lock:
            return self._resource is not None

    def get(self) -> object:
        """Return the resource or raise after lease invalidation."""
        with self._lock:
            if self._resource is None:
                raise GpuFrameExpiredError("GPU buffer handle has expired")

            return self._resource

    def invalidate(self) -> None:
        """Drop the borrowed resource reference permanently."""
        with self._lock:
            self._resource = None
            self._owner = None


@dataclass(slots=True)
class _FrameLease:
    """Mutable validity token shared by a frozen :class:`GpuFrame`."""

    valid: bool = True
    lock: Lock = field(default_factory=Lock)


@dataclass(frozen=True, slots=True)
class GpuFrame:
    """One borrowed frame whose pixels remain in NVIDIA device memory.

    The capture source owns the underlying buffer. A consumer may use the
    payload only while :attr:`valid` is true and must finish before requesting
    or observing the next frame from the same source. Sources invalidate the
    previous frame when they publish its successor. Consumers must not retain,
    close, unref, or mutate ``buffer`` and must not close ``dmabuf_fd``.

    Exactly one payload representation is present: ``dmabuf_fd`` is a borrowed
    DMA-BUF file descriptor, while ``buffer`` is a checked
    :class:`GpuBufferHandle` wrapping an opaque ``Gst.Buffer`` or
    ``NvBufSurface``. No NumPy image is exposed by this contract. Call
    :meth:`payload` immediately before use to validate the lease.

    ``timestamp_ns`` is the monotonic host timestamp assigned when capture
    made the frame available. ``capture_timestamp_ns`` may additionally carry
    a hardware or GStreamer timestamp in its native clock domain.
    """

    sequence: int
    timestamp_ns: int
    width: int
    height: int
    format: FrameFormat
    memory_type: MemoryType
    dmabuf_fd: int | None = None
    buffer: GpuBufferHandle | None = field(default=None, repr=False)
    capture_timestamp_ns: int | None = None
    _lease: _FrameLease = field(
        default_factory=_FrameLease,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Validate metadata and the exclusive borrowed payload."""
        integer_fields = ("sequence", "timestamp_ns", "width", "height")
        for field_name in integer_fields:
            value = getattr(self, field_name)

            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")

        if self.sequence <= 0:
            raise ValueError("sequence must be a positive integer")

        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

        if self.format is not FrameFormat.NV12_NVMM:
            raise ValueError("GpuFrame format must be FrameFormat.NV12_NVMM")

        if self.memory_type is not MemoryType.NVMM:
            raise ValueError("GpuFrame memory_type must be MemoryType.NVMM")

        has_fd = self.dmabuf_fd is not None
        has_buffer = self.buffer is not None

        if has_fd == has_buffer:
            raise ValueError("provide exactly one of dmabuf_fd or buffer")

        if has_fd and (
            isinstance(self.dmabuf_fd, bool)
            or not isinstance(self.dmabuf_fd, int)
            or self.dmabuf_fd < 0
        ):
            raise ValueError("dmabuf_fd must be a non-negative integer")

        if self.capture_timestamp_ns is not None and (
            isinstance(self.capture_timestamp_ns, bool)
            or not isinstance(self.capture_timestamp_ns, int)
            or self.capture_timestamp_ns < 0
        ):
            raise ValueError("capture_timestamp_ns must be non-negative or None")

    @property
    def output_format(self) -> FrameFormat:
        """Explicit output identity shared with the CPU frame contract."""
        return self.format

    @property
    def valid(self) -> bool:
        """Whether the borrowed payload may still be accessed."""
        with self._lease.lock:
            lease_valid = self._lease.valid

        buffer_valid = self.buffer is None or self.buffer.valid
        return lease_valid and buffer_valid

    def payload(self) -> int | object:
        """Return the borrowed payload after checking its frame lease.

        Raises:
            GpuFrameExpiredError: If the source has published a newer frame.
        """
        with self._lease.lock:
            if not self._lease.valid:
                raise GpuFrameExpiredError(
                    "GPU frame lease expired after a newer frame was published"
                )
            if self.dmabuf_fd is not None:
                return self.dmabuf_fd

            if self.buffer is None:  # Defensive guard for type narrowing.
                raise GpuFrameExpiredError("GPU frame payload is unavailable")

            return self.buffer.get()

    def invalidate(self) -> None:
        """Expire this borrowed frame.

        Capture-source implementations call this immediately before publishing
        the next frame. It is public so custom, model-agnostic sources can
        implement the same lifetime contract without private toolkit classes.
        """
        with self._lease.lock:
            self._lease.valid = False

        if self.buffer is not None:
            self.buffer.invalidate()
