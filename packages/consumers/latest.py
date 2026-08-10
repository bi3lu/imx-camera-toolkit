"""Non-blocking latest-value subscriptions and worker consumers."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from math import isfinite
from typing import Generic, TypeVar, cast

T = TypeVar("T")
logger = logging.getLogger(__name__)


class LatestFrameSubscription(Generic[T]):
    """A private single-slot view of the newest value published by a source.

    A subscription never queues history. When its slot has not yet been read,
    publishing a newer value replaces the old value and increments
    :attr:`dropped_frames`. Closing the subscription releases its retained
    reference and wakes a waiting worker.
    """

    def __init__(
        self,
        name: str,
        unsubscribe: Callable[[str], None],
        on_drop: Callable[[str, int], None] | None = None,
        release: Callable[[T], None] | None = None,
    ) -> None:
        """Create an empty slot owned by one named consumer."""
        self._name = name
        self._unsubscribe = unsubscribe
        self._on_drop = on_drop
        self._release = release
        self._condition = threading.Condition()
        self._item: T | None = None
        self._has_item = False
        self._closed = False
        self._dropped_frames = 0

    @property
    def name(self) -> str:
        """Stable consumer name used for per-consumer drop metrics."""
        return self._name

    @property
    def closed(self) -> bool:
        """Whether this subscription no longer accepts publications."""
        with self._condition:
            return self._closed

    @property
    def dropped_frames(self) -> int:
        """Number of unread values replaced by a newer publication."""
        with self._condition:
            return self._dropped_frames

    def receive(self, timeout: float | None = None) -> T | None:
        """Return and consume the current slot, waiting for one if necessary.

        ``None`` means that the timeout elapsed or the subscription was closed.
        Reading consumes only this subscriber's slot and never affects another
        consumer.
        """
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be a finite non-negative number or None")

        with self._condition:
            self._condition.wait_for(
                lambda: self._has_item or self._closed,
                timeout=None if timeout is None else float(timeout),
            )

            if not self._has_item or self._closed:
                return None

            item = cast(T, self._item)
            self._item = None
            self._has_item = False
            return item

    def close(self) -> None:
        """Unsubscribe, release the current slot, and wake a waiting worker."""
        should_unsubscribe = False
        item: T | None = None

        with self._condition:
            if not self._closed:
                self._closed = True

                if self._has_item:
                    item = self._item

                self._item = None
                self._has_item = False
                should_unsubscribe = True
                self._condition.notify_all()

        if item is not None:
            self.release(item)

        if should_unsubscribe:
            self._unsubscribe(self._name)

    def release(self, item: T) -> None:
        """Release an item after its consumer has finished processing it."""
        if self._release is not None:
            self._release(item)

    def record_drop(self, count: int = 1) -> None:
        """Record an item rejected after receipt but before processing."""
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("drop count must be a positive integer")

        with self._condition:
            self._dropped_frames += count

        if self._on_drop is not None:
            self._on_drop(self._name, count)

    def _publish(self, item: T) -> None:
        """Replace this slot without invoking consumer code."""
        dropped_item: T | None = None
        closed = False

        with self._condition:
            if self._closed:
                closed = True

            elif self._has_item:
                self._dropped_frames += 1
                dropped_item = self._item

            if not closed:
                self._item = item
                self._has_item = True
                self._condition.notify()

        if closed:
            self.release(item)
            return
        if dropped_item is not None:
            self.release(dropped_item)
        if dropped_item is not None and self._on_drop is not None:
            self._on_drop(self._name, 1)

    def _close_from_hub(self) -> None:
        """Close without recursively unregistering from an already closed hub."""
        item: T | None = None

        with self._condition:
            if self._has_item:
                item = self._item

            self._closed = True
            self._item = None
            self._has_item = False
            self._condition.notify_all()

        if item is not None:
            self.release(item)


class LatestFrameHub(Generic[T]):
    """Fan out each publication into one bounded slot per subscriber."""

    def __init__(
        self,
        on_drop: Callable[[str, int], None] | None = None,
        *,
        retain: Callable[[T], T] | None = None,
        release: Callable[[T], None] | None = None,
    ) -> None:
        """Initialize an active hub with no subscribers or retained value."""
        if (retain is None) != (release is None):
            raise ValueError("retain and release must be provided together")

        self._lock = threading.Lock()
        self._subscriptions: dict[str, LatestFrameSubscription[T]] = {}
        self._latest: T | None = None
        self._closed = False
        self._on_drop = on_drop
        self._retain = retain
        self._release = release

    def _retain_item(self, item: T) -> T:
        """Create an independent item lease when retention is configured."""
        return item if self._retain is None else self._retain(item)

    def _release_item(self, item: T) -> None:
        """Release one hub-owned item lease when configured."""
        if self._release is not None:
            self._release(item)

    @property
    def closed(self) -> bool:
        """Whether the source has closed every subscription."""
        with self._lock:
            return self._closed

    @property
    def subscriber_count(self) -> int:
        """Number of active, independently buffered consumers."""
        with self._lock:
            return len(self._subscriptions)

    @property
    def latest(self) -> T | None:
        """Newest published value retained by the hub, if any."""
        with self._lock:
            return self._latest

    def subscribe(self, name: str) -> LatestFrameSubscription[T]:
        """Create one named latest-value slot, initially containing latest."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("consumer name must be a non-empty string")

        normalized_name = name.strip()

        with self._lock:
            if self._closed:
                raise RuntimeError("latest-frame source is closed")
            if normalized_name in self._subscriptions:
                raise ValueError(
                    f"consumer name {normalized_name!r} is already subscribed"
                )

            subscription = LatestFrameSubscription[T](
                normalized_name,
                self._unsubscribe,
                self._on_drop,
                self._release,
            )
            latest = self._latest
            if latest is not None:
                subscription._publish(self._retain_item(latest))
            self._subscriptions[normalized_name] = subscription

        return subscription

    def publish(self, item: T) -> None:
        """Replace every consumer slot without waiting for consumer work."""
        previous: T | None = None

        with self._lock:
            if self._closed:
                return

            previous = self._latest
            self._latest = self._retain_item(item)
            subscriptions = tuple(self._subscriptions.values())
            retained_items = tuple(
                self._retain_item(item) for _ in subscriptions
            )

        if previous is not None:
            self._release_item(previous)

        for subscription, retained_item in zip(
            subscriptions,
            retained_items,
            strict=True,
        ):
            subscription._publish(retained_item)

    def close(self) -> None:
        """Close all slots and release every retained value."""
        latest: T | None = None

        with self._lock:
            if self._closed:
                return

            self._closed = True
            latest = self._latest
            self._latest = None
            subscriptions = tuple(self._subscriptions.values())
            self._subscriptions.clear()

        if latest is not None:
            self._release_item(latest)

        for subscription in subscriptions:
            subscription._close_from_hub()

    def _unsubscribe(self, name: str) -> None:
        """Remove a closed subscription without touching other slots."""
        with self._lock:
            self._subscriptions.pop(name, None)


class FrameConsumer(Generic[T]):
    """Run one potentially expensive frame callback on its own worker thread."""

    def __init__(
        self,
        subscription: LatestFrameSubscription[T],
        handler: Callable[[T], None],
        *,
        thread_name: str | None = None,
        wait_timeout: float = 0.25,
        on_error: Callable[[Exception], None] | None = None,
        error_log_interval: float = 5.0,
        initial_failure_backoff: float = 0.05,
        max_failure_backoff: float = 1.0,
        drop_exceptions: tuple[type[Exception], ...] = (),
    ) -> None:
        """Configure a worker without starting it."""
        if not callable(handler):
            raise TypeError("handler must be callable")

        if (
            isinstance(wait_timeout, bool)
            or not isinstance(wait_timeout, (int, float))
            or not isfinite(wait_timeout)
            or wait_timeout <= 0
        ):
            raise ValueError("wait_timeout must be a finite positive number")

        if on_error is not None and not callable(on_error):
            raise TypeError("on_error must be callable or None")

        for name, value in (
            ("error_log_interval", error_log_interval),
            ("initial_failure_backoff", initial_failure_backoff),
            ("max_failure_backoff", max_failure_backoff),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")

        if max_failure_backoff < initial_failure_backoff:
            raise ValueError(
                "max_failure_backoff must be at least initial_failure_backoff"
            )

        if not isinstance(drop_exceptions, tuple) or any(
            not isinstance(error_type, type)
            or not issubclass(error_type, Exception)
            for error_type in drop_exceptions
        ):
            raise TypeError("drop_exceptions must contain exception classes")

        self._subscription = subscription
        self._handler = handler
        self._thread_name = thread_name or f"imx-consumer-{subscription.name}"
        self._wait_timeout = float(wait_timeout)
        self._on_error = on_error
        self._error_log_interval = float(error_log_interval)
        self._initial_failure_backoff = float(initial_failure_backoff)
        self._max_failure_backoff = float(max_failure_backoff)
        self._drop_exceptions = drop_exceptions
        self._running = threading.Event()
        self._stop_requested = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.processed_frames = 0
        self.failed_frames = 0
        self.consecutive_failures = 0
        self.last_error: Exception | None = None
        self.last_failure: Exception | None = None
        self._last_error_log_at = 0.0
        self._suppressed_error_logs = 0

    @property
    def name(self) -> str:
        """Consumer name inherited from its source subscription."""
        return self._subscription.name

    @property
    def running(self) -> bool:
        """Whether the worker is accepting newest frames."""
        return self._running.is_set()

    @property
    def dropped_frames(self) -> int:
        """Number of frames replaced before this worker received them."""
        return self._subscription.dropped_frames

    @property
    def healthy(self) -> bool:
        """Whether the most recently handled frame completed successfully."""
        return self.last_error is None and self.consecutive_failures == 0

    @property
    def thread_ident(self) -> int | None:
        """Identifier of the dedicated worker thread once started."""
        thread = self._thread
        return None if thread is None else thread.ident

    def start(self) -> None:
        """Start the dedicated worker exactly once."""
        with self._lifecycle_lock:
            if self.running:
                return

            if self._subscription.closed:
                raise RuntimeError("cannot start a closed frame subscription")

            if self._thread is not None:
                raise RuntimeError("frame consumer cannot be restarted")

            self.last_error = None
            self.consecutive_failures = 0
            self._stop_requested.clear()
            self._running.set()
            self._thread = threading.Thread(
                target=self._run,
                name=self._thread_name,
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float | None = 3.0) -> bool:
        """Close the slot, join the worker, and report whether it stopped."""
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("timeout must be a finite non-negative number or None")

        with self._lifecycle_lock:
            self._running.clear()
            self._stop_requested.set()
            self._subscription.close()
            thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=None if timeout is None else float(timeout))

        return thread is None or not thread.is_alive()

    def _run(self) -> None:
        """Consume only the newest available slot until stopped."""
        while self.running:
            item = self._subscription.receive(self._wait_timeout)

            if item is None:
                if self._subscription.closed:
                    break

                continue

            try:
                self._handler(item)

            except Exception as error:
                if isinstance(error, self._drop_exceptions):
                    self._subscription.record_drop()

                else:
                    self.failed_frames += 1
                    self.consecutive_failures += 1
                    self.last_error = error
                    self.last_failure = error
                    self._report_error(error)
                    self._wait_after_failure()

            else:
                self.processed_frames += 1
                self.consecutive_failures = 0
                self.last_error = None
                self._last_error_log_at = 0.0
                self._suppressed_error_logs = 0

            finally:
                self._subscription.release(item)

        self._running.clear()

    def _report_error(self, error: Exception) -> None:
        """Log failures at a bounded rate and notify the optional callback."""
        now = time.monotonic()
        if (
            self._last_error_log_at == 0.0
            or now - self._last_error_log_at >= self._error_log_interval
        ):
            suffix = (
                ""
                if self._suppressed_error_logs == 0
                else f" ({self._suppressed_error_logs} similar errors suppressed)"
            )
            logger.error(
                "Frame consumer %s failed%s: %s",
                self.name,
                suffix,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
            self._last_error_log_at = now
            self._suppressed_error_logs = 0

        else:
            self._suppressed_error_logs += 1

        if self._on_error is not None:
            try:
                self._on_error(error)

            except Exception:
                logger.exception(
                    "Frame consumer %s on_error callback failed",
                    self.name,
                )

    def _wait_after_failure(self) -> None:
        """Apply bounded exponential backoff after consecutive failures."""
        if self._initial_failure_backoff == 0:
            return

        exponent = min(max(self.consecutive_failures - 1, 0), 30)
        delay = min(
            self._initial_failure_backoff * (2**exponent),
            self._max_failure_backoff,
        )
        self._stop_requested.wait(delay)

    def __enter__(self) -> FrameConsumer[T]:
        """Start this consumer in a context-manager block."""
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop this consumer when leaving a context-manager block."""
        self.stop()


__all__ = ["FrameConsumer", "LatestFrameHub", "LatestFrameSubscription"]
