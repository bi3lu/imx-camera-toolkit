"""Non-blocking latest-value subscriptions and worker consumers."""

from __future__ import annotations

import threading
from collections.abc import Callable
from math import isfinite
from typing import Generic, TypeVar, cast

T = TypeVar("T")


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
    ) -> None:
        """Create an empty slot owned by one named consumer."""
        self._name = name
        self._unsubscribe = unsubscribe
        self._on_drop = on_drop
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
        with self._condition:
            if not self._closed:
                self._closed = True
                self._item = None
                self._has_item = False
                should_unsubscribe = True
                self._condition.notify_all()

        if should_unsubscribe:
            self._unsubscribe(self._name)

    def _publish(self, item: T) -> None:
        """Replace this slot without invoking consumer code."""
        dropped = False

        with self._condition:
            if self._closed:
                return

            if self._has_item:
                self._dropped_frames += 1
                dropped = True

            self._item = item
            self._has_item = True
            self._condition.notify()

        if dropped and self._on_drop is not None:
            self._on_drop(self._name, 1)

    def _close_from_hub(self) -> None:
        """Close without recursively unregistering from an already closed hub."""
        with self._condition:
            self._closed = True
            self._item = None
            self._has_item = False
            self._condition.notify_all()


class LatestFrameHub(Generic[T]):
    """Fan out each publication into one bounded slot per subscriber."""

    def __init__(
        self,
        on_drop: Callable[[str, int], None] | None = None,
    ) -> None:
        """Initialize an active hub with no subscribers or retained value."""
        self._lock = threading.Lock()
        self._subscriptions: dict[str, LatestFrameSubscription[T]] = {}
        self._latest: T | None = None
        self._closed = False
        self._on_drop = on_drop

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
            )
            latest = self._latest
            if latest is not None:
                subscription._publish(latest)
            self._subscriptions[normalized_name] = subscription

        return subscription

    def publish(self, item: T) -> None:
        """Replace every consumer slot without waiting for consumer work."""
        with self._lock:
            if self._closed:
                return

            self._latest = item
            subscriptions = tuple(self._subscriptions.values())

        for subscription in subscriptions:
            subscription._publish(item)

    def close(self) -> None:
        """Close all slots and release every retained value."""
        with self._lock:
            if self._closed:
                return

            self._closed = True
            self._latest = None
            subscriptions = tuple(self._subscriptions.values())
            self._subscriptions.clear()

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

        self._subscription = subscription
        self._handler = handler
        self._thread_name = thread_name or f"imx-consumer-{subscription.name}"
        self._wait_timeout = float(wait_timeout)
        self._running = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.processed_frames = 0
        self.failed_frames = 0
        self.last_error: Exception | None = None

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
                self.failed_frames += 1
                self.last_error = error

            else:
                self.processed_frames += 1

        self._running.clear()

    def __enter__(self) -> FrameConsumer[T]:
        """Start this consumer in a context-manager block."""
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop this consumer when leaving a context-manager block."""
        self.stop()


__all__ = ["FrameConsumer", "LatestFrameHub", "LatestFrameSubscription"]
