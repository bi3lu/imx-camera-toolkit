"""Backend-independent state shared by CPU and GPU recovery."""

from __future__ import annotations

import re

CAMERA_STATES = frozenset({"stopped", "starting", "running", "recovering", "failed"})

_I2C_FAILURE_PATTERN = re.compile(
    r"(?:i2c.*(?:-121|remote i/o)|(?:-121|remote i/o).*i2c|" r"(?:^|\D)-121(?:\D|$))",
    re.IGNORECASE,
)
_BUSY_PATTERN = re.compile(
    r"(?:already\s*_?\s*allocated|device or resource busy|resource busy|"
    r"already in use|failed to create capturesession)",
    re.IGNORECASE,
)
_MISSING_PATTERN = re.compile(
    r"(?:no cameras? available|invalid camera device|sensor-id.*not found)",
    re.IGNORECASE,
)


def _error_chain_text(error: object) -> str:
    """Flatten a wrapped exception chain for stable failure classification."""
    if not isinstance(error, BaseException):
        return str(error)

    messages: list[str] = []
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return ": ".join(message for message in messages if message)


def classify_camera_failure(error: object) -> str:
    """Classify a camera error for recovery and health diagnostics."""
    detail = _error_chain_text(error)
    if _I2C_FAILURE_PATTERN.search(detail):
        return "i2c"
    if _BUSY_PATTERN.search(detail):
        return "busy"
    if _MISSING_PATTERN.search(detail):
        return "sensor-missing"
    return "capture"


def is_permanent_sensor_failure(error: object) -> bool:
    """Return whether retrying cannot repair the reported sensor failure."""
    return classify_camera_failure(error) == "i2c"


def is_non_retryable_camera_failure(error: object) -> bool:
    """Return whether automatic backend reopen attempts should stop."""
    return classify_camera_failure(error) in {"busy", "i2c"}


class RecoveryController:
    """Track a consecutive retry budget and the last reported recovery error.

    The owning camera serializes budget operations with its statistics lock
    or initializes them before starting its capture worker.
    This controller performs no I/O or waiting and owns no capture resources.
    Error clearing remains explicit so each camera preserves its diagnostic
    contract independently of retry admission.
    """

    def __init__(self) -> None:
        """Initialize an unused budget with no reported recovery error."""
        self._attempts = 0
        self._consecutive_failed_restarts = 0
        self._restart_pending_frame = False
        self._state = "stopped"
        self._last_failure_reason: str | None = None
        self._failure_kind: str | None = None
        self.last_error: Exception | None = None

    @property
    def attempts(self) -> int:
        """Number of admitted attempts since the last successful frame."""
        return self._attempts

    @property
    def state(self) -> str:
        """Current camera lifecycle/recovery state."""
        return self._state

    @property
    def last_failure_reason(self) -> str | None:
        """Most recent failure text, retained after later recovery."""
        return self._last_failure_reason

    @property
    def failure_kind(self) -> str | None:
        """Stable category assigned to the most recent failure."""
        return self._failure_kind

    @property
    def consecutive_failed_restarts(self) -> int:
        """Restart attempts that have not yet produced a valid frame."""
        return self._consecutive_failed_restarts

    def set_state(self, state: str) -> None:
        """Set a validated lifecycle state owned by the camera."""
        if state not in CAMERA_STATES:
            raise ValueError(f"unsupported camera state: {state}")
        self._state = state

    def record_failure(self, error: BaseException) -> None:
        """Retain the latest failure without changing the retry budget."""
        self._last_failure_reason = _error_chain_text(error) or type(error).__name__
        self._failure_kind = classify_camera_failure(error)

    def record_restart_failure(self, error: BaseException) -> None:
        """Count a backend restart that failed before producing a frame."""
        self._restart_pending_frame = False
        self._consecutive_failed_restarts += 1
        self.last_error = (
            error if isinstance(error, Exception) else RuntimeError(str(error))
        )
        self.record_failure(error)

    def record_restart_opened(self) -> None:
        """Mark an opened replacement as unverified until its first frame."""
        self._restart_pending_frame = True

    def begin_attempt(self, max_attempts: int) -> int | None:
        """Admit a retry and return its one-based index, or None if exhausted.

        Args:
            max_attempts: Non-negative limit from the camera's validated policy.
                Read on each admission so policy replacement retains its budget.
        """
        if self._restart_pending_frame:
            self._restart_pending_frame = False
            self._consecutive_failed_restarts += 1

        if self._attempts >= max_attempts:
            return None

        self._attempts += 1
        return self._attempts

    def record_frame_success(self) -> None:
        """Renew the retry budget without changing camera-owned diagnostics."""
        self._attempts = 0
        self._consecutive_failed_restarts = 0
        self._restart_pending_frame = False
        self._state = "running"
