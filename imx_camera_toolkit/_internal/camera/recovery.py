"""Backend-independent state shared by CPU and GPU recovery."""


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
        self.last_error: Exception | None = None

    @property
    def attempts(self) -> int:
        """Number of admitted attempts since the last successful frame."""
        return self._attempts

    def begin_attempt(self, max_attempts: int) -> int | None:
        """Admit a retry and return its one-based index, or None if exhausted.

        Args:
            max_attempts: Non-negative limit from the camera's validated policy.
                Read on each admission so policy replacement retains its budget.
        """
        if self._attempts >= max_attempts:
            return None

        self._attempts += 1
        return self._attempts

    def record_frame_success(self) -> None:
        """Renew the retry budget without changing camera-owned diagnostics."""
        self._attempts = 0
