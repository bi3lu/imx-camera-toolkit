"""Interfaces shared by camera capture backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CaptureBackend(ABC):
    """Open, read, and close a BGR camera capture source."""

    @property
    def argus_source(self) -> Any | None:
        """Return the live Argus GObject when the backend exposes one."""
        return None

    @abstractmethod
    def open(self) -> None:
        """Open the capture source.

        Raises:
            RuntimeError: If the source cannot be opened.
        """

    @abstractmethod
    def read(self) -> tuple[bool, Any | None]:
        """Read one owned BGR frame from the active source."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources retained by the capture source."""
