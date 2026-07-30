"""Frame-source protocols and bundled deterministic source implementations."""

from .base import FrameSource
from .file import FileFrameSource
from .synthetic import SyntheticFrameSource

__all__ = ["FileFrameSource", "FrameSource", "SyntheticFrameSource"]
