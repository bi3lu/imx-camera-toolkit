"""Explicit frame-format and memory-domain identifiers."""

from __future__ import annotations

from enum import Enum


class FrameFormat(str, Enum):
    """Public output formats exposed by camera frame sources."""

    BGR_CPU = "BGR_CPU"
    NV12_NVMM = "NV12_NVMM"


class MemoryType(str, Enum):
    """Memory domain containing a frame payload."""

    CPU = "CPU"
    NVMM = "NVMM"
