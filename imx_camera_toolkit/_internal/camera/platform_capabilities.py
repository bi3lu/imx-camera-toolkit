"""Jetson platform capability declarations and local discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEVICE_TREE_MODEL_PATH = Path("/proc/device-tree/model")


@dataclass(frozen=True, slots=True)
class PlatformCapabilities:
    """Hardware capabilities relevant to camera pipeline selection.

    Attributes:
        model: Human-readable device model, when known.
        supports_nvenc: Whether the device has an NVENC hardware block. ``None``
            means the hardware capability is unknown and runtime plugin probing
            remains authoritative.
    """

    model: str | None = None
    supports_nvenc: bool | None = None

    def __post_init__(self) -> None:
        """Validate explicitly declared platform metadata."""
        if self.model is not None and (
            not isinstance(self.model, str) or not self.model.strip()
        ):
            raise ValueError("model must be a non-empty string or None")

        if self.supports_nvenc is not None and not isinstance(
            self.supports_nvenc, bool
        ):
            raise ValueError("supports_nvenc must be a boolean or None")


def detect_platform_capabilities(
    model_path: Path = DEVICE_TREE_MODEL_PATH,
) -> PlatformCapabilities:
    """Discover known encoder capabilities from the Device Tree model.

    Unknown or unreadable models deliberately return an unspecified NVENC
    capability. This preserves plugin-based selection on other Jetson devices
    while preventing false NVENC selection on Orin Nano.

    Args:
        model_path: Device Tree model file to inspect.

    Returns:
        Detected platform metadata and known hardware capabilities.
    """
    try:
        model = model_path.read_bytes().decode("utf-8").strip("\x00 \t\r\n")

    except (OSError, UnicodeDecodeError):
        return PlatformCapabilities()

    if not model:
        return PlatformCapabilities()

    supports_nvenc = False if "orin nano" in model.lower() else None
    return PlatformCapabilities(model=model, supports_nvenc=supports_nvenc)


__all__ = ["PlatformCapabilities"]
