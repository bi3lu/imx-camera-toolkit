"""Exposure-bracket fusion for sensors without native HDR modes."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..errors import CameraConfigurationError, CameraDependencyError

cv2_module: Any | None
np_module: Any | None

try:
    cv2_module = importlib.import_module("cv2")
    np_module = importlib.import_module("numpy")

except ImportError:
    cv2_module = None
    np_module = None


@dataclass(frozen=True)
class SoftwareHDRSettings:
    """Settings for exposure-bracket fusion performed on the Jetson.

    Args:
        enabled: Whether software HDR processing is active.
        base_exposure_us: Middle bracket exposure in microseconds.
        settle_frames: Frames discarded after each exposure change so the
            sensor can settle before the bracket frame is captured.
    """

    enabled: bool = False
    base_exposure_us: int = 5_000
    settle_frames: int = 2

    def __post_init__(self) -> None:
        """Validate software HDR settings."""
        if not isinstance(self.enabled, bool):
            raise CameraConfigurationError("software HDR enabled must be a boolean")

        if isinstance(self.base_exposure_us, bool) or not isinstance(
            self.base_exposure_us, int
        ):
            raise CameraConfigurationError(
                "software HDR base exposure must be an integer"
            )

        if self.base_exposure_us <= 0:
            raise CameraConfigurationError(
                "software HDR base exposure must be greater than zero"
            )

        if isinstance(self.settle_frames, bool) or not isinstance(
            self.settle_frames, int
        ):
            raise CameraConfigurationError(
                "software HDR settle frames must be an integer"
            )

        if not 0 <= self.settle_frames <= 10:
            raise CameraConfigurationError(
                "software HDR settle frames must be between 0 and 10"
            )


class SoftwareHDRProcessor:
    """Collect three exposure brackets and fuse them into one BGR frame."""

    bracket_ev = (-2, 0, 2)

    def __init__(self, settings: SoftwareHDRSettings, max_exposure_us: int) -> None:
        """Initialize a bracket sequence constrained by the capture period.

        Raises:
            CameraDependencyError: If JetPack OpenCV or NumPy is unavailable.
        """
        if cv2_module is None or np_module is None:
            raise CameraDependencyError(
                "Software HDR requires the JetPack OpenCV and NumPy packages"
            )

        self._settings = settings
        self._exposures_us = tuple(
            min(
                max_exposure_us,
                max(100, round(settings.base_exposure_us * (2**exposure_ev))),
            )
            for exposure_ev in self.bracket_ev
        )
        self._frames: list[Any] = []
        self._bracket_index = 0
        self._settle_remaining = settings.settle_frames
        self._merge = cv2_module.createMergeMertens()

    @property
    def exposures_us(self) -> tuple[int, int, int]:
        """tuple[int, int, int]: Resolved -2 EV, 0 EV, and +2 EV exposures."""
        return self._exposures_us

    def start(self, set_exposure: Callable[[int], None]) -> None:
        """Select the first bracket exposure before capture begins."""
        set_exposure(self._exposures_us[0])

    def process(
        self,
        frame: Any,
        set_exposure: Callable[[int], None],
    ) -> Any | None:
        """Consume one BGR frame and return a fused frame when complete."""
        if self._settle_remaining > 0:
            self._settle_remaining -= 1
            return None

        self._frames.append(frame)
        self._bracket_index += 1

        if self._bracket_index < len(self._exposures_us):
            set_exposure(self._exposures_us[self._bracket_index])
            self._settle_remaining = self._settings.settle_frames
            return None

        merged = self._merge.process(self._frames)

        assert np_module is not None

        output = np_module.clip(merged * 255.0, 0, 255).astype(np_module.uint8)

        self._frames.clear()
        self._bracket_index = 0
        self._settle_remaining = self._settings.settle_frames

        set_exposure(self._exposures_us[0])
        return output
