"""Optional OpenCV overlay rendering for detection results."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

from ..models import Frame, InferenceResult, OverlayFrame


@runtime_checkable
class Overlay(Protocol):
    """Render an optional image view from a source frame and model output."""

    def render(self, frame: Frame, result: InferenceResult) -> OverlayFrame:
        """Render one result without modifying the stored source frame."""
        ...


class OpenCVOverlay:
    """Draw detection boxes and labels onto a copied OpenCV-compatible image.

    Args:
        color: BGR box and label colour.
        thickness: OpenCV line thickness in pixels.
    """

    def __init__(
        self,
        *,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> None:
        """Validate rendering settings."""
        if (
            len(color) != 3
            or any(
                isinstance(component, bool)
                or not isinstance(component, int)
                or component < 0
                or component > 255
                for component in color
            )
        ):
            raise ValueError("color must contain three BGR values from 0 to 255")

        if (
            isinstance(thickness, bool)
            or not isinstance(thickness, int)
            or thickness <= 0
        ):
            raise ValueError("thickness must be a positive integer")

        self._color = color
        self._thickness = thickness

    def render(self, frame: Frame, result: InferenceResult) -> OverlayFrame:
        """Render detections on an image copy.

        Args:
            frame: Source frame whose image must support ``copy()``.
            result: Result produced for the same source frame.

        Returns:
            Independent overlay image associated with the frame.

        Raises:
            ValueError: If the result belongs to another frame.
            RuntimeError: If OpenCV or an image copy operation is unavailable.
        """
        if result.frame_sequence != frame.sequence:
            raise ValueError("overlay result does not match the source frame")

        try:
            cv2: Any = importlib.import_module("cv2")

        except ImportError as error:
            raise RuntimeError("OpenCV is required to render overlays") from error

        image = frame.image
        copy_method = getattr(image, "copy", None)

        if not callable(copy_method):
            raise RuntimeError("overlay frame image must provide copy()")

        rendered = copy_method()

        for detection in result.detections:
            box = detection.box
            start = (box.x, box.y)
            end = (box.x + box.width, box.y + box.height)
            label = f"{detection.label} {detection.confidence:.2f}"
            cv2.rectangle(rendered, start, end, self._color, self._thickness)
            cv2.putText(
                rendered,
                label,
                (box.x, max(0, box.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self._color,
                self._thickness,
                cv2.LINE_AA,
            )

        return OverlayFrame(frame_sequence=frame.sequence, image=rendered)
