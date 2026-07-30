"""OpenCV-backed file source for images and videos."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


class FileFrameSource:
    """Read frames from one image or a video file through OpenCV.

    Args:
        path: Existing image or video file.
        loop: Whether to restart at the beginning after the final frame.

    Raises:
        FileNotFoundError: If ``path`` does not exist when opening the source.
        RuntimeError: If OpenCV is unavailable or cannot open the file.
    """

    def __init__(self, path: str | Path, *, loop: bool = False) -> None:
        """Store the file location without opening it."""
        self._path = Path(path)
        self._loop = loop
        self._cv2: Any | None = None
        self._capture: Any | None = None
        self._image: Any | None = None
        self._image_emitted = False
        self._opened = False
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """bool: Whether a non-looping source has reached its final frame."""
        return self._exhausted

    def open(self) -> None:
        """Open the image or video file with the locally installed OpenCV."""
        if not self._path.is_file():
            raise FileNotFoundError(f"frame-source file does not exist: {self._path}")

        try:
            self._cv2 = importlib.import_module("cv2")

        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required to read a file frame source"
            ) from error

        self._image_emitted = False
        self._exhausted = False
        self._opened = True

        if self._path.suffix.lower() in IMAGE_SUFFIXES:
            image = self._cv2.imread(str(self._path), self._cv2.IMREAD_COLOR)

            if image is None:
                self.close()
                raise RuntimeError(f"could not decode image file: {self._path}")

            self._image = image
            return

        capture = self._cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            self.close()
            raise RuntimeError(f"could not open video file: {self._path}")

        self._capture = capture

    def read(self) -> object | None:
        """Read one image or video frame according to the configured loop mode."""
        if not self._opened:
            raise RuntimeError("file frame source is not open")

        if self._image is not None:
            if self._image_emitted and not self._loop:
                self._exhausted = True
                return None

            self._image_emitted = True
            return cast(object, self._image.copy())

        capture = self._capture
        cv2 = self._cv2

        if capture is None or cv2 is None:
            raise RuntimeError("file frame source has no active decoder")

        success, frame = capture.read()

        if success:
            return cast(object, frame)

        if not self._loop:
            self._exhausted = True
            return None

        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = capture.read()

        if success:
            return cast(object, frame)

        self._exhausted = True
        return None

    def close(self) -> None:
        """Release OpenCV decoding resources."""
        if self._capture is not None:
            self._capture.release()

        self._capture = None
        self._image = None
        self._opened = False
        self._exhausted = True
