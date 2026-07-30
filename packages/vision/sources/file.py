"""OpenCV-backed file source for images and videos."""

from __future__ import annotations

import importlib
import logging
import math
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


def _load_opencv() -> Any:
    """Load OpenCV lazily so synthetic and non-OpenCV users remain supported."""
    return importlib.import_module("cv2")


class PlaybackMode(str, Enum):
    """Scheduling policy for video frames read from a file source."""

    UNBOUNDED = "unbounded"
    SOURCE_FPS = "source_fps"


class FileFrameSource:
    """Read frames from one image or a video file through OpenCV.

    Args:
        path: Existing image or video file.
        loop: Whether to restart at the beginning after the final frame.
        playback: Whether video frames are decoded as fast as possible or at
            the source file's declared frame rate. Images are unaffected.

    Raises:
        FileNotFoundError: If ``path`` does not exist when opening the source.
        RuntimeError: If OpenCV is unavailable or cannot open the file.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        loop: bool = False,
        playback: PlaybackMode = PlaybackMode.UNBOUNDED,
    ) -> None:
        """Store the file location without opening it."""
        if not isinstance(playback, PlaybackMode):
            raise ValueError("playback must be a PlaybackMode value")

        self._path = Path(path)
        self._loop = loop
        self._playback = playback
        self._cv2: Any | None = None
        self._capture: Any | None = None
        self._image: Any | None = None
        self._image_emitted = False
        self._opened = False
        self._exhausted = False
        self._close_requested = threading.Event()
        self._frame_interval: float | None = None
        self._next_frame_deadline: float | None = None

    @property
    def exhausted(self) -> bool:
        """bool: Whether a non-looping source has reached its final frame."""
        return self._exhausted

    def open(self) -> None:
        """Open the image or video file with the locally installed OpenCV."""
        if not self._path.is_file():
            raise FileNotFoundError(f"frame-source file does not exist: {self._path}")

        try:
            self._cv2 = _load_opencv()

        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required to read a file frame source"
            ) from error

        self._image_emitted = False
        self._exhausted = False
        self._opened = True
        self._close_requested.clear()
        self._frame_interval = None
        self._next_frame_deadline = None

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
        self._configure_playback(capture)

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
            if not self._wait_for_playback_slot():
                return None
            return cast(object, frame)

        if not self._loop:
            self._exhausted = True
            return None

        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = capture.read()

        if success:
            if not self._wait_for_playback_slot():
                return None
            return cast(object, frame)

        self._exhausted = True
        return None

    def close(self) -> None:
        """Release OpenCV decoding resources."""
        self._close_requested.set()
        if self._capture is not None:
            self._capture.release()

        self._capture = None
        self._image = None
        self._opened = False
        self._exhausted = True
        self._frame_interval = None
        self._next_frame_deadline = None

    def _configure_playback(self, capture: Any) -> None:
        """Resolve source FPS pacing when video playback requests it."""
        if self._playback is PlaybackMode.UNBOUNDED:
            return

        cv2 = self._cv2
        assert cv2 is not None
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))

        if not math.isfinite(source_fps) or source_fps <= 0:
            logger.warning(
                "File source %s has no valid FPS; using unbounded playback",
                self._path,
            )
            return

        self._frame_interval = 1.0 / source_fps
        self._next_frame_deadline = time.monotonic()

    def _wait_for_playback_slot(self) -> bool:
        """Wait until the next source-FPS frame deadline when configured.

        Returns:
            ``False`` when the source is closed while waiting.
        """
        interval = self._frame_interval
        deadline = self._next_frame_deadline

        if interval is None or deadline is None:
            return not self._close_requested.is_set()

        now = time.monotonic()

        if deadline > now and self._close_requested.wait(deadline - now):
            self._exhausted = True
            return False

        now = time.monotonic()
        next_deadline = deadline + interval
        self._next_frame_deadline = max(next_deadline, now + interval)
        return not self._close_requested.is_set()
