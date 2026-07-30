"""Public data models returned by the camera package."""

from .frame import CameraFrame, Frame
from .stats import CameraStats

__all__ = ["CameraFrame", "CameraStats", "Frame"]
