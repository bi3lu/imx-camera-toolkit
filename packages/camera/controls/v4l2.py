"""V4L2 runtime controls for Jetson CSI sensors."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Sequence

from .argus import parse_argus_property


class V4L2Controls:
    """Apply exposure and gain updates through one V4L2 video device."""

    def __init__(self, sensor_id: int) -> None:
        """Associate controls with the video device for one Argus sensor."""
        self._device = f"/dev/video{sensor_id}"

    def apply_manual_controls(
        self,
        current: Sequence[str],
        requested: Sequence[str],
    ) -> None:
        """Apply changed exposure and gain ranges without restarting capture.

        JetPack 6 can pass an invalid maximum range through dynamic
        ``nvarguscamerasrc`` updates. V4L2 bypasses that path while retaining
        the active preview.
        """
        current_values = dict(parse_argus_property(value) for value in current)
        requested_values = dict(parse_argus_property(value) for value in requested)
        controls: list[str] = []

        exposure_range = requested_values.get("exposuretimerange")

        if (
            exposure_range != current_values.get("exposuretimerange")
            and exposure_range is not None
        ):
            exposure_ns = int(str(exposure_range).split()[0])
            controls.append(f"exposure={exposure_ns // 1_000}")

        gain_range = requested_values.get("gainrange")

        if gain_range != current_values.get("gainrange") and gain_range is not None:
            gain = float(str(gain_range).split()[0])
            controls.append(f"gain={round(200 * math.log10(gain))}")

        self.set_controls(controls)

    def set_exposure(self, exposure_us: int) -> None:
        """Set one fixed exposure duration in microseconds."""
        self.set_controls([f"exposure={exposure_us}"])

    def set_controls(self, controls: Sequence[str]) -> None:
        """Apply one atomic list of V4L2 control assignments.

        Args:
            controls: ``v4l2-ctl`` assignments without the ``--set-ctrl``
                prefix.

        Raises:
            RuntimeError: If the V4L2 control device cannot apply a setting.
        """
        if not controls:
            return

        command = ["v4l2-ctl", "-d", self._device, f"--set-ctrl={','.join(controls)}"]

        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )

        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(
                f"Could not apply V4L2 camera controls: {error}"
            ) from error

        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Could not apply V4L2 camera controls: {message}")
