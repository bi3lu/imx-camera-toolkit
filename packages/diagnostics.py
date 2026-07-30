"""Read-only runtime diagnostics for toolkit deployments."""

from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiagnosticCheck:
    """One diagnostic check and its outcome."""

    name: str
    status: str
    detail: str


def _command_check(name: str, command: Sequence[str]) -> DiagnosticCheck:
    """Run a bounded diagnostic command without changing system state."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )

    except (OSError, subprocess.TimeoutExpired) as error:
        return DiagnosticCheck(name, "unavailable", str(error))

    if result.returncode == 0:
        return DiagnosticCheck(name, "ok", "available")

    detail = result.stderr.strip() or result.stdout.strip() or "command failed"
    return DiagnosticCheck(name, "error", detail)


def collect_diagnostics(include_hardware: bool = False) -> list[DiagnosticCheck]:
    """Collect environment and optional Jetson camera stack diagnostics.

    Args:
        include_hardware: Whether to inspect installed Argus and V4L2 tools.

    Returns:
        Read-only diagnostic results suitable for human or JSON output.
    """
    camera_config_path = Path(__file__).parent / "camera" / "config.yml"
    camera_config_exists = camera_config_path.is_file()
    checks = [
        DiagnosticCheck("python", "ok", sys.version.split()[0]),
        DiagnosticCheck("platform", "ok", platform.platform()),
        DiagnosticCheck(
            "opencv",
            "ok" if importlib.util.find_spec("cv2") else "unavailable",
            "importable" if importlib.util.find_spec("cv2") else "not installed",
        ),
        DiagnosticCheck(
            "camera_config",
            "ok" if camera_config_exists else "error",
            "present" if camera_config_exists else "missing",
        ),
    ]

    if include_hardware:
        checks.append(
            _command_check("nvarguscamerasrc", ("gst-inspect-1.0", "nvarguscamerasrc"))
        )
        checks.append(_command_check("v4l2", ("v4l2-ctl", "--list-devices")))

    return checks


def diagnostics_as_dict(include_hardware: bool = False) -> list[dict[str, object]]:
    """Return diagnostics in JSON-ready form."""
    results: list[dict[str, object]] = []

    for check in collect_diagnostics(include_hardware):
        results.append(asdict(check))

    return results
