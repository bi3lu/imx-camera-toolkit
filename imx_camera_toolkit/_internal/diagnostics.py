"""Read-only runtime diagnostics for toolkit deployments."""

from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .camera.camera import Camera, CameraConfig
from .camera.gpu_camera import GpuCamera


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
        checks.extend(
            _command_check(element, ("gst-inspect-1.0", element))
            for element in (
                "nvarguscamerasrc",
                "nvvidconv",
                "nvjpegenc",
                "appsink",
                "tee",
                "queue",
            )
        )
        checks.append(_command_check("v4l2", ("v4l2-ctl", "--list-devices")))

    return checks


def diagnostics_as_dict(include_hardware: bool = False) -> list[dict[str, object]]:
    """Return diagnostics in JSON-ready form."""
    results: list[dict[str, object]] = []

    for check in collect_diagnostics(include_hardware):
        results.append(asdict(check))

    return results


def run_camera_smoke_test(
    *,
    frames: int = 30,
    timeout: float = 5.0,
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    backend: str = "cpu",
) -> list[DiagnosticCheck]:
    """Open a physical camera, capture frames, and verify clean teardown.

    This test is intentionally opt-in because it accesses the connected CSI
    sensor. It opens the selected raw-frame camera, waits for ``frames``
    distinct source frames, reports the observed capture rate, and always
    attempts to release the backend before returning.

    Args:
        frames: Number of distinct raw frames to observe.
        timeout: Maximum wait for opening and for each expected frame.
        sensor_id: Zero-based CSI sensor identifier.
        width: Capture and output width in pixels.
        height: Capture and output height in pixels.
        fps: Requested capture rate in frames per second.
        backend: ``"cpu"`` for BGR/OpenCV or ``"gpu"`` for NV12/NVMM.

    Returns:
        Ordered diagnostic checks for open, first frame, frame rate, and close.
    """
    if frames <= 0:
        raise ValueError("frames must be greater than zero")

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be cpu or gpu")

    config = CameraConfig(
        sensor_id=sensor_id,
        capture_width=width,
        capture_height=height,
        output_width=width,
        output_height=height,
        fps=fps,
        enable_preview=False,
    )
    camera = GpuCamera(config) if backend == "gpu" else Camera(config)
    subscription = (
        camera.subscribe_latest("diagnostic-smoke-test")
        if isinstance(camera, GpuCamera)
        else None
    )
    checks: list[DiagnosticCheck] = []
    previous_frame_number = -1
    captured = 0

    try:
        camera.start()
        checks.append(DiagnosticCheck("camera_open", "ok", "opened"))
        started_at = time.monotonic()

        while captured < frames:
            if isinstance(camera, GpuCamera):
                if subscription is None:
                    raise RuntimeError("GPU diagnostic subscription is unavailable")
                frame = subscription.receive(timeout=timeout)
                if frame is None:
                    frame_number, image = previous_frame_number, None
                else:
                    try:
                        frame_number, image = frame.sequence, frame.payload()
                    finally:
                        frame.release()
            else:
                frame_number, image = camera.wait_for_raw_frame(
                    previous_frame_number,
                    timeout=timeout,
                )

            if image is None or frame_number == previous_frame_number:
                check_name = "first_frame" if captured == 0 else "capture_rate"
                checks.append(
                    DiagnosticCheck(
                        check_name,
                        "error",
                        f"timed out after {timeout:.1f}s waiting for a camera frame",
                    )
                )
                return checks

            previous_frame_number = frame_number
            captured += 1

            if captured == 1:
                checks.append(DiagnosticCheck("first_frame", "ok", "captured"))

        duration = max(time.monotonic() - started_at, 1e-9)
        observed_fps = captured / duration
        checks.append(
            DiagnosticCheck(
                "capture_rate",
                "ok",
                f"{observed_fps:.2f} FPS across {captured} frames",
            )
        )

    except Exception as error:
        checks.append(DiagnosticCheck("camera_open", "error", str(error)))

    finally:
        try:
            if subscription is not None:
                subscription.close()
            camera.stop()
            status = "ok" if not camera.running else "error"
            detail = "released" if status == "ok" else "camera is still running"
            checks.append(DiagnosticCheck("camera_release", status, detail))

        except Exception as error:
            checks.append(DiagnosticCheck("camera_release", "error", str(error)))

    return checks
