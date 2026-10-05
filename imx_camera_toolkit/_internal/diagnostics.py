"""Read-only runtime diagnostics for toolkit deployments."""

from __future__ import annotations

import importlib.util
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .camera.camera import Camera, CameraConfig
from .camera.gpu_camera import GpuCamera
from .camera.profiles import list_camera_profiles

DEFAULT_SENSOR_PROBE_IDS = (0, 1)
DEVICE_TREE_ROOT = Path("/proc/device-tree")
DEVICE_ROOT = Path("/dev")

_CAMERA_NODE_PATTERN = re.compile(r"(?:imx\d+|camera)", re.IGNORECASE)

_I2C_ERROR_PATTERN = re.compile(
    r"(?:i2c.*(?:-121|remote i/o)|(?:-121|remote i/o).*i2c)",
    re.IGNORECASE,
)

_SENSOR_PROBE_FAILURE_PATTERN = re.compile(r"probe failed", re.IGNORECASE)

_BUSY_PATTERN = re.compile(
    r"(?:device or resource busy|resource busy|already in use|camera.*busy"
    r"|failed to create capturesession)",
    re.IGNORECASE,
)

_MISSING_SENSOR_PATTERN = re.compile(
    r"(?:no cameras? available|no camera device|camera index.*(?:invalid|out of)"
    r"|sensor-id.*(?:invalid|not found)|invalid camera device)",
    re.IGNORECASE,
)

_NO_FRAME_PATTERN = re.compile(
    r"(?:no frames?|failed to get frame|buffer.*timeout|capture.*timeout)",
    re.IGNORECASE,
)


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


def _command_output_check(name: str, command: Sequence[str]) -> DiagnosticCheck:
    """Run a bounded inventory command and preserve its successful output."""
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

    output = result.stdout.strip() or result.stderr.strip()

    if result.returncode == 0:
        return DiagnosticCheck(name, "ok", output or "no devices reported")

    return DiagnosticCheck(name, "error", output or "command failed")


def _v4l2_nodes_check(device_root: Path = DEVICE_ROOT) -> DiagnosticCheck:
    """List V4L2 nodes without assigning them an Argus sensor identifier."""
    try:
        nodes = sorted(path.name for path in device_root.glob("video*"))

    except OSError as error:
        return DiagnosticCheck("v4l2_nodes", "unavailable", str(error))

    if not nodes:
        return DiagnosticCheck("v4l2_nodes", "warning", "no /dev/video* nodes")

    return DiagnosticCheck(
        "v4l2_nodes",
        "ok",
        ", ".join(f"/dev/{node}" for node in nodes),
    )


def _device_tree_camera_check(
    device_tree_root: Path = DEVICE_TREE_ROOT,
) -> DiagnosticCheck:
    """Report camera-related Device Tree nodes without opening hardware."""
    if not device_tree_root.is_dir():
        return DiagnosticCheck(
            "device_tree_cameras",
            "unavailable",
            f"{device_tree_root} is unavailable",
        )

    discovered: set[str] = set()

    try:
        candidate_files = (
            *device_tree_root.rglob("compatible"),
            *device_tree_root.rglob("badge"),
        )

        for candidate in candidate_files:
            try:
                value = candidate.read_bytes().decode("utf-8", errors="replace")

            except OSError:
                continue

            normalized_value = value.replace("\x00", ", ").strip(", ")

            if not _CAMERA_NODE_PATTERN.search(normalized_value):
                continue

            node = candidate.parent.relative_to(device_tree_root)
            discovered.add(f"/{node}: {normalized_value}")

    except OSError as error:
        return DiagnosticCheck("device_tree_cameras", "unavailable", str(error))

    if not discovered:
        return DiagnosticCheck(
            "device_tree_cameras",
            "warning",
            "no IMX/camera nodes found in Device Tree",
        )

    return DiagnosticCheck(
        "device_tree_cameras",
        "ok",
        "; ".join(sorted(discovered)),
    )


def _kernel_camera_check() -> DiagnosticCheck:
    """Recognize camera probe and I2C failures in the readable kernel log."""
    dmesg = shutil.which("dmesg")

    if dmesg is None:
        return DiagnosticCheck(
            "kernel_camera_log",
            "warning",
            "dmesg is unavailable; inspect the system journal for sensor errors",
        )

    try:
        result = subprocess.run(
            (dmesg, "--color=never"),
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )

    except (OSError, subprocess.TimeoutExpired) as error:
        return DiagnosticCheck("kernel_camera_log", "warning", str(error))

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "dmesg failed"
        return DiagnosticCheck("kernel_camera_log", "warning", detail)

    camera_lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if _CAMERA_NODE_PATTERN.search(line)
    ]

    failures = [line for line in camera_lines if _I2C_ERROR_PATTERN.search(line)]

    if failures:
        return DiagnosticCheck(
            "kernel_camera_log",
            "i2c-error",
            "hardware/driver sensor probe failure; Python cannot repair it: "
            + " | ".join(failures[-8:]),
        )

    probe_failures = [
        line for line in camera_lines if _SENSOR_PROBE_FAILURE_PATTERN.search(line)
    ]

    if probe_failures:
        return DiagnosticCheck(
            "kernel_camera_log",
            "hardware-error",
            "sensor probe failed at the hardware/driver layer: "
            + " | ".join(probe_failures[-8:]),
        )

    return DiagnosticCheck(
        "kernel_camera_log",
        "ok",
        "no camera I2C/probe failures found in readable kernel log",
    )


def _bounded_output(value: object, limit: int = 1200) -> str:
    """Normalize subprocess output for a compact diagnostic detail."""
    if value is None:
        return ""

    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")

    else:
        text = str(value)

    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[:limit]}..."


def _classify_argus_failure(
    output: str,
    *,
    timed_out: bool = False,
) -> tuple[str, str]:
    """Classify an Argus failure into an actionable operational category."""
    detail = output or "Argus probe failed without diagnostic output"

    if _I2C_ERROR_PATTERN.search(detail):
        return (
            "i2c-error",
            "sensor I2C probe failed at the hardware/driver layer; Python "
            f"cannot repair it: {detail}",
        )

    if _BUSY_PATTERN.search(detail):
        return "busy", f"camera is already in use: {detail}"

    if _MISSING_SENSOR_PATTERN.search(detail):
        return "missing", f"Argus sensor is not available: {detail}"

    if timed_out or _NO_FRAME_PATTERN.search(detail):
        return "no-frame", f"Argus opened no usable frame before timeout: {detail}"

    return "error", detail


def _run_argus_pipeline(
    name: str,
    description: str,
    command: Sequence[str],
    timeout: float,
) -> DiagnosticCheck:
    """Run one bounded Argus pipeline and classify its result."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    except subprocess.TimeoutExpired as error:
        output = _bounded_output(error.stderr or error.stdout)
        status, detail = _classify_argus_failure(output, timed_out=True)
        return DiagnosticCheck(name, status, f"{description}; {detail}")

    except OSError as error:
        return DiagnosticCheck(name, "unavailable", f"{description}; {error}")

    output = _bounded_output(result.stderr or result.stdout)

    if result.returncode == 0:
        return DiagnosticCheck(name, "ok", f"{description}; captured one frame")

    status, detail = _classify_argus_failure(output)
    return DiagnosticCheck(name, status, f"{description}; {detail}")


def probe_argus_sensors(
    sensor_ids: Sequence[int] = DEFAULT_SENSOR_PROBE_IDS,
    *,
    timeout: float = 5.0,
) -> list[DiagnosticCheck]:
    """Probe selected Argus identifiers against curated operating modes.

    V4L2 node numbers are intentionally not consulted. Every selected Argus
    ``sensor-id`` is bound explicitly to every curated camera profile and must
    deliver one frame within the supplied timeout.

    Args:
        sensor_ids: Argus identifiers to test independently.
        timeout: Per-profile process timeout in seconds.

    Returns:
        One availability result per sensor identifier followed by successful
        sensor mode probes for each curated profile.

    Raises:
        ValueError: If no identifiers are supplied or a value is invalid.
    """
    if not sensor_ids:
        raise ValueError("at least one sensor_id is required")

    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    if any(
        isinstance(sensor_id, bool) or not isinstance(sensor_id, int) or sensor_id < 0
        for sensor_id in sensor_ids
    ):
        raise ValueError("sensor_ids must contain non-negative integers")

    executable = shutil.which("gst-launch-1.0")

    if executable is None:
        return [
            DiagnosticCheck(
                "argus_sensor_probe",
                "unavailable",
                "gst-launch-1.0 is required to probe Argus sensors",
            )
        ]

    checks: list[DiagnosticCheck] = []

    for sensor_id in dict.fromkeys(sensor_ids):
        sensor_description = f"sensor-id={sensor_id}"
        sensor_check = _run_argus_pipeline(
            f"argus_sensor_{sensor_id}",
            sensor_description,
            (
                executable,
                "-q",
                "nvarguscamerasrc",
                f"sensor-id={sensor_id}",
                "num-buffers=1",
                "!",
                "fakesink",
                "sync=false",
            ),
            timeout,
        )
        checks.append(sensor_check)

        if sensor_check.status != "ok":
            continue

        for profile in list_camera_profiles():
            config = profile.config_for_sensor(sensor_id)
            name = f"argus_mode_{sensor_id}_{profile.name}"
            mode = (
                "automatic" if config.sensor_mode is None else str(config.sensor_mode)
            )
            description = (
                f"sensor-id={sensor_id}, profile={profile.name}, "
                f"sensor-mode={mode}, {config.capture_width}x"
                f"{config.capture_height}@{config.fps} FPS"
            )
            source_properties = (
                () if config.sensor_mode is None else (f"sensor-mode={mode}",)
            )
            command = (
                executable,
                "-q",
                "nvarguscamerasrc",
                f"sensor-id={sensor_id}",
                *source_properties,
                "num-buffers=1",
                "!",
                "video/x-raw(memory:NVMM),"
                f"width={config.capture_width},height={config.capture_height},"
                f"framerate={config.fps}/1",
                "!",
                "fakesink",
                "sync=false",
            )
            checks.append(_run_argus_pipeline(name, description, command, timeout))

    return checks


def collect_diagnostics(
    include_hardware: bool = False,
    *,
    probe_sensors: bool = False,
    sensor_ids: Sequence[int] = DEFAULT_SENSOR_PROBE_IDS,
    probe_timeout: float = 5.0,
) -> list[DiagnosticCheck]:
    """Collect environment and optional Jetson camera stack diagnostics.

    Args:
        include_hardware: Whether to inspect installed Argus and V4L2 tools.
        probe_sensors: Whether to open selected Argus sensors for one frame.
        sensor_ids: Argus identifiers tested when probing is enabled.
        probe_timeout: Per-profile Argus probe timeout in seconds.

    Returns:
        Read-only diagnostic results suitable for human or JSON output.
    """
    if probe_sensors and not include_hardware:
        raise ValueError("probe_sensors requires include_hardware=True")

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
                "fakesink",
                "tee",
                "queue",
            )
        )
        checks.extend(
            (
                _command_output_check("v4l2", ("v4l2-ctl", "--list-devices")),
                _v4l2_nodes_check(),
                _device_tree_camera_check(),
                DiagnosticCheck(
                    "sensor_id_mapping",
                    "ok",
                    "V4L2 node numbers are inventory only and are not mapped "
                    "to Argus sensor-id values",
                ),
            )
        )

        if probe_sensors:
            checks.extend(probe_argus_sensors(sensor_ids, timeout=probe_timeout))

        # Read this after optional probes so fresh driver/I2C errors are visible.
        checks.append(_kernel_camera_check())

    return checks


def diagnostics_as_dict(
    include_hardware: bool = False,
    *,
    probe_sensors: bool = False,
    sensor_ids: Sequence[int] = DEFAULT_SENSOR_PROBE_IDS,
    probe_timeout: float = 5.0,
) -> list[dict[str, object]]:
    """Return diagnostics in JSON-ready form."""
    results: list[dict[str, object]] = []

    for check in collect_diagnostics(
        include_hardware,
        probe_sensors=probe_sensors,
        sensor_ids=sensor_ids,
        probe_timeout=probe_timeout,
    ):
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
