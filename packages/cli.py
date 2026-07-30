"""Command-line interface for serving, diagnostics, and benchmarks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .benchmarks import (
    benchmark_camera_capture,
    benchmark_capture,
    benchmark_streaming,
)
from .camera.camera import Camera, CameraConfig, CameraTimeoutError
from .diagnostics import diagnostics_as_dict, run_camera_smoke_test


def _build_parser() -> argparse.ArgumentParser:
    """Create the project command-line parser."""
    parser = argparse.ArgumentParser(prog="imx-camera")
    subcommands = parser.add_subparsers(dest="command", required=True)

    diagnose = subcommands.add_parser(
        "diagnose",
        help="inspect runtime prerequisites",
    )
    diagnose.add_argument(
        "--hardware",
        action="store_true",
        help="inspect Argus and V4L2 tools",
    )
    diagnose.add_argument("--json", action="store_true", help="emit JSON output")

    benchmark = subcommands.add_parser(
        "benchmark",
        help="run deterministic microbenchmarks",
    )
    benchmark.add_argument(
        "target",
        choices=("capture", "streaming", "camera", "all"),
    )
    benchmark.add_argument("--frames", type=int, default=1_000)
    benchmark.add_argument("--json", action="store_true", help="emit JSON output")
    _add_camera_arguments(benchmark)
    benchmark.add_argument(
        "--preview-only",
        action="store_true",
        help="measure only capture with JPEG preview enabled",
    )

    preview = subcommands.add_parser(
        "preview",
        aliases=("serve",),
        help="start the simple browser camera preview",
    )
    preview.add_argument("--host", default="0.0.0.0")
    preview.add_argument("--port", type=int, default=8000)
    _add_camera_arguments(preview)

    snapshot = subcommands.add_parser(
        "snapshot",
        help="capture one JPEG preview frame to a file",
    )
    snapshot.add_argument("path", nargs="?", default="snapshot.jpg")
    snapshot.add_argument("--timeout", type=float, default=5.0)
    _add_camera_arguments(snapshot)

    info = subcommands.add_parser(
        "info",
        help="show runtime and profile information",
    )
    info.add_argument("--hardware", action="store_true")
    info.add_argument("--json", action="store_true", help="emit JSON output")

    camera_test = subcommands.add_parser(
        "test",
        help="open a connected camera, capture frames, and release it",
    )
    camera_test.add_argument("--frames", type=int, default=30)
    camera_test.add_argument("--timeout", type=float, default=5.0)
    camera_test.add_argument("--json", action="store_true", help="emit JSON output")
    _add_camera_arguments(camera_test)
    return parser


def _add_camera_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared static-camera options to one CLI subcommand."""
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)


def _camera_config(arguments: argparse.Namespace, *, preview: bool) -> CameraConfig:
    """Resolve one CLI camera configuration from parsed command arguments."""
    return CameraConfig(
        sensor_id=arguments.sensor_id,
        capture_width=arguments.width,
        capture_height=arguments.height,
        output_width=arguments.width,
        output_height=arguments.height,
        fps=arguments.fps,
        enable_preview=preview,
    )


def _print_results(
    results: Sequence[Mapping[str, object]],
    as_json: bool,
) -> None:
    """Print structured output in JSON or a compact human-readable form."""
    if as_json:
        print(json.dumps(results, indent=2))
        return

    for result in results:
        detail = result.get("detail") or result.get("frames_per_second")
        print(f"{result['name']}: {result.get('status', 'ok')} ({detail})")


def _has_errors(results: Sequence[Mapping[str, object]]) -> bool:
    """Return whether a structured diagnostic result contains a failure."""
    return any(result.get("status") in {"error", "unavailable"} for result in results)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Optional arguments excluding the executable name.

    Returns:
        Process exit status.
    """
    arguments = _build_parser().parse_args(argv)

    if arguments.command == "diagnose":
        diagnostic_results = diagnostics_as_dict(arguments.hardware)
        _print_results(diagnostic_results, arguments.json)
        return 1 if _has_errors(diagnostic_results) else 0

    if arguments.command == "info":
        from .camera.profiles import list_camera_profiles

        info_results = diagnostics_as_dict(arguments.hardware)
        profiles = ", ".join(profile.name for profile in list_camera_profiles())
        info_results.append(
            {
                "name": "camera_profiles",
                "status": "ok",
                "detail": profiles,
            }
        )
        _print_results(info_results, arguments.json)
        return 1 if _has_errors(info_results) else 0

    if arguments.command == "test":
        test_results = [
            {
                "name": result.name,
                "status": result.status,
                "detail": result.detail,
            }
            for result in run_camera_smoke_test(
                frames=arguments.frames,
                timeout=arguments.timeout,
                sensor_id=arguments.sensor_id,
                width=arguments.width,
                height=arguments.height,
                fps=arguments.fps,
            )
        ]
        _print_results(test_results, arguments.json)
        return 1 if _has_errors(test_results) else 0

    if arguments.command == "benchmark":
        benchmark_results: list[Mapping[str, object]] = []

        if arguments.target in {"capture", "all"}:
            benchmark_results.append(benchmark_capture(arguments.frames).as_dict())

        if arguments.target in {"streaming", "all"}:
            benchmark_results.append(benchmark_streaming(arguments.frames).as_dict())

        if arguments.target == "camera":
            preview_modes = (True,) if arguments.preview_only else (False, True)
            config = _camera_config(arguments, preview=False)
            for preview_enabled in preview_modes:
                benchmark_results.append(
                    benchmark_camera_capture(
                        arguments.frames,
                        preview=preview_enabled,
                        config=config,
                    ).as_dict()
                )

        _print_results(benchmark_results, arguments.json)
        return 0

    if arguments.command in {"preview", "serve"}:
        from imx_camera_toolkit.preview import preview

        preview(
            sensor_id=arguments.sensor_id,
            width=arguments.width,
            height=arguments.height,
            fps=arguments.fps,
            host=arguments.host,
            port=arguments.port,
        )
        return 0

    camera = Camera(_camera_config(arguments, preview=True))

    with camera:
        _, jpeg = camera.wait_for_jpeg(-1, timeout=arguments.timeout)
        if jpeg is None:
            raise CameraTimeoutError(
                "camera did not provide a JPEG frame before the timeout"
            )

        Path(arguments.path).write_bytes(jpeg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
