"""Command-line interface for serving, diagnostics, and benchmarks."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from .benchmarks import (
    benchmark_capture,
    benchmark_cpu_capture,
    benchmark_cpu_capture_jpeg,
    benchmark_cpu_capture_model,
    benchmark_gpu_capture,
    benchmark_streaming,
)
from .camera.camera import Camera, CameraConfig, CameraTimeoutError
from .camera.gpu_camera import GpuCamera
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
        choices=("capture", "streaming", "camera", "jetson", "all"),
    )
    benchmark.add_argument("--frames", type=int, default=1_000)
    benchmark.add_argument("--json", action="store_true", help="emit JSON output")
    _add_camera_arguments(benchmark)
    benchmark.add_argument("--timeout", type=float, default=5.0)
    benchmark.add_argument("--sensor-mode", type=int)
    benchmark.add_argument(
        "--backend",
        choices=("cpu", "gpu", "all"),
        default="all",
        help="capture backend used by the jetson benchmark",
    )
    benchmark.add_argument(
        "--resolution",
        choices=("custom", "720p", "1080p", "all"),
        help="preset matrix; jetson defaults to both 720p and 1080p",
    )
    benchmark.add_argument(
        "--preview-only",
        action="store_true",
        help="measure only capture with JPEG preview enabled",
    )
    benchmark.add_argument(
        "--cpu-model",
        metavar="MODULE:CALLABLE",
        help="also benchmark an application CPU model callable",
    )

    preview = subcommands.add_parser(
        "preview",
        aliases=("serve",),
        help="start the simple browser camera preview",
    )
    preview.add_argument("--host", default="127.0.0.1")
    preview.add_argument("--port", type=int, default=8000)
    preview.add_argument(
        "--allow-remote",
        action="store_true",
        help="explicitly allow a non-loopback development bind",
    )
    preview.add_argument(
        "--field-mode",
        "--secure",
        action="store_true",
        help="require scoped auth, TLS, host checks, and rate limits",
    )
    preview.add_argument(
        "--token-file",
        type=Path,
        help="0600/0640 JSON file containing hashed bearer-token grants",
    )
    preview.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="accepted Host header in field mode; repeat for multiple hosts",
    )
    preview.add_argument("--tls-certfile", type=Path)
    preview.add_argument("--tls-keyfile", type=Path)
    preview.add_argument(
        "--behind-tls-proxy",
        action="store_true",
        help="require forwarded HTTPS from a loopback reverse proxy",
    )
    _add_camera_arguments(preview)
    _add_capture_backend_argument(preview)

    snapshot = subcommands.add_parser(
        "snapshot",
        help="capture one JPEG preview frame to a file",
    )
    snapshot.add_argument("path", nargs="?", default="snapshot.jpg")
    snapshot.add_argument("--timeout", type=float, default=5.0)
    _add_camera_arguments(snapshot)
    _add_capture_backend_argument(snapshot)

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
    _add_capture_backend_argument(camera_test)
    return parser


def _add_camera_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared static-camera options to one CLI subcommand."""
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)


def _add_capture_backend_argument(parser: argparse.ArgumentParser) -> None:
    """Add the stable CPU/GPU capture selection to a CLI command."""
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="cpu for BGR/OpenCV or gpu for NV12/NVMM capture",
    )


def _camera_config(
    arguments: argparse.Namespace,
    *,
    preview: bool,
    width: int | None = None,
    height: int | None = None,
) -> CameraConfig:
    """Resolve one CLI camera configuration from parsed command arguments."""
    resolved_width = arguments.width if width is None else width
    resolved_height = arguments.height if height is None else height
    return CameraConfig(
        sensor_id=arguments.sensor_id,
        sensor_mode=getattr(arguments, "sensor_mode", None),
        capture_width=resolved_width,
        capture_height=resolved_height,
        output_width=resolved_width,
        output_height=resolved_height,
        fps=arguments.fps,
        enable_preview=preview,
    )


def _benchmark_configs(arguments: argparse.Namespace) -> tuple[CameraConfig, ...]:
    """Resolve a custom size or the standard 720p/1080p report matrix."""
    resolution = arguments.resolution

    if resolution is None:
        resolution = (
            "all"
            if arguments.command == "benchmark" and arguments.target == "jetson"
            else "custom"
        )

    if resolution == "custom":
        return (_camera_config(arguments, preview=False),)

    presets = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
    }
    names = tuple(presets) if resolution == "all" else (resolution,)
    return tuple(
        _camera_config(
            arguments,
            preview=False,
            width=presets[name][0],
            height=presets[name][1],
        )
        for name in names
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


def _load_cpu_model(specification: str) -> Callable[[object], object]:
    """Load an application-owned CPU model callable for a hardware benchmark."""
    module_name, separator, attribute_name = specification.partition(":")

    if not separator or not module_name or not attribute_name:
        raise ValueError("--cpu-model must use MODULE:CALLABLE syntax")

    module = importlib.import_module(module_name)
    model = getattr(module, attribute_name, None)

    if not callable(model):
        raise ValueError("--cpu-model must resolve to a callable")

    return cast(Callable[[object], object], model)


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
                backend=arguments.backend,
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
            configs = _benchmark_configs(arguments)

            if arguments.preview_only:
                benchmark_results.extend(
                    benchmark_cpu_capture_jpeg(
                        arguments.frames,
                        timeout=arguments.timeout,
                        config=config,
                    ).as_dict()
                    for config in configs
                )

            else:
                for config in configs:
                    benchmark_results.extend(
                        (
                            benchmark_cpu_capture(
                                arguments.frames,
                                timeout=arguments.timeout,
                                config=config,
                            ).as_dict(),
                            benchmark_cpu_capture_jpeg(
                                arguments.frames,
                                timeout=arguments.timeout,
                                config=config,
                            ).as_dict(),
                        )
                    )

                    if arguments.cpu_model is not None:
                        model = _load_cpu_model(arguments.cpu_model)
                        benchmark_results.append(
                            benchmark_cpu_capture_model(
                                model,
                                arguments.frames,
                                timeout=arguments.timeout,
                                config=config,
                            ).as_dict()
                        )

        if arguments.target == "jetson":
            for config in _benchmark_configs(arguments):
                if arguments.backend in {"cpu", "all"}:
                    benchmark_results.append(
                        benchmark_cpu_capture(
                            arguments.frames,
                            timeout=arguments.timeout,
                            config=config,
                        ).as_dict()
                    )

                if arguments.backend in {"gpu", "all"}:
                    benchmark_results.append(
                        benchmark_gpu_capture(
                            arguments.frames,
                            timeout=arguments.timeout,
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
            backend=arguments.backend,
            host=arguments.host,
            port=arguments.port,
            allow_remote=arguments.allow_remote,
            field_mode=arguments.field_mode,
            token_file=arguments.token_file,
            allowed_hosts=tuple(arguments.allowed_host),
            behind_tls_proxy=arguments.behind_tls_proxy,
            ssl_certfile=arguments.tls_certfile,
            ssl_keyfile=arguments.tls_keyfile,
        )
        return 0

    camera_config = _camera_config(arguments, preview=True)
    camera = (
        GpuCamera(camera_config)
        if arguments.backend == "gpu"
        else Camera(camera_config)
    )

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
