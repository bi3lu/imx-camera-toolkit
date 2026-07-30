"""Command-line interface for serving, diagnostics, and benchmarks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence

from .benchmarks import benchmark_capture, benchmark_streaming
from .diagnostics import diagnostics_as_dict


def _build_parser() -> argparse.ArgumentParser:
    """Create the project command-line parser."""
    parser = argparse.ArgumentParser(prog="imx-camera-toolkit")
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
    benchmark.add_argument("target", choices=("capture", "streaming", "all"))
    benchmark.add_argument("--frames", type=int, default=1_000)
    benchmark.add_argument("--json", action="store_true", help="emit JSON output")

    serve = subcommands.add_parser("serve", help="start the FastAPI camera service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    return parser


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
        return (
            0
            if all(result["status"] == "ok" for result in diagnostic_results)
            else 1
        )

    if arguments.command == "benchmark":
        benchmark_results: list[Mapping[str, object]] = []

        if arguments.target in {"capture", "all"}:
            benchmark_results.append(benchmark_capture(arguments.frames).as_dict())

        if arguments.target in {"streaming", "all"}:
            benchmark_results.append(benchmark_streaming(arguments.frames).as_dict())

        _print_results(benchmark_results, arguments.json)
        return 0

    import uvicorn

    from packages.api.api import app

    uvicorn.run(app, host=arguments.host, port=arguments.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
