"""Unit tests for non-hardware CLI commands."""

from __future__ import annotations

import pytest

from packages.cli import main


def test_cli_runs_capture_benchmark_as_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Benchmark command must provide structured output without a camera."""
    assert main(("benchmark", "capture", "--frames", "3", "--json")) == 0
    captured = capsys.readouterr()
    assert '"name": "capture"' in captured.out
