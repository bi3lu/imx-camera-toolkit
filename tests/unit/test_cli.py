"""Unit tests for non-hardware CLI commands."""

from __future__ import annotations

import pytest

from packages import cli
from packages.cli import main


def test_cli_runs_capture_benchmark_as_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Benchmark command must provide structured output without a camera."""
    assert main(("benchmark", "capture", "--frames", "3", "--json")) == 0
    captured = capsys.readouterr()
    assert '"name": "capture"' in captured.out


def test_cli_info_and_hardware_test_use_structured_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Operational commands must remain testable without camera hardware."""
    monkeypatch.setattr(
        cli,
        "diagnostics_as_dict",
        lambda include_hardware: [
            {"name": "python", "status": "ok", "detail": "3.10"}
        ],
    )
    monkeypatch.setattr(cli, "run_camera_smoke_test", lambda **_: [])

    assert main(("info", "--json")) == 0
    assert main(("test", "--json")) == 0
    captured = capsys.readouterr()
    assert '"camera_profiles"' in captured.out
