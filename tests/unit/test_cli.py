"""Unit tests for non-hardware CLI commands."""

from __future__ import annotations

import pytest

from packages import cli
from packages.benchmarks import CameraBenchmarkResult
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


def test_cli_camera_benchmark_can_load_an_application_cpu_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Camera benchmark must expose all CPU paths without owning a model."""
    def model(image: object) -> object:
        """Return an image as a deterministic application model."""
        return image

    def result(name: str) -> CameraBenchmarkResult:
        """Build one deterministic CLI benchmark result."""
        return CameraBenchmarkResult(name, 1, 1.0, 1.0, 1, 0, 0.0)

    monkeypatch.setattr(cli, "benchmark_cpu_capture", lambda *_, **__: result("raw"))
    monkeypatch.setattr(
        cli,
        "benchmark_cpu_capture_jpeg",
        lambda *_, **__: result("jpeg"),
    )
    monkeypatch.setattr(
        cli,
        "benchmark_cpu_capture_model",
        lambda loaded_model, *_, **__: (
            result("model") if loaded_model is model else result("unexpected")
        ),
    )
    monkeypatch.setattr(cli, "_load_cpu_model", lambda _: model)

    assert (
        main(
            (
                "benchmark",
                "camera",
                "--frames",
                "1",
                "--cpu-model",
                "application:model",
                "--json",
            )
        )
        == 0
    )
    captured = capsys.readouterr()
    assert '"name": "raw"' in captured.out
    assert '"name": "jpeg"' in captured.out
    assert '"name": "model"' in captured.out
