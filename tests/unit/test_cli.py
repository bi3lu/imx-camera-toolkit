"""Unit tests for non-hardware CLI commands."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import cast

import pytest

from packages import cli
from packages.benchmarks import CameraBenchmarkResult
from packages.camera.config import CameraConfig
from packages.cli import main

preview_module = importlib.import_module("imx_camera_toolkit.preview")


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


def test_cli_jetson_benchmark_defaults_to_cpu_gpu_and_two_resolutions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One command must emit the complete 720p/1080p deployment matrix."""
    calls: list[tuple[str, int, int]] = []

    def run(backend: str, *args: object, **kwargs: object) -> CameraBenchmarkResult:
        config = cast(CameraConfig, kwargs["config"])
        width = config.output_width
        height = config.output_height
        calls.append((backend, width, height))
        return CameraBenchmarkResult(
            backend,
            1,
            1.0,
            1.0,
            1,
            0,
            0.0,
            width=width,
            height=height,
            backend=backend,
        )

    monkeypatch.setattr(
        cli,
        "benchmark_cpu_capture",
        lambda *args, **kwargs: run("cpu", *args, **kwargs),
    )
    monkeypatch.setattr(
        cli,
        "benchmark_gpu_capture",
        lambda *args, **kwargs: run("gpu", *args, **kwargs),
    )

    assert main(("benchmark", "jetson", "--frames", "1", "--json")) == 0

    assert calls == [
        ("cpu", 1280, 720),
        ("gpu", 1280, 720),
        ("cpu", 1920, 1080),
        ("gpu", 1920, 1080),
    ]
    assert '"process_cpu_percent"' in capsys.readouterr().out


def test_cli_preview_defaults_to_loopback_and_forwards_field_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI deployment choices must reach the validated preview facade."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        preview_module,
        "preview",
        lambda **values: calls.append(values),
    )

    assert main(("preview",)) == 0
    assert calls[-1]["host"] == "127.0.0.1"
    assert calls[-1]["allow_remote"] is False
    assert calls[-1]["field_mode"] is False

    assert (
        main(
            (
                "preview",
                "--host",
                "0.0.0.0",
                "--field-mode",
                "--token-file",
                "tokens.json",
                "--allowed-host",
                "camera.example",
                "--behind-tls-proxy",
            )
        )
        == 0
    )
    assert calls[-1]["field_mode"] is True
    assert calls[-1]["token_file"] == Path("tokens.json")
    assert calls[-1]["allowed_hosts"] == ("camera.example",)
    assert calls[-1]["behind_tls_proxy"] is True
