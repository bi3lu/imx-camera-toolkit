"""Tests for the public camera-preview facade."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from imx_camera_toolkit import CameraPreview, preview

preview_module = importlib.import_module("imx_camera_toolkit.preview")


def test_camera_preview_composes_camera_api_and_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preview facade must configure existing components without hardware."""
    captured: dict[str, Any] = {}
    application = object()

    class FakeCamera:
        """Record camera construction without opening a real backend."""

        def __init__(self, **kwargs: object) -> None:
            """Store facade-provided camera configuration."""
            captured["camera"] = kwargs

    def fake_create_app(camera: object, *, view_mode: str) -> object:
        """Record API composition and return an opaque application."""
        captured["api"] = {"camera": camera, "view_mode": view_mode}
        return application

    def fake_run(server_app: object, *, host: str, port: int) -> None:
        """Record server launch instead of blocking on Uvicorn."""
        captured["server"] = {"app": server_app, "host": host, "port": port}

    monkeypatch.setattr(preview_module, "Camera", FakeCamera)
    monkeypatch.setattr(preview_module, "create_app", fake_create_app)
    monkeypatch.setattr(preview_module.uvicorn, "run", fake_run)

    preview(sensor_id=1, width=1920, height=1080, fps=30, port=9000)

    config = captured["camera"]["config"]
    assert config.sensor_id == 1
    assert config.capture_width == 1920
    assert config.capture_height == 1080
    assert config.output_width == 1920
    assert config.output_height == 1080
    assert config.fps == 30
    assert config.max_fps == 30.0
    assert config.enable_preview is True
    assert captured["api"]["view_mode"] == "simple"
    assert isinstance(captured["api"]["camera"], FakeCamera)
    assert captured["server"] == {
        "app": application,
        "host": "0.0.0.0",
        "port": 9000,
    }


def test_camera_preview_uses_documented_defaults() -> None:
    """Default facade configuration must match the public API contract."""
    assert CameraPreview() == CameraPreview(
        sensor_id=0,
        width=1280,
        height=720,
        fps=30,
        host="0.0.0.0",
        port=8000,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sensor_id": -1}, "sensor_id"),
        ({"width": 0}, "width"),
        ({"height": 0}, "height"),
        ({"fps": 0}, "fps"),
        ({"host": ""}, "host"),
        ({"port": 0}, "port"),
        ({"port": 65536}, "port"),
    ],
)
def test_camera_preview_rejects_invalid_configuration(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    """Invalid facade configuration must fail before a camera is constructed."""
    with pytest.raises(ValueError, match=message):
        CameraPreview(**kwargs)
