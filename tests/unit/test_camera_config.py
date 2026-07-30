"""Tests for the public immutable camera configuration model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict

import pytest

from imx_camera_toolkit import Camera, CameraConfig


def test_camera_config_has_documented_defaults() -> None:
    """The zero-argument model must support the documented integration API."""
    assert CameraConfig() == CameraConfig(
        sensor_id=0,
        capture_width=1920,
        capture_height=1080,
        output_width=1280,
        output_height=720,
        fps=30,
        flip_method=0,
        sensor_mode=None,
        enable_preview=False,
    )


def test_camera_uses_explicit_camera_config() -> None:
    """Camera(config) must build the pipeline from every static field."""
    config = CameraConfig(
        sensor_id=1,
        capture_width=1920,
        capture_height=1080,
        output_width=960,
        output_height=540,
        fps=40,
        flip_method=2,
        sensor_mode=3,
    )

    camera = Camera(config)

    assert camera.config == config
    assert "sensor-id=1" in camera.pipeline
    assert "sensor-mode=3" in camera.pipeline
    assert "width=(int)1920" in camera.pipeline
    assert "height=(int)1080" in camera.pipeline
    assert "framerate=(fraction)40/1" in camera.pipeline


def test_camera_config_is_frozen_and_serializable() -> None:
    """The public model must be safe to share, compare, and serialize."""
    config = CameraConfig()

    with pytest.raises(FrozenInstanceError):
        config.fps = 40  # type: ignore[misc]

    assert asdict(config)["fps"] == 30


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fps": 0}, "framerate"),
        ({"sensor_mode": -1}, "sensor_mode"),
        ({"enable_preview": 1}, "enable_preview"),
        ({"max_fps": 0}, "max_fps"),
    ],
)
def test_camera_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Invalid static settings must fail before a camera is opened."""
    with pytest.raises(ValueError, match=message):
        CameraConfig(**kwargs)  # type: ignore[arg-type]


def test_legacy_camera_arguments_override_a_camera_config() -> None:
    """Existing named constructor arguments remain supported during migration."""
    camera = Camera(CameraConfig(fps=30), capture_fps=40, enable_preview=True)

    assert camera.config.fps == 40
    assert camera.preview_enabled is True


def test_legacy_positional_quality_and_preview_rate_remain_supported() -> None:
    """The new first positional config argument must not break prior callers."""
    camera = Camera(80, 20.0)

    assert camera.config.quality == 80
    assert camera.config.max_fps == 20.0
