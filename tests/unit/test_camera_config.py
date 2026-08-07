"""Tests for the public immutable camera configuration model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict

import pytest

from imx_camera_toolkit import (
    Camera,
    CameraConfig,
    CameraProfileStatus,
    FrameFormat,
    MemoryType,
    get_camera_profile,
    list_camera_profiles,
)
from packages.camera.config.loader import _read_config_values


def test_camera_config_has_documented_defaults() -> None:
    """The zero-argument model must support the documented integration API."""
    assert CameraConfig() == CameraConfig(
        sensor_id=0,
        capture_width=1920,
        capture_height=1080,
        output_width=1280,
        output_height=720,
        output_format=FrameFormat.BGR_CPU,
        fps=30,
        flip_method=0,
        sensor_mode=None,
        enable_preview=False,
    )
    assert CameraConfig().output_memory is MemoryType.CPU
    assert CameraConfig().copies_to_host_memory is True


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


def test_default_camera_keeps_the_compatible_bgr_cpu_pipeline() -> None:
    """Existing users must continue receiving one latest BGR host array."""
    camera = Camera()

    assert camera.config.output_format is FrameFormat.BGR_CPU
    assert "video/x-raw(memory:NVMM)" in camera.pipeline
    assert "format=(string)NV12" in camera.pipeline
    assert "format=(string)BGRx" in camera.pipeline
    assert "video/x-raw, format=(string)BGR" in camera.pipeline
    assert "appsink name=camera_sink max-buffers=1 drop=true" in camera.pipeline


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
        ({"output_format": FrameFormat.NV12_NVMM}, "BGR_CPU"),
        ({"output_format": "BGR_CPU"}, "FrameFormat"),
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


def test_camera_config_preserves_legacy_positional_field_order() -> None:
    """Appending output_format must not shift existing positional settings."""
    config = CameraConfig(1, 1920, 1080, 640, 360, 25, 2, None, False, 70, 15.0)

    assert config.sensor_id == 1
    assert config.output_height == 360
    assert config.fps == 25
    assert config.flip_method == 2
    assert config.quality == 70
    assert config.max_fps == 15.0
    assert config.output_format is FrameFormat.BGR_CPU


def test_yaml_values_resolve_the_explicit_bgr_cpu_output() -> None:
    """String-based configuration must resolve to the public format enum."""
    config = _read_config_values({"output_format": "BGR_CPU"})

    assert config.output_format is FrameFormat.BGR_CPU
    assert config.output_memory is MemoryType.CPU


def test_imx219_77_profile_is_explicitly_tested() -> None:
    """The curated IMX219-77 profile must expose tested hardware settings."""
    profile = get_camera_profile("imx219-1080p")
    config = CameraConfig.from_profile("imx219-1080p")

    assert profile.sensor == "IMX219-77"
    assert profile.status is CameraProfileStatus.TESTED
    assert config == profile.config
    assert config.sensor_id == 0
    assert config.sensor_mode == 2
    assert config.capture_width == 1920
    assert config.capture_height == 1080
    assert config.fps == 30
    assert config.output_width == 1280
    assert config.output_height == 720


def test_camera_profile_alias_and_hardware_mapping() -> None:
    """Aliases and portable mappings must resolve to the same hardware mode."""
    profile = get_camera_profile("imx219-77-1080p")

    assert profile.name == "imx219-1080p"
    assert profile.hardware_settings() == {
        "sensor_id": 0,
        "sensor_mode": 2,
        "capture": {"width": 1920, "height": 1080, "fps": 30},
        "output": {"width": 1280, "height": 720},
    }


def test_camera_profile_catalog_contains_only_documented_profiles() -> None:
    """The stable profile list must not imply unverified sensor support."""
    assert list_camera_profiles() == (get_camera_profile("imx219-1080p"),)


@pytest.mark.parametrize(
    ("name", "exception"),
    [
        ("imx477-720p60", ValueError),
        (1, TypeError),
    ],
)
def test_camera_profile_lookup_rejects_unavailable_or_invalid_names(
    name: object,
    exception: type[Exception],
) -> None:
    """Profiles must not make unsupported hardware appear available."""
    with pytest.raises(exception):
        CameraConfig.from_profile(name)  # type: ignore[arg-type]
