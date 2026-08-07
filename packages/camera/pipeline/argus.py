"""GStreamer pipeline construction for NVIDIA Argus cameras."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..errors import CameraConfigurationError


def normalize_argus_properties(properties: Sequence[str]) -> tuple[str, ...]:
    """Validate properties that are safe to place in an Argus pipeline.

    Args:
        properties: ``nvarguscamerasrc`` property assignments.

    Returns:
        Normalized property assignments in their original order.

    Raises:
        ValueError: If an assignment is malformed or could alter pipeline
            structure.
    """
    if isinstance(properties, str):
        raise CameraConfigurationError(
            "argus_properties must be a sequence of assignments"
        )

    normalized: list[str] = []
    property_pattern = re.compile(
        r'[A-Za-z][A-Za-z0-9-]*=(?:[A-Za-z0-9_.-]+|"[A-Za-z0-9_. -]+")'
    )

    for property_value in properties:
        if not isinstance(property_value, str):
            raise CameraConfigurationError("each Argus property must be a string")

        if not property_pattern.fullmatch(property_value):
            raise CameraConfigurationError(
                f"invalid Argus property: {property_value!r}"
            )

        normalized.append(property_value)

    return tuple(normalized)


def build_gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    output_width: int = 640,
    output_height: int = 360,
    framerate: int = 30,
    flip_method: int = 0,
    argus_properties: Sequence[str] = (),
) -> str:
    """Build the compatible Argus pipeline with a BGR/CPU appsink.

    NV12 frames begin in NVMM, but ``nvvidconv`` and ``videoconvert`` produce
    BGR in system memory before ``appsink``. Capture backends then materialize
    an owned host array. This deliberate compatibility path is not GPU
    zero-copy.

    Args:
        sensor_id: Zero-based CSI sensor identifier used by Argus.
        capture_width: Width captured directly from the sensor, in pixels.
        capture_height: Height captured directly from the sensor, in pixels.
        output_width: Width of frames delivered to the backend, in pixels.
        output_height: Height of frames delivered to the backend, in pixels.
        framerate: Camera capture rate, in frames per second.
        flip_method: NVIDIA ``nvvidconv`` flip transformation, from 0 to 7.
        argus_properties: Validated ``nvarguscamerasrc`` properties to append.

    Returns:
        A GStreamer pipeline string suitable for the available backends.

    Raises:
        ValueError: If an identifier, dimension, frame rate, flip method, or
            source property is outside its supported range.
    """
    if sensor_id < 0:
        raise CameraConfigurationError(
            "sensor_id must be greater than or equal to zero"
        )

    if min(capture_width, capture_height, output_width, output_height, framerate) <= 0:
        raise CameraConfigurationError(
            "frame dimensions and framerate must be greater than zero"
        )

    if not 0 <= flip_method <= 7:
        raise CameraConfigurationError("flip_method must be between 0 and 7")

    source_properties = normalize_argus_properties(argus_properties)
    source_arguments = " ".join(source_properties)
    source_suffix = f" {source_arguments}" if source_arguments else ""
    return (
        f"nvarguscamerasrc name=argus_source sensor-id={sensor_id}{source_suffix} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, "
        f"height=(int){capture_height}, "
        "format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, "
        f"width=(int){output_width}, "
        f"height=(int){output_height}, "
        "format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink name=camera_sink max-buffers=1 drop=true sync=false"
    )


def build_gpu_gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1920,
    capture_height: int = 1080,
    output_width: int = 1280,
    output_height: int = 720,
    framerate: int = 30,
    flip_method: int = 0,
    *,
    enable_preview: bool = False,
    jpeg_quality: int = 65,
    argus_properties: Sequence[str] = (),
) -> str:
    """Build an NV12/NVMM pipeline with isolated inference and preview branches.

    The inference branch terminates at ``gpu_sink`` without leaving NVMM and
    without a CPU color conversion. When preview is enabled, a second leaky
    queue feeds NVIDIA's JPEG encoder and a separate encoded-byte appsink.
    Every queue and appsink retains at most one buffer so slow consumers do not
    increase end-to-end latency.
    """
    if sensor_id < 0:
        raise CameraConfigurationError(
            "sensor_id must be greater than or equal to zero"
        )

    if min(
        capture_width,
        capture_height,
        output_width,
        output_height,
        framerate,
    ) <= 0:
        raise CameraConfigurationError(
            "frame dimensions and framerate must be greater than zero"
        )

    if not 0 <= flip_method <= 7:
        raise CameraConfigurationError("flip_method must be between 0 and 7")

    if not isinstance(enable_preview, bool):
        raise CameraConfigurationError("enable_preview must be a boolean")

    if (
        isinstance(jpeg_quality, bool)
        or not isinstance(jpeg_quality, int)
        or not 0 <= jpeg_quality <= 100
    ):
        raise CameraConfigurationError("jpeg_quality must be between 0 and 100")

    source_properties = normalize_argus_properties(argus_properties)
    source_arguments = " ".join(source_properties)
    source_suffix = f" {source_arguments}" if source_arguments else ""
    nvmm_caps = (
        "video/x-raw(memory:NVMM), "
        f"width=(int){output_width}, "
        f"height=(int){output_height}, "
        "format=(string)NV12, "
        f"framerate=(fraction){framerate}/1"
    )
    queue_policy = (
        "max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
    )
    appsink_policy = (
        "max-buffers=1 drop=true sync=false enable-last-sample=false "
        "wait-on-eos=false"
    )
    pipeline = (
        f"nvarguscamerasrc name=argus_source sensor-id={sensor_id}{source_suffix} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, "
        f"height=(int){capture_height}, "
        "format=(string)NV12, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"{nvmm_caps} ! "
        "tee name=camera_tee "
        f"camera_tee. ! queue name=gpu_queue {queue_policy} ! "
        f"{nvmm_caps} ! "
        f"appsink name=gpu_sink {appsink_policy}"
    )

    if enable_preview:
        pipeline += (
            f" camera_tee. ! queue name=preview_queue {queue_policy} ! "
            f"{nvmm_caps} ! "
            f"nvjpegenc quality={jpeg_quality} ! "
            f"appsink name=preview_sink {appsink_policy}"
        )

    return pipeline
