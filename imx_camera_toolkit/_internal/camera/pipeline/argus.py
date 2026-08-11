"""GStreamer pipeline construction for NVIDIA Argus cameras."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..errors import CameraConfigurationError
from ..models.video import (
    EncodedStreamDescription,
    HardwareVideoConfig,
    VideoCodec,
    VideoEncoderBackend,
    VideoEncoderConfig,
    VideoEncoderPipeline,
    VideoEncoderPipelineFactory,
)


def build_video_encoder_pipeline(
    config: VideoEncoderConfig,
    width: int,
    height: int,
    framerate: int,
    *,
    backend: VideoEncoderBackend | None = None,
) -> VideoEncoderPipeline:
    """Build the standard NVENC or x264 production encoder segment.

    The input to the returned segment is NV12/NVMM.  NVENC consumes it
    directly.  x264 performs the sole device-to-host conversion in the whole
    camera graph and requests I420 system-memory video.
    """
    selected = backend or config.backend

    if selected is VideoEncoderBackend.AUTO:
        # Pipeline-only callers have no registry to inspect.  Runtime camera
        # startup resolves AUTO against Gst.ElementFactory before parsing.
        selected = VideoEncoderBackend.NVENC

    description = EncodedStreamDescription(
        codec=config.codec,
        width=width,
        height=height,
        fps=framerate,
    )
    parser = "h264parse" if config.codec is VideoCodec.H264 else "h265parse"
    media_type = "video/x-h264" if config.codec is VideoCodec.H264 else "video/x-h265"

    if selected is VideoEncoderBackend.X264:
        if config.codec is not VideoCodec.H264:
            raise CameraConfigurationError("x264 backend supports only H.264")

        pipeline = (
            "nvvidconv ! "
            "video/x-raw, "
            f"width=(int){width}, height=(int){height}, "
            "format=(string)I420, "
            f"framerate=(fraction){framerate}/1 ! "
            "x264enc name=video_encoder tune=zerolatency "
            f"speed-preset={config.software_preset} "
            f"bitrate={max(config.bitrate_bps // 1000, 1)} "
            f"key-int-max={config.keyframe_interval} byte-stream=true aud=true ! "
            f"{parser} config-interval=-1 ! "
            f"{media_type}, stream-format=(string)byte-stream, "
            "alignment=(string)au ! "
            "appsink name=video_sink max-buffers=1 drop=true sync=false "
            "enable-last-sample=false wait-on-eos=false"
        )
        return VideoEncoderPipeline(
            pipeline=pipeline,
            backend=selected.value,
            description=description,
            required_elements=("nvvidconv", "x264enc", parser, "appsink"),
        )

    if selected is not VideoEncoderBackend.NVENC:
        raise CameraConfigurationError(f"unsupported encoder backend: {selected}")

    encoder = "nvv4l2h264enc" if config.codec is VideoCodec.H264 else "nvv4l2h265enc"
    pipeline = (
        f"{encoder} name=video_encoder control-rate=1 "
        f"bitrate={config.bitrate_bps} "
        f"iframeinterval={config.keyframe_interval} "
        f"idrinterval={config.keyframe_interval} "
        "insert-sps-pps=1 insert-vui=1 "
        f"maxperf-enable={int(config.max_performance)} "
        f"MeasureEncoderLatency={int(config.measure_latency)} ! "
        f"{parser} config-interval=-1 ! "
        f"{media_type}, stream-format=(string)byte-stream, "
        "alignment=(string)au ! "
        "appsink name=video_sink max-buffers=1 drop=true sync=false "
        "enable-last-sample=false wait-on-eos=false"
    )
    return VideoEncoderPipeline(
        pipeline=pipeline,
        backend=selected.value,
        description=description,
        required_elements=(encoder, parser, "appsink"),
    )


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
    video_config: HardwareVideoConfig | None = None,
    enable_video_overlay: bool = False,
    encoder_backend: VideoEncoderBackend | None = None,
    encoder_pipeline_factory: VideoEncoderPipelineFactory | None = None,
    argus_properties: Sequence[str] = (),
) -> str:
    """Build an NV12/NVMM pipeline with isolated inference and preview branches.

    The inference branch terminates at ``gpu_sink`` without leaving NVMM and
    without a CPU color conversion. When preview is enabled, a second leaky
    queue feeds NVIDIA's JPEG encoder and a separate encoded-byte appsink. An
    optional production branch keeps NV12 in NVMM through a Jetson V4L2 H.264
    or H.265 encoder. Every queue and appsink is bounded so slow consumers do
    not increase end-to-end latency.
    """
    if sensor_id < 0:
        raise CameraConfigurationError(
            "sensor_id must be greater than or equal to zero"
        )

    if (
        min(
            capture_width,
            capture_height,
            output_width,
            output_height,
            framerate,
        )
        <= 0
    ):
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

    if video_config is not None and not isinstance(video_config, HardwareVideoConfig):
        raise CameraConfigurationError(
            "video_config must be a HardwareVideoConfig or None"
        )

    if not isinstance(enable_video_overlay, bool):
        raise CameraConfigurationError("enable_video_overlay must be a boolean")

    if enable_video_overlay and video_config is None:
        raise CameraConfigurationError(
            "video overlay requires a video encoder configuration"
        )

    if encoder_backend is not None and not isinstance(
        encoder_backend, VideoEncoderBackend
    ):
        raise CameraConfigurationError(
            "encoder_backend must be a VideoEncoderBackend or None"
        )

    if encoder_pipeline_factory is not None and not callable(encoder_pipeline_factory):
        raise CameraConfigurationError("encoder_pipeline_factory must be callable")

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

    if video_config is not None:
        overlay = ""

        if enable_video_overlay:
            overlay = (
                "nvvidconv ! " f"{nvmm_caps} ! " "identity name=video_overlay_hook ! "
            )

        encoder_pipeline = (
            build_video_encoder_pipeline(
                video_config,
                output_width,
                output_height,
                framerate,
                backend=encoder_backend,
            )
            if encoder_pipeline_factory is None
            else encoder_pipeline_factory(
                video_config,
                output_width,
                output_height,
                framerate,
            )
        )

        if not isinstance(encoder_pipeline, VideoEncoderPipeline):
            raise CameraConfigurationError(
                "encoder_pipeline_factory must return VideoEncoderPipeline"
            )

        pipeline += (
            f" camera_tee. ! queue name=video_queue {queue_policy} ! "
            f"{nvmm_caps} ! "
            f"{overlay}"
            f"{encoder_pipeline.pipeline}"
        )

    return pipeline
