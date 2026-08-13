"""Run an application-owned end-to-end YOLO decoder with WebRTC preview."""

from __future__ import annotations

import argparse
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import uvicorn

from imx_camera_toolkit import (
    CameraConfig,
    FrameFormat,
    FrameSpec,
    GpuCamera,
    MemoryType,
    ShapeProfile,
    TensorRTRunner,
    VideoEncoderConfig,
)
from imx_camera_toolkit.api import SecurityConfig
from imx_camera_toolkit.consumers import InferenceConsumer
from imx_camera_toolkit.inference import InferenceResult
from imx_camera_toolkit.production_preview import (
    CudaOverlayRenderer,
    OverlayRectangle,
    ProductionPreviewConfig,
    ProductionPreviewServer,
    create_production_preview_app,
)


def _is_loopback(host: str) -> bool:
    """Return whether a server bind address is local to the Jetson."""
    normalized = host.strip().strip("[]").lower()

    if normalized == "localhost":
        return True

    try:
        return ip_address(normalized).is_loopback

    except ValueError:
        return False


def _create_security_config(arguments: argparse.Namespace) -> SecurityConfig:
    """Create a fail-closed HTTP policy from parsed example arguments.

    Args:
        arguments: Parsed YOLO example command-line arguments.

    Returns:
        Validated development or field-mode security policy.

    Raises:
        ValueError: If a remote bind or field-mode option is unsafe or
            incomplete.
    """
    remote = not _is_loopback(arguments.host)
    direct_tls = arguments.tls_certfile is not None

    if (arguments.tls_certfile is None) != (arguments.tls_keyfile is None):
        raise ValueError("--tls-certfile and --tls-keyfile must be used together")

    if remote and not arguments.field_mode:
        raise ValueError("a non-loopback --host requires --field-mode")

    if not arguments.field_mode:
        if (
            arguments.token_file is not None
            or arguments.allowed_host
            or arguments.behind_tls_proxy
        ):
            raise ValueError(
                "token, allowed-host, and proxy options require --field-mode"
            )
        return SecurityConfig()

    if arguments.token_file is None:
        raise ValueError("--field-mode requires --token-file")

    if "*" in arguments.allowed_host:
        raise ValueError("--field-mode does not permit a wildcard allowed host")

    if remote and not arguments.allowed_host:
        raise ValueError("a remote field deployment requires --allowed-host")

    if arguments.behind_tls_proxy and remote:
        raise ValueError("--behind-tls-proxy requires a loopback --host")

    if remote and not direct_tls:
        raise ValueError("a remote field deployment requires direct TLS")

    allowed_hosts = tuple(arguments.allowed_host) or (
        "localhost",
        "127.0.0.1",
        "[::1]",
    )
    return SecurityConfig.from_token_file(
        arguments.token_file,
        field_mode=True,
        allowed_hosts=allowed_hosts,
        require_https=direct_tls or arguments.behind_tls_proxy,
    )


def _rows(result: InferenceResult, output_name: str) -> list[list[float]]:
    """Decode one common end-to-end YOLO Nx6 export outside the toolkit.

    Args:
        result: Model-neutral TensorRT inference result.
        output_name: Name of the output tensor containing detection rows.

    Returns:
        Detection rows converted to Python floats.

    Raises:
        StopIteration: If ``output_name`` is absent from the model outputs.
        TypeError: If the output does not support NumPy-compatible conversion.
    """
    output = next(item for item in result.outputs if item.name == output_name)
    tolist = getattr(output.data, "tolist", None)

    if not callable(tolist):
        raise TypeError("YOLO output must provide ndarray-compatible tolist()")

    values: Any = tolist()

    if (
        values
        and isinstance(values[0], list)
        and values[0]
        and isinstance(values[0][0], list)
    ):
        values = values[0]

    return [[float(value) for value in row] for row in values]


def main() -> None:
    """Serve H.264 WebRTC with rectangles decoded from a custom YOLO export."""
    parser = argparse.ArgumentParser(
        description=(
            "Run GPU-first YOLO inference and a shared H.264 WebRTC preview. "
            "The ONNX model must expose end-to-end Nx6 detections."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", type=Path, help="path to the YOLO ONNX model")
    parser.add_argument(
        "--output",
        default="boxes",
        help="name of the Nx6 detection output tensor",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=0.5,
        help="minimum confidence rendered in the preview",
    )
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--sensor-mode", type=int)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stun-server", help="WebRTC STUN URI")
    parser.add_argument("--turn-server", help="WebRTC TURN URI")
    parser.add_argument(
        "--field-mode",
        action="store_true",
        help="enable authentication, host checks, rate limits, and hardening",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        help="0600/0640 JSON file containing hashed bearer-token grants",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="accepted HTTP Host value; repeat for multiple names",
    )
    parser.add_argument(
        "--behind-tls-proxy",
        action="store_true",
        help="require HTTPS forwarded by a trusted loopback reverse proxy",
    )
    parser.add_argument("--tls-certfile", type=Path)
    parser.add_argument("--tls-keyfile", type=Path)
    args = parser.parse_args()

    try:
        security = _create_security_config(args)

    except (OSError, ValueError) as error:
        parser.error(str(error))

    if not 0.0 <= args.score <= 1.0:
        parser.error("--score must be between 0 and 1")

    camera_config = CameraConfig(
        sensor_id=args.sensor_id,
        sensor_mode=args.sensor_mode,
        capture_width=args.width,
        capture_height=args.height,
        output_width=args.width,
        output_height=args.height,
        fps=args.fps,
    )
    runner = TensorRTRunner(
        args.model,
        cache_dir=Path(".cache/tensorrt"),
        shape_profile=ShapeProfile(
            minimum=(1, 3, 320, 320),
            optimum=(1, 3, 640, 640),
            maximum=(1, 3, 1280, 1280),
        ),
        inference_shape=(1, 3, 640, 640),
        resize_mode="letterbox",
    )
    camera = GpuCamera(
        camera_config,
        video_config=VideoEncoderConfig(),
    )
    runner.prepare(
        FrameSpec(
            width=camera_config.output_width,
            height=camera_config.output_height,
            format=FrameFormat.NV12_NVMM,
            memory_type=MemoryType.NVMM,
        )
    )
    inference = InferenceConsumer(
        camera.subscribe_latest("yolo"),
        runner,
    )

    def rectangles(result: InferenceResult) -> tuple[OverlayRectangle, ...]:
        """Map the application's chosen YOLO schema to toolkit overlays."""
        overlays: list[OverlayRectangle] = []
        transform = runner.resize_transform

        if transform is None:
            return ()

        scale_x, scale_y = transform.scale

        for row in _rows(result, args.output):
            if len(row) < 6 or row[4] < args.score:
                continue

            left = max(round((row[0] - transform.pad_x) / scale_x), 0)
            top = max(round((row[1] - transform.pad_y) / scale_y), 0)
            right = min(
                round((row[2] - transform.pad_x) / scale_x),
                camera_config.output_width,
            )
            bottom = min(
                round((row[3] - transform.pad_y) / scale_y),
                camera_config.output_height,
            )

            if right > left and bottom > top:
                overlays.append(OverlayRectangle(left, top, right - left, bottom - top))
        return tuple(overlays)

    overlay = CudaOverlayRenderer(inference, mapper=rectangles)
    camera.set_video_overlay(overlay)
    transport = ProductionPreviewServer(
        camera,
        ProductionPreviewConfig(
            stun_server=args.stun_server,
            turn_server=args.turn_server,
        ),
        health_providers={
            "inference": inference.health,
            "overlay": overlay.health,
        },
    )
    app = create_production_preview_app(transport, security_config=security)

    try:
        with camera, inference:
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                ssl_certfile=args.tls_certfile,
                ssl_keyfile=args.tls_keyfile,
            )

    finally:
        overlay.close()


if __name__ == "__main__":
    main()
