# IMX Camera Toolkit

![NVIDIA Jetson Orin](https://img.shields.io/badge/NVIDIA-Jetson%20Orin-76B900?logo=nvidia&logoColor=white)
[![JetPack 6.2.3](https://img.shields.io/badge/JetPack-6.2.3-76B900)](https://developer.nvidia.com/embedded/jetpack-sdk-623)
![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

IMX Camera Toolkit is a Python library for working with CSI-connected IMX cameras on NVIDIA Jetson. It uses Argus and GStreamer for capture, and can deliver frames to your own processing code or stream them to a browser.

There are two capture paths:

- **`Camera`** returns BGR images for OpenCV and other CPU-side code.
- **`GpuCamera`** exposes NV12/NVMM buffers for CUDA and TensorRT pipelines without converting frames to NumPy images.

Both use latest-frame delivery: if processing falls behind, old frames are skipped instead of piling up in a queue. The library also includes camera controls, diagnostics, and optional MJPEG, WebRTC, and HLS previews. It doesn't bundle inference models or tracking logic.

## Hardware and requirements

The current compatibility target is an **IMX219-77 on Jetson Orin Nano**, using **JetPack 6.2.3** at 30 FPS. The documented configurations include BGR/CPU capture with 1080p input and 720p output, and NV12/NVMM GPU capture at 720p or 1080p. GPU support for IMX477 is planned, not yet supported.

You'll need:

- A compatible Jetson and CSI camera, with NVIDIA Argus and `nvarguscamerasrc` available
- Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/)
- JetPack's system OpenCV (`python3-opencv`) with GStreamer support

**Keep the system OpenCV build.** Installing OpenCV from PyPI in its place can break Jetson camera capture. The virtual environment must have access to system packages.

See [camera support and sensor modes](imx_camera_toolkit/_internal/camera/README.md) for the full compatibility details and hardware validation procedure.

## Installation

The package isn't on PyPI yet. In your Jetson project, create an environment that can use the system OpenCV build, then install the pinned release:

```bash
uv venv --system-site-packages
uv add "imx-camera-toolkit @ git+https://github.com/bi3lu/imx-camera-toolkit.git@v0.8.0"
```

For the browser preview, install the `preview` extra instead:

```bash
uv add "imx-camera-toolkit[preview] @ git+https://github.com/bi3lu/imx-camera-toolkit.git@v0.8.0"
```

WebRTC and HLS use the separate `production-preview` extra. On Orin Nano, H.264 encoding uses the system GStreamer x264 plugin because NVENC isn't available; H.265 requires a Jetson with NVENC support. The [preview documentation](imx_camera_toolkit/_internal/production_preview/README.md) covers that setup.

## Reading frames

### CPU: OpenCV / NumPy

Use `Camera` when the next step needs a regular BGR image:

```python
from imx_camera_toolkit import Camera, CameraConfig

with Camera(CameraConfig(enable_preview=False)) as camera:
    frame = camera.read(timeout=1.0)

    if frame is not None:
        print(frame.image.shape)
```

`read()` gives you the latest available frame, or `None` if nothing new arrives before the timeout. Passing `copy=False` avoids an extra CPU copy; it does not make the image a GPU buffer.

### GPU: NV12 / NVMM

Use `GpuCamera` when you want to work with the captured buffer on the GPU:

```python
from imx_camera_toolkit import GpuCamera

with GpuCamera() as camera:
    frame = camera.read(timeout=1.0)

    if frame is not None:
        buffer = frame.payload()  # Use before the next read
```

These buffers are borrowed from the capture pipeline, so don't keep one after a newer frame arrives. For work on another thread, use `subscribe_latest()` and release each subscribed GPU frame after processing. Buffer ownership is explained in the [CPU/GPU guide](docs/GPU_PATH_GUIDE.md).

## Preview and diagnostics

With the `preview` extra installed, you can check the camera and start an MJPEG preview:

```bash
uv run imx-camera diagnose --hardware
uv run imx-camera test --backend gpu --frames 30 --timeout 5
uv run imx-camera preview --backend gpu --port 8000
```

Then open [http://localhost:8000/](http://localhost:8000/) on the Jetson. The development server binds to `127.0.0.1` by default.

The preview is intended for local use. For remote or production access, use field mode with authentication and TLS; see the [deployment guide](docs/GPU_CAMERA_YOLO_GUIDE.md). Don't expose the unauthenticated development server to the internet.

## How it works

A camera instance owns the Argus/GStreamer capture pipeline. CPU and GPU consumers work with the most recent frame rather than a growing queue, so a slow model or browser client won't block capture. Preview encoding runs separately from the raw-frame processing path.

The library handles capture, camera settings, and transport. Your application remains responsible for its models, preprocessing, results, and any higher-level vision logic.

## Documentation

- [Documentation index](docs/README.md) — architecture and component references
- [CPU, GPU, and browser modes](docs/GPU_PATH_GUIDE.md) — choosing an API, buffer lifetimes, and streaming options
- [Camera hardware](imx_camera_toolkit/_internal/camera/README.md) — supported profiles, sensor modes, and validation
- [Inference integration](imx_camera_toolkit/_internal/inference/README.md) — TensorRT runner and GPU interop
- [GPU camera + YOLO deployment](docs/GPU_CAMERA_YOLO_GUIDE.md) — setup, WebRTC, TLS, and systemd

Small usage examples are in [`examples/`](examples/).

## Development

To work on the library itself:

```bash
git clone https://github.com/bi3lu/imx-camera-toolkit.git
cd imx-camera-toolkit
uv venv --system-site-packages
uv sync --extra preview --group dev
```

Run formatting, static checks, and tests with:

```bash
uv run black --check .
uv run ruff check .
uv run mypy imx_camera_toolkit tests
uv run pytest tests/unit tests/integration -m "not hardware and not benchmark"
```

Hardware tests and benchmarks are separate from the regular test suite.

## License

[MIT](LICENSE) © 2026 Jakub Bielecki.
