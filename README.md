# IMX Camera Toolkit

## Abstract

IMX Camera Toolkit is a Python toolkit for acquiring, encoding, and serving
live video from CSI-connected IMX image sensors on NVIDIA Jetson systems. It is
designed for NVIDIA Jetson Orin Nano deployments using the JetPack software
stack, NVIDIA Argus, GStreamer, and the system-provided OpenCV build.

The project separates sensor acquisition, image processing, runtime camera
controls, MJPEG framing, and HTTP delivery into independent packages. This
separation keeps hardware-specific code isolated and allows alternative
processing or transport layers to be introduced without coupling them to the
camera capture loop.

## What this project is not

IMX Camera Toolkit is a camera-capture and image-transport foundation. It does
not provide inference models, model loaders, trackers, batching, CUDA-stream
management, DeepStream pipelines, ROS 2 integration, multi-camera
synchronization, or telemetry backends. Applications retain ownership of those
policies and receive a BGR/CPU `Frame.image` payload for their own chosen vision
stack. Optional GPU sources use the separate borrowed `GpuFrame` contract.

Start with the [CPU/GPU and browser mode guide](docs/GPU_PATH_GUIDE.md) before
selecting `Camera`, experimental `GpuCamera`, MJPEG, WebRTC, or HLS.

## System architecture

```text
CSI IMX sensor
      |
      v
Camera package <--- runtime controls --- Camera Control package
Argus + GStreamer capture, BGR conversion, software HDR, JPEG encoding
      |
      +----> Frames package
      |      Minimal raw-frame source contract for external pipelines
      v
Stream package
MJPEG multipart framing
      |
      v
API package
FastAPI snapshots, health data, browser view, and MJPEG endpoint
      |
      v
Browser or other HTTP client
```

The camera package maintains a single, shared capture instance. It retains only
the newest encoded JPEG frame. Consequently, slow consumers do not create an
unbounded frame queue; they receive the newest available image and may skip
intermediate frames.

## Components

| Package | Responsibility |
| --- | --- |
| [`packages/camera`](packages/camera/README.md) | CSI camera acquisition through `nvarguscamerasrc`, frame conversion, JPEG encoding, and frame synchronization. |
| [`packages/camera_control`](packages/camera_control/README.md) | Validated runtime exposure, gain, white-balance, denoise, sensor-mode, and HDR control for NVIDIA Argus cameras. |
| [`packages/frames`](packages/frames/README.md) | Minimal `FrameSource` protocol and adapter from the toolkit camera to external processing pipelines. |
| [`packages/inference`](packages/inference/README.md) | Model-neutral inference contracts, validated TensorRT engine caching, and optional NvBufSurface/CUDA interop. |
| [`packages/consumers`](packages/consumers/README.md) | Independent latest-frame slots, worker consumers, and inference-preview adaptation. |
| [`packages/production_preview`](packages/production_preview/README.md) | Optional shared H.264/H.265 encoding through NVENC or, for H.264, x264, with WebRTC, HLS, GPU overlays, and client metrics. |
| [`packages/stream`](packages/stream/README.md) | Framework-neutral construction of `multipart/x-mixed-replace` MJPEG body parts. |
| [`packages/api`](packages/api/README.md) | FastAPI application, camera lifecycle management, snapshots, health reporting, MJPEG delivery, and browser view rendering. |
| [`packages/testing`](packages/testing/mock_camera.py) | Deterministic, thread-safe camera substitute for tests and benchmarks without Jetson hardware. |
| [`view/advanced.html`](view/advanced.html) | Browser preview with runtime camera controls. |
| [`view/simple.html`](view/simple.html) | Browser preview without a camera-control panel. |

## Platform requirements

- NVIDIA Jetson Orin Nano or a compatible Jetson platform.
- A supported CSI-connected IMX camera sensor.
- JetPack 6.2.2 with NVIDIA Argus, `nvarguscamerasrc`, and GStreamer support.
- System OpenCV with GStreamer support. On JetPack this is normally supplied by
  the `python3-opencv` system package.
- Python 3.10–3.12 (JetPack 6.2.2 provides Python 3.10).
- [uv](https://docs.astral.sh/uv/) for dependency and virtual-environment
  management.

OpenCV is intentionally not installed from PyPI. The JetPack-provided build is
required because it integrates with NVIDIA's camera and GStreamer stack.

## Compatibility matrix

| JetPack | Jetson | Camera module | API / Argus mode | Capture and output | Status |
| --- | --- | --- | --- | --- | --- |
| 6.2.2 | Orin Nano | IMX219-77 | `Camera`, `imx219-1080p` / 2 | 1920×1080 → 1280×720 at 30 FPS, BGR/CPU | tested |
| 6.2.2 | Orin Nano | IMX219-77 | `GpuCamera` / 4 | 1280×720 at 30 FPS, NV12/NVMM + JPEG | tested |
| 6.2.2 | Orin Nano | IMX219-77 | `GpuCamera` / 2 | 1920×1080 at 30 FPS, NV12/NVMM + JPEG | tested |
| 6.2.2 | Orin Nano | IMX219-77 | `GpuCamera` + WebRTC / 4 | 1280×720 at 30 FPS, H.264/x264 | tested |
| 6.2.2 | Orin Nano | IMX477 | `GpuCamera` | 1280×720 and 1920×1080 at 30 FPS | planned |

Only rows marked `tested` have been verified on the stated hardware. “Planned”
is not a support claim and must not be treated as a working configuration.

## Installation

Install the core package when the application only needs camera capture and
raw-frame integration:

```bash
uv add imx-camera-toolkit
```

The core package has no PyPI runtime dependencies. JetPack supplies the system
OpenCV build with GStreamer support required for camera capture.

Install the optional browser preview stack when FastAPI and Uvicorn are needed:

```bash
uv add "imx-camera-toolkit[preview]"
```

Install the production WebRTC/HLS HTTP layer separately; JetPack continues to
provide GStreamer and CUDA. Orin Nano additionally needs the system x264
GStreamer plugin because that SoC does not expose NVENC:

```bash
uv add "imx-camera-toolkit[production-preview]"
```

Clone the repository and create a virtual environment that can access the
system OpenCV installation:

```bash
uv venv --system-site-packages
uv sync
```

For local browser preview development, include the preview extra:

```bash
uv sync --extra preview
```

If a project virtual environment already exists without system package access,
reconfigure it before synchronizing dependencies:

```bash
uv venv --system-site-packages --allow-existing .venv
uv sync
```

Verify that the project environment can see JetPack OpenCV:

```bash
uv run python -c "import cv2; print(cv2.__version__)"
```

## Using the toolkit as a Git dependency

Pin the v0.6.0 release tag in the consuming project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "imx-camera-toolkit @ git+https://github.com/bi3lu/imx-camera-toolkit.git@v0.6.0"
]
```

Then synchronize the environment with `uv`:

```bash
uv sync
```

On Jetson, create the consuming project's environment with
`--system-site-packages` before synchronizing so the JetPack-provided OpenCV
build remains available. For reproducible production deployments, pin a
release tag or commit rather than a moving branch.

To consume the Git dependency with the browser-preview extra, declare it as:

```toml
[project]
dependencies = [
    "imx-camera-toolkit[preview] @ git+https://github.com/bi3lu/imx-camera-toolkit.git@v0.6.0"
]
```

## Public Python namespace

External applications should import from `imx_camera_toolkit`. The repository
internal `packages` namespace remains an implementation detail and should not
be used by new projects.

```python
from imx_camera_toolkit import Camera, CameraConfig
from imx_camera_toolkit.frames import CameraFrameSource

with Camera(CameraConfig(enable_preview=False)) as camera:
    source = CameraFrameSource(camera)
    frame = source.read(timeout=1.0)
```

The public namespace also provides `api`, `camera_control`, `controls`,
`stream`, and deterministic test doubles:

```python
from imx_camera_toolkit.camera_control import CameraController
from imx_camera_toolkit.controls import CameraControls, ExposureConfig
from imx_camera_toolkit.stream import MJPEGStream
from imx_camera_toolkit.testing import MockCamera
```

For direct integration with an external AI or image-processing pipeline, use
the stable raw-frame API:

```python
from imx_camera_toolkit import Camera, CameraConfig

with Camera(CameraConfig()) as camera:
    frame = camera.read(timeout=1.0, copy=False)

    if frame is not None:
        result = my_tensor_rt_engine(frame.image)
```

`read()` returns a formal CPU `Frame` with a BGR `image`, monotonic `sequence`,
nanosecond `timestamp_ns`, optional hardware `capture_timestamp_ns`, dimensions,
and pixel `format`. It retains only the newest BGR frame, may skip stale frames,
and never performs JPEG encoding or inference. The default `copy=True` gives
the caller an independent image buffer; `copy=False` returns a read-only shared
payload for copy-avoiding CPU pipelines.

`copy=False` avoids an additional CPU-buffer copy only. It does not turn the
BGR payload into CUDA or NVMM memory and is not a GPU zero-copy guarantee.

### CPU and GPU frame contracts

The existing `Camera` and `Frame` contracts remain the OpenCV-compatible CPU
path. Their explicit output identity is `FrameFormat.BGR_CPU` in
`MemoryType.CPU`, while the legacy `frame.format` field remains `"BGR"`.

Optional GPU-first capture sources expose `GpuFrame` with
`FrameFormat.NV12_NVMM` and `MemoryType.NVMM`. A GPU frame has no `image` or
implicit NumPy conversion. It carries exactly one borrowed DMA-BUF descriptor
or a checked `GpuBufferHandle` around `Gst.Buffer` or `NvBufSurface`.

GPU buffers remain owned by capture. Values returned directly by `read()` are
borrowed and publishing a successor invalidates the previous direct-read
lease. Each `subscribe_latest()` subscriber instead receives an independent,
reference-counted lease, so a worker may finish its current frame while newer
frames arrive. `FrameConsumer` and `InferenceConsumer` release those leases
automatically; direct subscription users must call `frame.release()` when
finished. Access after release raises `GpuFrameExpiredError`.

### GPU-first capture

Use `GpuCamera` to opt into the Jetson NVMM path without changing the compatible
`Camera` API:

```python
from imx_camera_toolkit import CameraConfig, GpuCamera

with GpuCamera(
    CameraConfig(
        capture_width=1920,
        capture_height=1080,
        output_width=1920,
        output_height=1080,
        fps=30,
        enable_preview=True,
    ),
    experimental=True,
) as camera:
    frame = camera.read(timeout=1.0)

    if frame is not None:
        result = my_tensor_rt_consumer(frame.payload())
```

The inference branch stays `NV12/NVMM` through a one-buffer leaky queue and
`gpu_sink`. A separate one-buffer leaky branch feeds `nvjpegenc` and the MJPEG
preview source. Neither branch can accumulate stale frames. The GPU backend
checks the negotiated NVMM/NV12 caps and forwards the borrowed `Gst.Buffer`
without `buffer.map()`, NumPy conversion, or a host-memory image copy.

`GpuCamera` remains model-agnostic: applications own TensorRT engines,
preprocessing, CUDA synchronization, and model/GPU/TensorRT-aware engine cache
validation. See [the camera documentation](packages/camera/README.md#gpu-first-nvmm-capture)
for the pipeline contract and opt-in IMX219/IMX477 hardware validation commands.

### Optional TensorRT runner

The `tensorrt` extra adds a reference `TensorRTRunner` without making any model
framework a core dependency. On JetPack 6.2.2 it uses a small pybind11/CUDA
extension to import `NvBufSurface` through EGLImage, preprocess NV12 directly
into a TensorRT device binding, and execute on one runner-owned CUDA stream.
Camera pixels never become a BGR/NumPy host image and are never uploaded from
RAM.

```bash
uv sync --extra tensorrt
sudo apt-get install python-gi-dev libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev cmake ninja-build
uv run imx-camera-build-interop
```

The runner supports dynamic min/opt/max input profiles and caches local engine
bytes only beside matching metadata for the ONNX hash, TensorRT version,
compute capability, precision, input name, and complete shape profile.
Model-specific decoding and overlays remain application-owned. See
[GPU inference integration](packages/inference/README.md) for usage and the
TensorRT/ONNX Runtime parity test.

### Production preview

MJPEG/OpenCV remains the intentionally simple debug transport. For deployed
Jetson applications, pass `VideoEncoderConfig` to `GpuCamera`. Backend `AUTO`
uses NVENC where present and falls back to CPU x264 on Orin Nano while keeping
capture, inference, and overlay in NVMM. `HardwareVideoConfig` is retained as
a compatibility alias. The built-in x264 backend supports H.264 only, so H.265
requires NVENC and is not available on Orin Nano through the built-in backends.

WebRTC is the preferred low-latency browser mode; HLS provides a rolling,
reverse-proxy-friendly alternative.

The production transport shares one encoder among all clients and exposes
encode FPS, actual bitrate, active clients, and per-client drop rates. Optional
`CudaOverlayRenderer` draws normalized rectangles on an isolated NVMM surface;
the existing `InferencePreviewSource` remains the CPU/JPEG fallback. See
[production browser preview](packages/production_preview/README.md) for
installation, signaling, HLS storage, overlay wiring, and the 720p/30 +
TensorRT acceptance benchmark.

Model-agnostic code can depend only on the public union:

```python
from imx_camera_toolkit import FrameFormat, GpuFrame
from imx_camera_toolkit.frames import CaptureFrame


def consume(frame: CaptureFrame) -> object:
    if isinstance(frame, GpuFrame):
        assert frame.format is FrameFormat.NV12_NVMM
        return gpu_engine(frame.payload())

    assert frame.output_format is FrameFormat.BGR_CPU
    return cpu_engine(frame.image)
```

`PipelineStage` and `Camera.record_stage_latency()` provide fixed-size timing
aggregates for transfer, inference, encoder, and end-to-end latency. External
consumers can report their inference timing and skipped latest frames without
adding inference code to the toolkit:

```python
camera.record_stage_latency("inference", inference_duration_ns)
camera.record_consumer_drop("primary-inference", skipped_frames)
```

`/api/health` exposes those aggregates together with the active backend,
explicit frame format and memory domain, output resolution, capture timestamp,
and per-consumer drop counters.

### Asynchronous latest-frame consumers

Both camera variants expose `subscribe_latest(name)`. Every subscription owns
one replaceable slot, so capture never invokes consumer code or waits for a
slow model. `FrameConsumer` executes arbitrary CPU work on its own thread;
`InferenceConsumer` prepares and runs a model-neutral `InferenceRunner` on a
dedicated thread. Give each expensive inference consumer its own runner;
`TensorRTRunner` then gives each one an independent CUDA stream.

```python
from imx_camera_toolkit import GpuCamera
from imx_camera_toolkit.consumers import InferenceConsumer

with GpuCamera(experimental=True) as camera:
    inference = InferenceConsumer(
        camera.subscribe_latest("primary-inference"),
        runner,
    )

    with inference:
        serve_application(inference)
```

`InferenceResult.frame_timestamp_ns` always identifies the monotonic timestamp
of the exact input frame. `InferencePreviewSource` reads the independent JPEG
branch at preview speed and passes every fresh JPEG, the newest result, and a
`PreviewOverlayContext` to an application renderer. Its `detection_age_ns`
property can be exposed directly in UI telemetry. The renderer remains
model-specific; capture remains unaware of boxes, masks, YOLO, or other output
schemas. See [consumer integration](packages/consumers/README.md).

Workers expose `healthy`, `consecutive_failures`, the current `last_error`, and
historical `last_failure`. A successful callback clears only the current error
state. Failures are logged with rate limiting, optionally reported through
`on_error`, and retried with bounded exponential backoff. An expired GPU lease
detected before inference is counted as a dropped frame rather than a model
failure.

### Diagnostics

`camera.stats()` exposes a typed, immutable `CameraStats` snapshot for external
health endpoints, monitoring adapters, watchdogs, dashboards, telemetry, and
alerting without requiring log parsing or adding telemetry dependencies to the
toolkit core.

```python
stats = camera.stats()

if stats.consecutive_failures:
    notify_watchdog(stats)
```

The FastAPI health endpoint also exposes these capture diagnostics in JSON.

### Camera error contract

The public camera API exposes stable exceptions so integrations never need to
parse logs or match backend-specific messages:

| Exception | Meaning |
| --- | --- |
| `CameraDependencyError` | Required JetPack runtime dependency, such as system OpenCV or GStreamer support, is unavailable. |
| `CameraConfigurationError` | A camera setting, pipeline property, or API option is invalid. It also subclasses `ValueError` for compatibility. |
| `CameraOpenError` | The selected capture backend cannot open the configured camera. |
| `CameraReadError` | A frame cannot be read safely, including calling `read()` before startup. |
| `CameraTimeoutError` | An operation that requires a frame reaches its deadline, for example the hardware benchmark or snapshot command. |
| `CameraRecoveryError` | Capture recovery is exhausted or a capture worker cannot stop safely. |

`Camera.read(timeout=...)` intentionally retains its lightweight polling
contract: it returns `None` when no newer raw frame arrives in time. This is
appropriate for latest-frame processing loops that do not want exceptions on a
normal temporary absence of frames.

### Reusing one camera for inference and preview

The browser preview can attach to an existing camera instead of constructing a
second capture pipeline. This lets raw-frame inference, snapshots, MJPEG,
diagnostics, and browser preview share the same source:

```python
from imx_camera_toolkit import Camera
from imx_camera_toolkit.preview import create_preview_app

camera = Camera()
app = create_preview_app(camera)

with camera:
    run_my_pipeline(camera)
```

`create_preview_app()` enables JPEG preview on the provided camera but leaves
its lifecycle to the application. It does not start, stop, or replace the
camera instance.

For a transport-only preview backed by any existing latest-frame source, use
`serve()`. It does not create a second capture pipeline or own the source
lifecycle:

```python
from imx_camera_toolkit import Camera
from imx_camera_toolkit.preview import serve

with Camera() as camera:
    serve(camera, port=8000)
```

### Generic processed-image preview

`PreviewServer` is a model-agnostic image transport. It does not define or
interpret detections, bounding boxes, labels, masks, segmentation, or tracking
metadata. An external application may publish any image it has already drawn:

```python
from imx_camera_toolkit.preview import PreviewServer

preview = PreviewServer()
app = preview.create_app()

frame = camera.read()
result = model(frame.image)
annotated = draw_results(frame.image, result)
preview.publish(annotated)
```

It can also forward the latest raw frame from an existing source without
creating another camera capture pipeline:

```python
camera_preview = PreviewServer(source=camera)
processed_preview = PreviewServer(source=processed_frame_buffer)
```

`PreviewServer` owns only its JPEG transport worker. It never starts, stops, or
otherwise owns the supplied source.

Raw application frames and browser preview JPEGs use separate publication
paths. `camera.latest_frame()` returns the newest raw `Frame`, while
`camera.latest_jpeg()` returns the newest encoded preview image. For
processing-only deployments, disable JPEG work entirely:

```python
camera = Camera(CameraConfig(enable_preview=False))
```

For development, install the additional test, lint, and type-checking tools:

```bash
uv sync --extra preview --group dev
```

## Packaging

The project uses Hatchling and produces standard Python source and wheel
distributions. Build artifacts are written to `dist/`:

```bash
uv build
```

The installed package provides the `imx-camera` console command. The legacy
`imx-camera-toolkit` command remains available as a compatibility alias. Both
are available inside the project environment through `uv run`.

## Development quality checks

The project uses Ruff for linting and import hygiene, and mypy in strict mode
for static type verification. Unit and integration tests use an in-memory mock
camera, so they run on ordinary development machines and in CI without a CSI
sensor or a Jetson camera stack.

Install development dependencies and run the standard quality gate:

```bash
uv sync --extra preview --group dev
uv run ruff check .
uv run mypy imx_camera_toolkit packages tests
uv run pytest -m "not benchmark"
```

Deterministic capture and MJPEG framing benchmarks are deliberately separate
from the normal test suite. They measure toolkit overhead only; they do not
represent sensor, ISP, JPEG encoder, network, or browser performance. A
separate, explicit hardware benchmark compares BGR/CPU capture-only, capture
plus JPEG, and optionally capture plus an application-owned CPU model. It
reports dropped source frames and mean raw-frame delivery latency.

```bash
uv run pytest -m benchmark
uv run imx-camera benchmark all --frames 1000 --json
uv run imx-camera benchmark camera --frames 300
uv run imx-camera benchmark camera --frames 300 \
  --cpu-model my_application.models:cpu_model
```

GitHub Actions runs linting and strict type checking in a dedicated job, unit
and integration tests in a separate job, and verifies that source and wheel
distributions can be built. This separation makes static-analysis and runtime
failures immediately distinguishable in CI.

## Command-line interface and diagnostics

The installed `imx-camera` command provides deployment checks, physical-camera
smoke testing, snapshots, browser preview, and benchmarks:

```bash
uv run imx-camera info --json
uv run imx-camera diagnose --hardware
uv run imx-camera test --frames 30 --timeout 5
uv run imx-camera snapshot snapshot.jpg
uv run imx-camera preview --port 8000
uv run imx-camera benchmark capture --frames 1000
uv run imx-camera benchmark camera --frames 300
uv run imx-camera benchmark camera --frames 300 \
  --cpu-model my_application.models:cpu_model
```

`diagnose --hardware` checks for the locally installed Argus GStreamer element
and V4L2 command-line utility. `test` is the explicit physical-camera check: it
opens the configured sensor, waits for the first frame, measures a sequence of
distinct frames, and verifies that the backend is released. It may access the
camera and should therefore not run concurrently with another camera process.

The `capture`, `streaming`, and `all` benchmark targets use `MockCamera`, so
they are repeatable on both Jetson and non-Jetson development machines. The
`camera` target is an explicit physical-camera benchmark. It always measures
BGR/CPU capture-only and capture plus JPEG. With `--cpu-model MODULE:CALLABLE`,
it also loads an application-owned callable and passes each BGR host image to
it, measuring capture plus the actual CPU model without making that model a
toolkit dependency. These benchmarks report local capture throughput, dropped
frames, and mean delivery latency; they do not measure network or browser
throughput.

## Examples

The runnable examples are intentionally small and use only public imports:

- [`examples/capture_frames.py`](examples/capture_frames.py) reads raw latest
  frames for an external processing pipeline.
- [`examples/browser_preview.py`](examples/browser_preview.py) starts the
  simple preview facade.
- [`examples/shared_preview.py`](examples/shared_preview.py) attaches generic
  browser transport to one existing camera instance.

## Running the local preview

Install the optional preview dependencies before starting a browser server:

```bash
uv sync --extra preview
```

For the simplest Python integration, start a preview through the public facade:

```python
from imx_camera_toolkit import preview

preview()
```

The facade starts a simple browser view and releases camera resources during
server shutdown. Its defaults are sensor `0`, `1280x720`, `30` FPS,
`127.0.0.1`, and port `8000`.

Configure the camera and server explicitly when needed:

```python
from imx_camera_toolkit import preview

preview(
    sensor_id=0,
    width=1920,
    height=1080,
    fps=30,
    port=8000,
)
```

For reusable configuration, use the object-oriented variant:

```python
from imx_camera_toolkit import CameraPreview

camera_preview = CameraPreview()
camera_preview.run()
```

To start the repository's local launcher directly:

```bash
uv run python main.py
```

Open the following address on the Jetson:

```text
http://localhost:8000/
```

Remote development binds must be explicit, for example
`--host 192.0.2.10 --allow-remote`. Use field mode rather than this unauthenticated
development override on a deployed Jetson.

The server also exposes FastAPI documentation at
`http://localhost:8000/docs`.

Camera hardware is opened during FastAPI startup and released during shutdown.
The default server binds only to `127.0.0.1:8000`.

## HTTP interface

| Endpoint | Required scope | Purpose |
| --- | --- | --- |
| `GET /healthz` | Public | Minimal process liveness without diagnostics. |
| `GET /debug/health` | `admin` | Camera state, counters, timestamps, clients, and background errors. |
| `GET /` | `stream:read` | Customizable browser preview. |
| `GET /api/camera/snapshot` | `stream:read` | Most recent JPEG frame. |
| `GET /api/camera/mjpeg` | `stream:read` | Continuous MJPEG stream. |
| `GET /api/camera/control` | `camera:read` | Current controls, capabilities, modes, and software HDR. |
| `PATCH /api/camera/control` | `camera:control` | Applies a validated partial camera-control update. |
| `GET /api/camera/control/profiles` | `camera:read` | Lists process-local control profiles. |
| `PUT` or `DELETE /api/camera/control/profiles/{name}` | `profiles:write` | Stores or deletes a profile. |
| `POST /api/camera/control/profiles/{name}/apply` | `profiles:write` + `camera:control` | Applies a stored profile. |
| `GET /api/camera/software-hdr` | `camera:read` | Current software-HDR configuration. |
| `PUT /api/camera/software-hdr` | `camera:control` | Configures software HDR. |

`GET /api/health` remains a deprecated, `admin`-protected alias. Interactive
documentation is available only outside field mode.

The snapshot endpoint returns `204 No Content` when the frame specified by
`after` remains current after the configured wait period. It returns `503` when
the camera has not supplied an image.

### Runtime camera controls

Runtime controls are applied through a partial JSON update. Omitted fields
preserve their current values. Set `exposure_us` or `gain` to `null` to restore
automatic control where it is supported.

```bash
curl -X PATCH http://localhost:8000/api/camera/control \
  -H 'Content-Type: application/json' \
  -d '{"exposure_us": 5000, "gain": 2.0, "awb_mode": "daylight"}'
```

Supported values are determined by `packages/camera_control/config.yml` and by
the active JetPack driver. Exposure and gain are expressed in microseconds and
linear gain respectively. The API rejects unsupported properties and malformed
values with `422 Unprocessable Entity`.

Changing sensor mode, switching native sensor HDR, or restoring automatic
exposure/gain may require a capture-pipeline restart. On supported JetPack 6
configurations, manual exposure and gain are applied live through V4L2 to avoid
an Argus dynamic-range update limitation.

### Native and software HDR

Native HDR is available only when the connected sensor driver publishes a
configured HDR sensor mode. It is selected through the standard runtime
control endpoint.

Software HDR is intended for sensors without native HDR. It captures a
three-exposure bracket on the Jetson and fuses the images before JPEG encoding:

```bash
curl -X PUT http://localhost:8000/api/camera/software-hdr \
  -H 'Content-Type: application/json' \
  -d '{"enabled": true, "base_exposure_us": 5000, "settle_frames": 2}'
```

Software HDR changes sensor exposure during capture. Do not combine it with
manual exposure or gain changes from another client while it is enabled.

## Configuration

Each functional layer has an independently documented YAML configuration file.
Invalid, unreadable, or missing configuration files fall back to validated
built-in defaults.

| File | Scope |
| --- | --- |
| [`packages/camera/config.yml`](packages/camera/config.yml) | Sensor ID, capture/output dimensions, frame rates, JPEG quality, and image transformation. |
| [`packages/camera_control/config.yml`](packages/camera_control/config.yml) | Supported Argus properties, optional native HDR sensor modes, and initial runtime-control values. |
| [`packages/stream/config.yml`](packages/stream/config.yml) | MJPEG multipart boundary and frame wait timeout. |
| [`packages/api/config.yml`](packages/api/config.yml) | FastAPI metadata and snapshot wait timeout. |

`CameraConfig` is the preferred immutable contract for passing and comparing
camera settings between components. It contains the sensor, capture/output
dimensions, `fps`, flip method, optional sensor mode, output format, and preview
setting. The compatible `Camera` accepts only `FrameFormat.BGR_CPU`: GStreamer
converts NV12/NVMM to BGR and the backend materializes an owned array in host
RAM before Python receives it.

```python
from imx_camera_toolkit import Camera, CameraConfig, FrameFormat

camera = Camera(
    CameraConfig(
        sensor_id=1,
        capture_width=1920,
        capture_height=1080,
        output_width=1280,
        output_height=720,
        output_format=FrameFormat.BGR_CPU,
        fps=30,
        enable_preview=False,
    )
)
```

Constructor arguments remain available for backwards compatibility and take
precedence over the relevant YAML or explicit configuration values.
`copy=False` only suppresses the subsequent Python API copy of this BGR host
array; it is not a GPU zero-copy mode.

### Hardware profiles

`CameraConfig.from_profile()` selects a curated static hardware profile. The
profile catalogue reports an explicit verification status and never treats an
unverified sensor mode as fully supported.

```python
from imx_camera_toolkit import Camera, CameraConfig, get_camera_profile

profile = get_camera_profile("imx219-1080p")
assert profile.status.value == "tested"

with Camera(CameraConfig.from_profile("imx219-1080p")) as camera:
    frame = camera.read()
```

The current supported profile is `imx219-1080p` for the tested IMX219-77
camera module: Argus sensor mode 2, 1920×1080 capture at 30 FPS, and 1280×720
output. `community-tested` and `experimental` are defined status levels for
future entries; no unverified profiles are currently advertised.

PyYAML is not a core dependency. When it is unavailable, the corresponding
component ignores its YAML file and uses validated built-in defaults instead.

## Browser view customization

The API provides two bundled browser views. `simple` is the default and
preserves the preview layout without a control panel; `advanced` adds runtime
camera controls. Select a variant when constructing the application:

```python
from imx_camera_toolkit.api import create_app

app = create_app(view_mode="simple")
```

The selected file is read for every request, so HTML, CSS, and JavaScript
changes are visible after a browser refresh without restarting the service.

| View mode | Template | Purpose |
| --- | --- | --- |
| `"simple"` | [`view/simple.html`](view/simple.html) | Default live preview without camera controls. |
| `"advanced"` | [`view/advanced.html`](view/advanced.html) | Live preview and runtime camera controls. |

The template must preserve the required stream image marker:

```html
<img data-camera-stream src="{{ camera_stream_url }}" alt="Live camera feed">
```

The API verifies this element and replaces `{{ camera_stream_url }}` with
`/api/camera/mjpeg` before returning the page. Classes, styles, layout,
surrounding markup, and the alternative text may be modified freely.

Applications that need a view outside the repository can construct the API with
an explicit path:

```python
from imx_camera_toolkit.api import create_app
from imx_camera_toolkit import Camera, CameraConfig

app = create_app(
    Camera(CameraConfig(sensor_id=1, enable_preview=True)),
    view_mode="simple",
    view_path="/etc/imx-camera/index.html",
)
```

`view_path` takes precedence over `view_mode`, allowing an application to use
its own template while retaining the same API factory.

## Operational characteristics

- Camera capture runs in a background thread.
- JPEG encoding is rate-limited independently of sensor capture rate.
- Only the latest JPEG is retained in memory.
- `MJPEGStream` does not own the camera lifecycle, allowing multiple clients to
  share the same camera instance.
- The API disables HTTP caching for snapshots and MJPEG responses.
- Camera startup failures are surfaced through the FastAPI lifespan.
- A running camera attempts backend recovery after unexpected backend errors or
  sustained failed reads. The default policy makes three attempts with
  exponential backoff. The global consecutive-attempt budget resets only after
  a valid frame is captured, not merely after a backend reaches `PLAYING`.
- GPU backend startup requires a first NVMM frame. Argus `AlreadyAllocated`
  failures are surfaced immediately as `CameraOpenError` instead of entering a
  recovery loop.
- `/debug/health` exposes `recovery_attempts`, `recoveries`, and
  `last_recovery_error`, in addition to capture counters and `last_error`.
- If recovery is exhausted, capture stops cleanly and the final error remains
  observable through the health endpoint.

## Troubleshooting

| Symptom | Likely cause and corrective action |
| --- | --- |
| `CameraDependencyError: System OpenCV with GStreamer support is required` | Recreate `.venv` with `uv venv --system-site-packages --allow-existing .venv`, then run `uv sync`. |
| Camera cannot open | Verify the CSI connection, `sensor_id`, JetPack installation, and availability of `nvarguscamerasrc`. |
| Argus cannot connect | Confirm that `nvargus-daemon` is running and that the process has access to the Jetson camera stack. Containers additionally require the Argus socket and relevant device access. |
| No image at the preview endpoint | Inspect authenticated `/debug/health` for `last_error`, camera state, and frame counters. |
| Intermittent camera failures | Inspect authenticated `/debug/health` for recovery counters and `last_recovery_error`. Run `uv run imx-camera diagnose --hardware` to verify Argus and V4L2 prerequisites. |
| Runtime control is rejected | Inspect `GET /api/camera/control` and declare only properties and sensor modes supported by the installed `nvarguscamerasrc` driver. |
| Software HDR cannot start | Confirm that manual sensor exposure control is available. Disable software HDR before applying external manual exposure or gain settings. |
| Remote browser cannot connect | Confirm network reachability to port `8000` and review host firewall or reverse-proxy configuration. |

## Security considerations

Field mode enables scoped bearer authentication, disables OpenAPI/Swagger,
limits request bodies and request rates, validates `Host`, adds browser security
headers, and hides detailed health data behind `admin`. Generate independent
256-bit tokens for the minimum required scopes and store only their SHA-256
digests in a root/process-owned `0600` or `0640` JSON file:

```json
{
  "schema_version": 1,
  "tokens": [
    {"sha256": "<64 lowercase hex characters>", "scopes": ["stream:read"]},
    {"sha256": "<another digest>", "scopes": ["admin"]}
  ]
}
```

Start a loopback service behind a TLS or mTLS reverse proxy:

```bash
uv run imx-camera preview --field-mode \
  --token-file /etc/imx-camera/tokens.json \
  --allowed-host camera.example --behind-tls-proxy
```

For a direct remote listener, field mode also requires an explicit Host
allowlist and either `--tls-certfile` plus `--tls-keyfile`, or
`--behind-tls-proxy` when the proxy forwards the HTTPS scheme. Never put an
admin token in preview JavaScript; use a separate `stream:read` credential or
let an authenticated reverse proxy enforce preview access. The recommended
topology is TLS/mTLS on `:443` forwarding to `127.0.0.1:8000`.
Deployment-specific device identity or signing keys can additionally be
provisioned through Jetson OP-TEE secure storage; the toolkit intentionally
does not copy those private keys into browser assets or ordinary YAML files.

## License

This project is licensed under the [MIT License](LICENSE). Copyright (c) 2026
Jakub Bielecki.

The MIT License permits use, copying, modification, distribution,
sublicensing, and sale of the software, provided that the copyright notice and
license text are included in copies or substantial portions of the software.
The software is provided without warranty; consult the [LICENSE](LICENSE) file
for the complete terms.
