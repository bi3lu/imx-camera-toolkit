# IMX Camera

Lightweight CSI camera capture for NVIDIA Jetson devices. The package opens an
IMX sensor through NVIDIA Argus, converts frames to BGR with GStreamer, and
keeps only the latest JPEG frame in memory.

This design is intended for live previews and streaming: a slow consumer gets
the newest available frame instead of accumulating a queue of stale frames.

## Internal architecture

The public `Camera` class is a coordinator. It owns lifecycle and capture
thread management, then connects the following focused components:

| Module | Responsibility |
| --- | --- |
| [config/](config) | YAML loading, validation, and resolved camera settings. |
| [pipeline/](pipeline) | Safe construction of the Argus GStreamer pipeline. |
| [backends/](backends) | PyGObject GStreamer capture with OpenCV GStreamer fallback. |
| [controls/](controls) | Live Argus properties and V4L2 exposure/gain updates. |
| [processing/software_hdr.py](processing/software_hdr.py) | Three-exposure software HDR fusion on the Jetson. |
| [publishing/](publishing) | JPEG encoding, newest-frame retention, and consumer synchronization. |

This separation keeps capture backends, sensor controls, and image processing
replaceable without changing the `Camera` API used by applications.

## Capture recovery

`Camera` automatically attempts to reopen its capture backend after an
unexpected backend exception or a sustained sequence of failed reads. The
default policy uses up to three retries with exponential backoff. The retry
budget spans reopened backends and resets only after a valid frame arrives.
The GPU backend also requires its first NVMM frame before `open()` succeeds;
an Argus `AlreadyAllocated` error fails fast as `CameraOpenError`. Recovery
statistics are available through `recovery_attempts`, `recoveries`, and
`last_recovery_error`; the FastAPI health endpoint exposes the same values.

Applications can supply a stricter or more tolerant policy:

```python
from imx_camera_toolkit.camera import Camera, CameraRecoveryPolicy

camera = Camera(
    recovery_policy=CameraRecoveryPolicy(
        max_attempts=5,
        initial_backoff=0.5,
        max_consecutive_read_failures=30,
    )
)
```

If every recovery attempt fails, capture stops cleanly and the final exception
remains available through `camera.last_error` and the API health endpoint.

## Requirements

- NVIDIA Jetson with JetPack and a connected CSI IMX camera.
- The `nvarguscamerasrc` GStreamer element supplied by JetPack.
- OpenCV built with GStreamer support in the Python environment that runs the
  application.
- `uv` for project dependency management.

The project does not install OpenCV with `pip`. On Jetson, OpenCV is provided
by JetPack; make sure that `import cv2` works in the Python interpreter used by
your application.

The core Python package has no PyPI runtime dependencies. YAML configuration is
used when PyYAML is available; otherwise `Camera` safely uses its built-in
configuration defaults.

For deployment choices and the stable GPU compatibility policy, see the
[CPU/GPU and browser mode guide](../../../docs/GPU_PATH_GUIDE.md).

Install the project dependencies with:

```bash
uv sync
```

## Stable raw-frame API

`Camera.read()` is the primary CPU integration API for external image-processing
pipelines. It returns a `Frame` containing a processed BGR image and
metadata. It does not encode JPEG data and does not perform inference.

```python
from imx_camera_toolkit import Camera, CameraConfig

config = CameraConfig(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    output_width=1280,
    output_height=720,
    fps=30,
)

with Camera(config) as camera:
    frame = camera.read()

    if frame is not None:
        result = my_tensor_rt_engine(frame.image)
```

`Frame` provides the following stable fields:

| Field | Meaning |
| --- | --- |
| `image` | Host-memory BGR payload, normally a NumPy array. |
| `sequence` | Monotonically increasing capture identifier. |
| `timestamp_ns` | Monotonic acquisition timestamp in nanoseconds. |
| `capture_timestamp_ns` | Optional hardware-provided capture timestamp; currently `None` for the standard backend. |
| `width`, `height` | Output image dimensions in pixels. |
| `format` | Pixel format, currently `"BGR"`. |

This enables latency measurements without binding the toolkit to an AI
framework:

```python
import time

latency_ns = time.monotonic_ns() - frame.timestamp_ns
```

The camera retains exactly one raw frame. `read()` returns the newest available
frame, never creates an unbounded queue, and may skip older frames when capture
runs faster than the consumer. It returns `None` when no frame arrives before
the timeout or capture stops while waiting.

```python
frame = camera.read(timeout=1.0, copy=False)
```

By default, `copy=True` returns an independent BGR image copy owned by the
caller. With `copy=False`, `frame.image` is the camera's shared image payload.
This avoids one additional host-memory copy for CPU consumers and preprocessing,
but the caller must treat the shared payload as read-only.

This is host-memory copy avoidance only; it does not guarantee GPU zero-copy.
GPU-first sources use the separate public `GpuFrame` contract and never replace
`raw_frame` with an NVMM buffer.

## GPU frame contract

`GpuFrame` identifies `FrameFormat.NV12_NVMM` in `MemoryType.NVMM` and exposes
one borrowed DMA-BUF descriptor or checked `GpuBufferHandle` around an opaque
`Gst.Buffer`/`NvBufSurface` resource.
It intentionally has no NumPy `image` field. Capture owns the buffer. Direct
`GpuCamera.read()` values remain short-lived borrowed leases invalidated by the
next publication. Every `subscribe_latest()` slot receives an independent,
reference-counted lease; replacing an unread slot releases only that slot and
does not invalidate a frame already being processed. Access after release
raises `GpuFrameExpiredError`.

Public `GpuFrameSource` and `CaptureFrameSource` protocols allow applications
to implement CPU/GPU consumers without importing capture internals. Test code
can use `MockFrameSource`, `mock_cpu_frame`, and `mock_gpu_frame` from
`imx_camera_toolkit.testing` without Jetson hardware.

## GPU-first NVMM capture

`GpuCamera` is the explicit Jetson-only capture API. It keeps NV12 in NVMM
through the inference appsink and returns a borrowed `GpuFrame`; it never maps
that buffer to NumPy or calls `Gst.Buffer.map()`. The optional browser branch
is split before encoding, so a slow MJPEG client cannot stall inference:

```text
nvarguscamerasrc -> NV12/NVMM -> nvvidconv -> NV12/NVMM -> tee
  +-> queue(max=1, leaky) -> NV12/NVMM -> gpu_sink(max=1, drop=true)
  +-> queue(max=1, leaky) -> NV12/NVMM -> nvjpegenc -> preview_sink(max=1, drop=true)
```

```python
from imx_camera_toolkit import CameraConfig, GpuCamera

config = CameraConfig(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    output_width=1920,
    output_height=1080,
    fps=30,
    enable_preview=True,
)

with GpuCamera(config) as camera:
    frame = camera.read(timeout=1.0)
    if frame is not None:
        run_tensor_rt(frame.payload(), frame.width, frame.height)
```

For direct `read()`, the consumer must complete all access to `frame.payload()`
before capture publishes the next frame. Subscription consumers instead release
their retained frame after processing; toolkit workers do this automatically.
The payload is an opaque borrowed `Gst.Buffer` whose caps were checked for
`video/x-raw(memory:NVMM), format=NV12`. The toolkit does not choose a TensorRT
model, preprocessing policy, CUDA stream, or engine cache.

When preview is enabled, `GpuCamera` implements the JPEG source contract used
by `MJPEGStream`, so the encoded branch can feed browser clients without a BGR
round trip. Any error or shutdown closes the complete tee pipeline; recovery
recreates both branches together.

## Production video encoder branch

`VideoEncoderConfig` adds a third, optional tee branch without changing the
JPEG debug path or borrowed inference frames:

```python
from imx_camera_toolkit import (
    GpuCamera,
    VideoEncoderBackend,
    VideoEncoderConfig,
    VideoCodec,
)

camera = GpuCamera(
    enable_preview=True,  # optional MJPEG debug output remains available
    video_config=VideoEncoderConfig(
        codec=VideoCodec.H264,
        backend=VideoEncoderBackend.AUTO,
        bitrate_bps=4_000_000,
        keyframe_interval=30,
    ),
)
```

`AUTO` prefers `nvv4l2h264enc`/`nvv4l2h265enc` and falls back to `x264enc` for
H.264 when NVENC is absent. Jetson Orin Nano therefore uses x264. Capture,
inference, and overlay remain NVMM; only its encoder branch converts to I420
system memory. `HardwareVideoConfig` remains a compatibility alias.
`subscribe_video(name)` gives a transport one latest compressed access-unit
slot. `video_stats` exposes recent encoder FPS and encoded bitrate without
retaining a per-frame history.

For a custom encoder, pass `encoder_pipeline_factory` returning a public
`VideoEncoderPipeline`. `set_video_overlay()` rebuilds from that factory, so
applications never need to patch `GpuCamera._pipeline`. Negotiated output is
available as the immutable `encoded_stream_description` contract.

An injected `VideoOverlayRenderer` must declare `MemoryType.NVMM`. When it is
active, the branch first makes an isolated device-side copy, then invokes the
renderer from the GStreamer encoder thread. `set_video_overlay()` supports
wiring an inference result source before camera startup. Host-memory overlays
remain available through the separate JPEG/MJPEG consumer adapter and are not
silently inserted into the NVMM production branch.

Physical validation is opt-in and must be run once with each target module:

```bash
IMX_CAMERA_SENSOR=IMX219 IMX_CAMERA_SENSOR_ID=0 \
  uv run pytest -m hardware tests/hardware/test_gpu_capture.py
IMX_CAMERA_SENSOR=IMX477 IMX_CAMERA_SENSOR_ID=0 \
  uv run pytest -m hardware tests/hardware/test_gpu_capture.py
```

Each run exercises 1280×720 and 1920×1080 at 30 FPS, verifies borrowed NVMM
frames from branch A, and verifies hardware JPEG output from branch B. The
unit suite also constructs all four sensor/resolution scenarios without camera
hardware, but that structural test is not a physical compatibility claim.

The IMX219 matrix has been run successfully for both resolutions on JetPack
6.2.2 and Jetson Orin Nano. IMX477 remains pending until that physical module
is connected and the same test completes; pipeline-construction coverage alone
does not mark it as verified.

## Pipeline observability

`Camera.stats()` includes fixed-size latency aggregates for transfer,
inference, encoder, and end-to-end processing, plus immutable per-consumer drop
counters. Capture records stages it owns. External model consumers report
their own work through `record_stage_latency("inference", duration_ns)` and
`record_consumer_drop(name, count)`; neither method retains frames or creates a
queue.

`read_image(timeout=..., copy=...)` is available for compatibility-oriented
code that requires only the image payload. New integrations should use `read()`
to retain frame sequence, timestamps, dimensions, and format.

## Diagnostics

`Camera.stats()` returns an immutable `CameraStats` snapshot without requiring
log parsing or a telemetry library. It is suitable as the direct input for a
health endpoint, Prometheus adapter, watchdog, dashboard, telemetry process,
or alerting policy.

```python
from imx_camera_toolkit import Camera, CameraConfig

with Camera(CameraConfig()) as camera:
    stats = camera.stats()

    if stats.consecutive_failures:
        report_capture_problem(stats)
```

| Field | Meaning |
| --- | --- |
| `captured_frames` | Successful source reads since the camera was created. |
| `dropped_frames` | Failed source reads and frames intentionally not published as raw frames. |
| `capture_fps` | Recent successful source-read rate over a one-second window. |
| `last_frame_timestamp_ns` | Monotonic timestamp of the most recent successful source read. |
| `recovery_count` | Successful backend recovery operations. |
| `consecutive_failures` | Current uninterrupted source-read failure count. |
| `running` | Whether the capture worker is active. |

Statistics do not introduce Prometheus, OpenTelemetry, or other telemetry
dependencies into the core package. Those integrations remain the
responsibility of the consuming application.

## Independent preview path

Raw frame publication and JPEG preview encoding are independent paths:

```text
Camera capture
├── raw/latest frame → application processing
└── JPEG preview     → browser and MJPEG clients
```

Use `latest_frame()` for an immediate, non-blocking lookup of the most recent
raw `Frame`. `latest_jpeg()` provides the latest encoded preview image.

```python
frame = camera.latest_frame(copy=False)
jpeg = camera.latest_jpeg()
```

Applications that do not need a browser preview or MJPEG output can remove JPEG
encoding from the capture loop:

```python
camera = Camera(CameraConfig(enable_preview=False))
```

Raw frames remain available through `read()` and `latest_frame()`, while
`latest_jpeg()` returns `None`. This avoids the per-frame JPEG encoding cost.
The FastAPI API and `MJPEGStream` retain one shared `Camera` instance, so
multiple browser or streaming clients do not start additional capture
pipelines.

An application that already owns capture for inference or another processing
pipeline can attach the browser preview to that same instance:

```python
from imx_camera_toolkit import Camera, CameraConfig
from imx_camera_toolkit.preview import create_preview_app

camera = Camera(CameraConfig(enable_preview=False))
app = create_preview_app(camera)

with camera:
    run_my_pipeline(camera)
```

`create_preview_app()` enables JPEG preview on the supplied camera but does
not start or stop it. The enclosing application remains responsible for the
camera lifecycle, and FastAPI snapshots, MJPEG, diagnostics, and inference all
use the same capture pipeline.

For a separate processed-image view, use the generic `PreviewServer`. It
transports opaque images only and does not contain any inference-model or
overlay semantics:

```python
from imx_camera_toolkit.preview import PreviewServer

preview = PreviewServer()
app = preview.create_app()

frame = camera.read()
annotated = draw_results(frame.image, model(frame.image))
preview.publish(annotated)
```

Alternatively, `PreviewServer(source=camera)` forwards the latest raw frame
from the existing camera without creating another capture pipeline. A custom
processed frame buffer may be used when it implements
`read(timeout=..., copy=False)` and returns an image or toolkit `Frame`.

## JPEG preview API

```python
from imx_camera_toolkit import Camera, CameraConfig

with Camera(CameraConfig(enable_preview=True)) as camera:
    frame_number, jpeg = camera.wait_for_jpeg(0, timeout=2.0)

    if jpeg is not None:
        with open("frame.jpg", "wb") as image_file:
            image_file.write(jpeg)
```

`start()` is idempotent, so calling it on an already running camera has no
effect. Always call `stop()` when not using the context manager.

```python
from imx_camera_toolkit import Camera, CameraConfig

camera = Camera(CameraConfig(enable_preview=True))
camera.start()

try:
    frame_number, jpeg = camera.wait_for_jpeg(0)

finally:
    camera.stop()
```

## Configuration

`CameraConfig` is the preferred explicit configuration contract. It is a frozen
and slotted dataclass, so it is safe to compare, serialize with
`dataclasses.asdict`, and pass between application components. It controls the
CSI sensor ID, capture and output resolution, frame rate, sensor mode,
`nvvidconv` flip method, explicit output format, and optional JPEG preview.

`CameraConfig.output_format` defaults to `FrameFormat.BGR_CPU`, the only format
accepted by the compatible `Camera`. In this mode GStreamer converts the
sensor's NV12/NVMM frame to BGR, transfers it to system memory, and the backend
materializes an owned host array. `output_memory` is therefore `MemoryType.CPU`
and `copies_to_host_memory` is `True`.

```python
from imx_camera_toolkit import Camera, CameraConfig, FrameFormat

config = CameraConfig(
    sensor_id=1,
    capture_width=1920,
    capture_height=1080,
    output_width=1280,
    output_height=720,
    output_format=FrameFormat.BGR_CPU,
    fps=30,
    flip_method=0,
    enable_preview=False,
)
camera = Camera(config)
```

`Camera.read(copy=False)` reuses that already materialized BGR host array. It
only disables the additional copy normally made by the Python API and never
promises NVMM, CUDA, or GPU zero-copy access.

## Hardware profiles

Curated profiles describe only static hardware and frame-layout settings. They
do not contain runtime controls, image-processing choices, JPEG quality, or
application networking settings.

```python
from imx_camera_toolkit import Camera, CameraConfig, get_camera_profile

profile = get_camera_profile("imx219-1080p")
config = CameraConfig.from_profile("imx219-1080p")

with Camera(config) as camera:
    frame = camera.read()
```

Each profile exposes a verification status. A status is evidence about the
specific configuration, not a blanket support claim for every operating mode
of a sensor.

| Profile | Camera module | Status | Capture | Output |
| --- | --- | --- | --- | --- |
| `imx219-1080p` | IMX219-77 | `tested` | 1920×1080 at 30 FPS, Argus mode 2 | 1280×720 |

The catalog also defines `community-tested` and `experimental` statuses for
future profiles. No profiles for unverified sensors or modes are currently
included. `imx219-77-1080p` is accepted as an alias for the tested profile.

The portable hardware-only representation is available without parsing YAML:

```python
profile.hardware_settings()
```

It returns the following structure, which can be serialized as a profile file
by an application if needed:

```yaml
sensor_id: 0
sensor_mode: 2

capture:
  width: 1920
  height: 1080
  fps: 30

output:
  width: 1280
  height: 720
```

Default settings live in [config.yml](config.yml) and are loaded only when
`Camera()` is created without an explicit `CameraConfig`.

If the file is missing, unreadable, malformed, or contains invalid values, the
camera uses its built-in defaults. The entire configuration falls back to those
defaults to avoid starting a camera with a partially invalid setup.

Legacy constructor arguments remain supported and override YAML or an explicit
configuration during migration:

```python
from imx_camera_toolkit.camera import Camera

camera = Camera(
    sensor_id=1,
    output_width=1280,
    output_height=720,
    quality=85,
)
```

To load a configuration from another location, pass `config_path`:

```python
camera = Camera(config_path="/etc/imx-camera/config.yml")
```

The resolved settings are available as `camera.config`, and the final GStreamer
pipeline string is available as `camera.pipeline`.

## Reading frames

`wait_for_jpeg(previous_frame_number, timeout)` waits for a frame newer than
`previous_frame_number` and returns `(frame_number, jpeg)`. It returns the
latest frame after a timeout or when the camera stops. Before the first frame,
`jpeg` is `None`.

Useful state and metrics:

- `camera.running` - whether the capture thread is active.
- `camera.frame_available` - whether a JPEG frame is available.
- `camera.jpeg` - the latest JPEG bytes, or `None`.
- `camera.frames_captured` and `camera.frames_encoded` - capture metrics.
- `camera.last_error` - an exception raised by the background capture loop, if
  one occurred.

## Troubleshooting

If `Camera.start()` cannot open the sensor, check the CSI cable connection,
`sensor_id`, and that `nvarguscamerasrc` is available on the device. If the
module reports that OpenCV is unavailable, run it with a JetPack Python/OpenCV
environment that includes GStreamer support.
