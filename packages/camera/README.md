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
default policy uses up to three retries with exponential backoff. Recovery
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

Install the project dependencies with:

```bash
uv sync
```

## Stable raw-frame API

`Camera.read()` is the primary integration API for external image-processing
pipelines. It returns a `Frame` containing an opaque processed BGR image and
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
| `image` | Opaque BGR payload; it may be a NumPy array, CUDA buffer, DMA-BUF, or another future backend type. |
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
This avoids a copy for TensorRT, DeepStream, OpenCV, and CUDA pipelines, but
the caller must treat the shared payload as read-only.

`read_image(timeout=..., copy=...)` is available for compatibility-oriented
code that requires only the image payload. New integrations should use `read()`
to retain frame sequence, timestamps, dimensions, and format.

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
`nvvidconv` flip method, and optional JPEG preview.

```python
from imx_camera_toolkit import Camera, CameraConfig

config = CameraConfig(
    sensor_id=1,
    capture_width=1920,
    capture_height=1080,
    output_width=1280,
    output_height=720,
    fps=30,
    flip_method=0,
    enable_preview=False,
)
camera = Camera(config)
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

- `camera.running` — whether the capture thread is active.
- `camera.frame_available` — whether a JPEG frame is available.
- `camera.jpeg` — the latest JPEG bytes, or `None`.
- `camera.frames_captured` and `camera.frames_encoded` — capture metrics.
- `camera.last_error` — an exception raised by the background capture loop, if
  one occurred.

## Troubleshooting

If `Camera.start()` cannot open the sensor, check the CSI cable connection,
`sensor_id`, and that `nvarguscamerasrc` is available on the device. If the
module reports that OpenCV is unavailable, run it with a JetPack Python/OpenCV
environment that includes GStreamer support.
