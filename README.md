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
| [`packages/stream`](packages/stream/README.md) | Framework-neutral construction of `multipart/x-mixed-replace` MJPEG body parts. |
| [`packages/api`](packages/api/README.md) | FastAPI application, camera lifecycle management, snapshots, health reporting, MJPEG delivery, and browser view rendering. |
| [`packages/testing`](packages/testing/mock_camera.py) | Deterministic, thread-safe camera substitute for tests and benchmarks without Jetson hardware. |
| [`view/advanced.html`](view/advanced.html) | Browser preview with runtime camera controls. |
| [`view/simple.html`](view/simple.html) | Browser preview without a camera-control panel. |

## Platform requirements

- NVIDIA Jetson Orin Nano or a compatible Jetson platform.
- A supported CSI-connected IMX camera sensor.
- JetPack with NVIDIA Argus, `nvarguscamerasrc`, and GStreamer support.
- System OpenCV with GStreamer support. On JetPack this is normally supplied by
  the `python3-opencv` system package.
- Python 3.10 or newer.
- [uv](https://docs.astral.sh/uv/) for dependency and virtual-environment
  management.

OpenCV is intentionally not installed from PyPI. The JetPack-provided build is
required because it integrates with NVIDIA's camera and GStreamer stack.

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

Add the stable branch to the consuming project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "imx-camera-toolkit @ git+https://github.com/bi3lu/imx-camera-toolkit.git@v0.3.1"
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
    "imx-camera-toolkit[preview] @ git+https://github.com/bi3lu/imx-camera-toolkit.git@v0.3.1"
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

The public namespace also provides `api`, `camera_control`, and `stream`:

```python
from imx_camera_toolkit.camera_control import CameraController
from imx_camera_toolkit.stream import MJPEGStream
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

`read()` returns a formal `Frame` with an opaque `image`, monotonic `sequence`,
nanosecond `timestamp_ns`, optional hardware `capture_timestamp_ns`, dimensions,
and pixel `format`. It retains only the newest BGR frame, may skip stale frames,
and never performs JPEG encoding or inference. The default `copy=True` gives
the caller an independent image buffer; `copy=False` returns a read-only shared
payload for zero-copy-oriented pipelines.

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

The installed package provides the `imx-camera-toolkit` console command. It is
also available inside the project environment through `uv run`.

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
represent sensor, ISP, JPEG encoder, network, or browser performance.

```bash
uv run pytest -m benchmark
uv run imx-camera-toolkit benchmark all --frames 1000 --json
```

GitHub Actions runs linting and strict type checking in a dedicated job, unit
and integration tests in a separate job, and verifies that source and wheel
distributions can be built. This separation makes static-analysis and runtime
failures immediately distinguishable in CI.

## Command-line interface and diagnostics

The installed `imx-camera-toolkit` command provides non-destructive deployment
checks and deterministic benchmarks:

```bash
uv run imx-camera-toolkit diagnose --json
uv run imx-camera-toolkit diagnose --hardware
uv run imx-camera-toolkit benchmark capture --frames 1000
uv run imx-camera-toolkit serve --host 0.0.0.0 --port 8000
```

`diagnose --hardware` checks for the locally installed Argus GStreamer element
and V4L2 command-line utility. It does not alter sensor settings or open a
camera stream.

The `benchmark` commands use `MockCamera`, so they are repeatable on both
Jetson and non-Jetson development machines. They characterize Python-side
capture publication and multipart framing overhead, rather than end-to-end
camera or network throughput.

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
`0.0.0.0`, and port `8000`.

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

From another device, replace `localhost` with the Jetson host name or IP
address.

The server also exposes FastAPI documentation at
`http://localhost:8000/docs`.

Camera hardware is opened during FastAPI startup and released during shutdown.
The default server binds to `0.0.0.0:8000`; use an appropriate firewall or
reverse proxy before exposing it beyond a trusted network.

## HTTP interface

| Endpoint | Response | Purpose |
| --- | --- | --- |
| `GET /` | `text/html` | Customizable browser preview containing the live camera feed. |
| `GET /api/health` | `application/json` | Camera state, frame availability, counters, timestamps, and background capture errors. |
| `GET /api/camera/control` | `application/json` | Current runtime controls, declared capabilities, sensor modes, and software-HDR state. |
| `PATCH /api/camera/control` | `application/json` | Applies a validated partial update to exposure, gain, AWB, denoise, sensor mode, or native HDR. |
| `GET /api/camera/control/profiles` | `application/json` | Lists process-local runtime-control profiles. |
| `PUT /api/camera/control/profiles/{name}` | `application/json` | Stores the current runtime controls under a name. |
| `POST /api/camera/control/profiles/{name}/apply` | `application/json` | Applies a stored runtime-control profile. |
| `DELETE /api/camera/control/profiles/{name}` | No content | Deletes a stored runtime-control profile. |
| `GET /api/camera/software-hdr` | `application/json` | Current software-HDR configuration and resolved exposure brackets. |
| `PUT /api/camera/software-hdr` | `application/json` | Enables, disables, or configures Jetson-side three-exposure HDR fusion. |
| `GET /api/camera/snapshot` | `image/jpeg` | Most recent JPEG frame. Supports the optional `after` frame-number parameter. |
| `GET /api/camera/mjpeg` | `multipart/x-mixed-replace` | Continuous MJPEG stream suitable for a browser image element or another HTTP client. |
| `GET /docs` | `text/html` | Interactive OpenAPI documentation supplied by FastAPI. |

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
dimensions, `fps`, flip method, optional sensor mode, and preview setting:

```python
from imx_camera_toolkit import Camera, CameraConfig

camera = Camera(
    CameraConfig(
        sensor_id=1,
        capture_width=1920,
        capture_height=1080,
        output_width=1280,
        output_height=720,
        fps=30,
        enable_preview=False,
    )
)
```

Constructor arguments remain available for backwards compatibility and take
precedence over the relevant YAML or explicit configuration values.

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
  exponential backoff.
- `/api/health` exposes `recovery_attempts`, `recoveries`, and
  `last_recovery_error`, in addition to capture counters and `last_error`.
- If recovery is exhausted, capture stops cleanly and the final error remains
  observable through the health endpoint.

## Troubleshooting

| Symptom | Likely cause and corrective action |
| --- | --- |
| `CameraDependencyError: System OpenCV with GStreamer support is required` | Recreate `.venv` with `uv venv --system-site-packages --allow-existing .venv`, then run `uv sync`. |
| Camera cannot open | Verify the CSI connection, `sensor_id`, JetPack installation, and availability of `nvarguscamerasrc`. |
| Argus cannot connect | Confirm that `nvargus-daemon` is running and that the process has access to the Jetson camera stack. Containers additionally require the Argus socket and relevant device access. |
| No image at the preview endpoint | Inspect `/api/health` for `last_error`, camera state, and frame counters. |
| Intermittent camera failures | Inspect `/api/health` for recovery counters and `last_recovery_error`. Run `uv run imx-camera-toolkit diagnose --hardware` to verify Argus and V4L2 prerequisites. |
| Runtime control is rejected | Inspect `GET /api/camera/control` and declare only properties and sensor modes supported by the installed `nvarguscamerasrc` driver. |
| Software HDR cannot start | Confirm that manual sensor exposure control is available. Disable software HDR before applying external manual exposure or gain settings. |
| Remote browser cannot connect | Confirm network reachability to port `8000` and review host firewall or reverse-proxy configuration. |

## Security considerations

The supplied API does not implement authentication, authorization, transport
encryption, or request rate limiting. It is appropriate for local development
and controlled networks. Production deployment should place the service behind
a properly configured reverse proxy that provides access control and TLS.

## License

This project is licensed under the [MIT License](LICENSE). Copyright (c) 2026
Jakub Bielecki.

The MIT License permits use, copying, modification, distribution,
sublicensing, and sale of the software, provided that the copyright notice and
license text are included in copies or substantial portions of the software.
The software is provided without warranty; consult the [LICENSE](LICENSE) file
for the complete terms.
