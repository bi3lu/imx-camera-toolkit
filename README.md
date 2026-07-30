# IMX Camera Toolkit

## Abstract

IMX Camera Toolkit is a Python toolkit for acquiring, encoding, and serving
live video from CSI-connected IMX image sensors on NVIDIA Jetson systems. It is
designed for NVIDIA Jetson Orin Nano deployments using the JetPack software
stack, NVIDIA Argus, GStreamer, and the system-provided OpenCV build.

The project separates sensor acquisition, MJPEG framing, and HTTP delivery into
independent packages. This separation allows image processing, camera controls,
and alternative transport layers to be introduced without coupling them to the
camera capture loop.

## System architecture

```text
CSI IMX sensor
      |
      v
Camera package
Argus + GStreamer capture, BGR conversion, JPEG encoding
      |
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
| [`packages/stream`](packages/stream/README.md) | Framework-neutral construction of `multipart/x-mixed-replace` MJPEG body parts. |
| [`packages/api`](packages/api/README.md) | FastAPI application, camera lifecycle management, snapshots, health reporting, MJPEG delivery, and browser view rendering. |
| [`view/index.html`](view/index.html) | Customizable browser-facing HTML and CSS template for the live preview. |

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

Clone the repository and create a virtual environment that can access the
system OpenCV installation:

```bash
uv venv --system-site-packages
uv sync
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

## Development quality checks

The project uses Ruff for linting and import hygiene, and mypy in strict mode
for static type verification. Unit and integration tests use an in-memory mock
camera, so they run on ordinary development machines and in CI without a CSI
sensor or a Jetson camera stack.

Install development dependencies and run the standard quality gate:

```bash
uv sync --group dev
uv run ruff check .
uv run mypy packages tests
uv run pytest -m "not benchmark"
```

Deterministic capture and MJPEG framing benchmarks are deliberately separate
from the normal test suite. They measure toolkit overhead only; they do not
represent sensor, ISP, JPEG encoder, network, or browser performance.

```bash
uv run pytest -m benchmark
uv run imx-camera-toolkit benchmark all --frames 1000 --json
```

GitHub Actions runs linting and type checking in a dedicated job, unit and
integration tests in a separate job, and verifies that source and wheel
distributions can be built.

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

## Running the local preview

Start the API server from the repository root:

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
| `GET /api/camera/snapshot` | `image/jpeg` | Most recent JPEG frame. Supports the optional `after` frame-number parameter. |
| `GET /api/camera/mjpeg` | `multipart/x-mixed-replace` | Continuous MJPEG stream suitable for a browser image element or another HTTP client. |
| `GET /docs` | `text/html` | Interactive OpenAPI documentation supplied by FastAPI. |

The snapshot endpoint returns `204 No Content` when the frame specified by
`after` remains current after the configured wait period. It returns `503` when
the camera has not supplied an image.

## Configuration

Each functional layer has an independently documented YAML configuration file.
Invalid, unreadable, or missing configuration files fall back to validated
built-in defaults.

| File | Scope |
| --- | --- |
| [`packages/camera/config.yml`](packages/camera/config.yml) | Sensor ID, capture/output dimensions, frame rates, JPEG quality, and image transformation. |
| [`packages/stream/config.yml`](packages/stream/config.yml) | MJPEG multipart boundary and frame wait timeout. |
| [`packages/api/config.yml`](packages/api/config.yml) | FastAPI metadata and snapshot wait timeout. |

Constructor arguments take precedence over the relevant YAML values. For
example, a different CSI sensor can be selected with `Camera(sensor_id=1)`.

## Browser view customization

The root endpoint serves [`view/index.html`](view/index.html). The file is read
for every request, so HTML, CSS, and JavaScript changes are visible after a
browser refresh without restarting the service.

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
from packages.api.api import create_app
from packages.camera.camera import Camera

app = create_app(
    Camera(sensor_id=1),
    view_path="/etc/imx-camera/index.html",
)
```

## Operational characteristics

- Camera capture runs in a background thread.
- JPEG encoding is rate-limited independently of sensor capture rate.
- Only the latest JPEG is retained in memory.
- `MJPEGStream` does not own the camera lifecycle, allowing multiple clients to
  share the same camera instance.
- The API disables HTTP caching for snapshots and MJPEG responses.
- Camera startup failures are surfaced through the FastAPI lifespan and capture
  failures are reported by the health endpoint.

## Troubleshooting

| Symptom | Likely cause and corrective action |
| --- | --- |
| `OpenCV is not available` | Recreate `.venv` with `uv venv --system-site-packages --allow-existing .venv`, then run `uv sync`. |
| Camera cannot open | Verify the CSI connection, `sensor_id`, JetPack installation, and availability of `nvarguscamerasrc`. |
| Argus cannot connect | Confirm that `nvargus-daemon` is running and that the process has access to the Jetson camera stack. Containers additionally require the Argus socket and relevant device access. |
| No image at the preview endpoint | Inspect `/api/health` for `last_error`, camera state, and frame counters. |
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
