# IMX Camera

Lightweight CSI camera capture for NVIDIA Jetson devices. The package opens an
IMX sensor through NVIDIA Argus, converts frames to BGR with GStreamer, and
keeps only the latest JPEG frame in memory.

This design is intended for live previews and streaming: a slow consumer gets
the newest available frame instead of accumulating a queue of stale frames.

## Requirements

- NVIDIA Jetson with JetPack and a connected CSI IMX camera.
- The `nvarguscamerasrc` GStreamer element supplied by JetPack.
- OpenCV built with GStreamer support in the Python environment that runs the
  application.
- `uv` for project dependency management.

The project does not install OpenCV with `pip`. On Jetson, OpenCV is provided
by JetPack; make sure that `import cv2` works in the Python interpreter used by
your application.

Install the project dependencies with:

```bash
uv sync
```

## Quick start

```python
from packages.camera.camera import Camera

with Camera() as camera:
    frame_number, jpeg = camera.wait_for_jpeg(0, timeout=2.0)

    if jpeg is not None:
        with open("frame.jpg", "wb") as image_file:
            image_file.write(jpeg)
```

`start()` is idempotent, so calling it on an already running camera has no
effect. Always call `stop()` when not using the context manager.

```python
from packages.camera.camera import Camera

camera = Camera()
camera.start()

try:
    frame_number, jpeg = camera.wait_for_jpeg(0)

finally:
    camera.stop()
```

## Configuration

Default settings live in [config.yml](config.yml) and are loaded when
`Camera()` is created. The file controls the CSI sensor ID, capture and output
resolution, frame rates, JPEG quality, and `nvvidconv` flip method.

If the file is missing, unreadable, malformed, or contains invalid values, the
camera uses its built-in defaults. The entire configuration falls back to those
defaults to avoid starting a camera with a partially invalid setup.

Constructor arguments override values loaded from YAML:

```python
from packages.camera.camera import Camera

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
