# MJPEG Stream

Framework-neutral MJPEG streaming primitives for JPEG frames produced by the
camera package. `MJPEGStream` turns the latest camera frames into valid
`multipart/x-mixed-replace` body parts that can be passed to an HTTP framework
or written directly to a client connection.

The stream does not open, start, stop, or otherwise own the camera. One running
camera can therefore serve multiple independent stream consumers.

## Requirements

- A running camera implementing `running` and `wait_for_jpeg()`. The
  `imx_camera_toolkit.camera.Camera` class provides this interface.
- An HTTP server or framework when exposing the stream over a network.

This package intentionally has no HTTP-framework dependency. It can be used
with FastAPI, Flask, Django, a standard-library server, or another transport.

## Basic usage

Start the camera once in your application lifecycle, then create a stream for a
consumer:

```python
from imx_camera_toolkit import Camera, CameraConfig
from imx_camera_toolkit.stream import MJPEGStream

camera = Camera(CameraConfig(enable_preview=True))
camera.start()

try:
    stream = MJPEGStream(camera)

    for multipart_part in stream:
        # Write multipart_part to the connected HTTP client.
        pass

finally:
    camera.stop()
```

Each item yielded by `MJPEGStream` contains all data required for one MJPEG
part: the boundary, `Content-Type`, `Content-Length`, JPEG bytes, and trailing
CRLF.

## HTTP framework integration

For example, an application that already uses FastAPI can return the iterator
through `StreamingResponse`:

```python
from fastapi.responses import StreamingResponse

from imx_camera_toolkit.stream import MJPEGStream


def camera_feed(camera):
    stream = MJPEGStream(camera)
    return StreamingResponse(stream, media_type=stream.content_type)
```

The application should create and start the camera outside this handler, then
stop it during application shutdown. Do not create one camera per client.

## Configuration

Default settings are stored in [config.yml](config.yml) and are loaded when
`MJPEGStream()` or `stream_mjpeg()` is created.

- `boundary` is the ASCII MIME multipart boundary without a leading `--`.
- `timeout` is the maximum number of seconds to wait for a newer camera frame.

When the configuration file is missing, unreadable, malformed, or invalid, the
built-in defaults are used. Constructor arguments have priority over YAML:

```python
stream = MJPEGStream(
    camera,
    boundary="camera",
    timeout=1.0,
)
```

Use another configuration file with `config_path`:

```python
stream = MJPEGStream(camera, config_path="/etc/imx-camera/stream.yml")
```

The resolved settings are available as `stream.config`. Use
`stream.content_type` as the HTTP response media type.

## API

- `MJPEGStream(camera, ...)` creates a reusable MJPEG iterator.
- `stream_mjpeg(camera, ...)` creates a one-shot iterator.
- `build_mjpeg_part(jpeg, boundary)` formats one JPEG image as a multipart
  body part.

`frames_sent` tracks the number of yielded frames, while `last_frame_number`
stores the latest yielded camera frame number. The stream only emits newer
frames; it intentionally does not buffer frames for slow consumers.
