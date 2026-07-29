# IMX Camera API

FastAPI application exposing a shared NVIDIA Jetson CSI camera as individual
JPEG snapshots and a live MJPEG stream. It combines the `camera` package with
the framework-neutral MJPEG iterator from the `stream` package.

## Requirements

- A Jetson-compatible camera setup supported by the `camera` package.
- Project dependencies installed through `uv`:

  ```bash
  uv sync
  ```

- An ASGI server to expose the application over HTTP. Uvicorn is a common
  choice, but it is not currently bundled with this package.

## Application lifecycle

The module exports `app`, a default FastAPI application. Its lifespan handler
starts one shared `Camera` during application startup and stops it during
shutdown. Endpoints share that camera; the API does not create a camera per
request or per connected MJPEG client.

When using Uvicorn after adding it to the project dependencies, the application
can be served with:

```bash
uv run uvicorn packages.api.api:app --host 0.0.0.0 --port 8000
```

FastAPI exposes interactive documentation at `/docs` and the OpenAPI schema at
`/openapi.json`.

## Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /` | JSON map of the camera API endpoints. |
| `GET /api/health` | Camera state, frame availability, capture metrics, and the latest background error. |
| `GET /api/camera/snapshot` | Latest JPEG camera image. |
| `GET /api/camera/mjpeg` | Live `multipart/x-mixed-replace` MJPEG response. |

### Snapshot endpoint

`GET /api/camera/snapshot` returns `image/jpeg` and includes the frame number
in the `X-Frame-Number` response header. Responses disable HTTP caching.

The optional `after` query parameter lets a client wait for a newer frame:

```text
GET /api/camera/snapshot?after=42
```

The endpoint waits for up to `snapshot_timeout` seconds. It returns:

- `200` with JPEG data when a frame is available;
- `204` when `after` is still the newest frame after the timeout;
- `503` when the camera has not produced a frame.

### MJPEG endpoint

`GET /api/camera/mjpeg` returns the latest JPEG frames as a multipart MJPEG
stream. Slow clients do not cause a server-side frame queue; they receive the
newest available frames and may skip older ones.

For a simple local preview, use the endpoint directly in a browser or in an
HTML image element:

```html
<img src="/api/camera/mjpeg" alt="Live camera feed">
```

## Configuration

[config.yml](config.yml) controls the FastAPI title, description, version, and
the snapshot endpoint timeout. Missing, unreadable, malformed, or invalid
configuration falls back to built-in defaults.

To create an application with another configuration file or a custom camera,
use the factory:

```python
from packages.api.api import create_app
from packages.camera.camera import Camera

camera = Camera(sensor_id=1)
app = create_app(
    camera,
    config_path="/etc/imx-camera/api.yml",
)
```

The resolved configuration is available as `app.state.config`.

## Security

The API does not implement authentication or authorization. Do not expose it
directly to an untrusted network. Bind it to a private interface or place it
behind an authenticated reverse proxy before remote use.
