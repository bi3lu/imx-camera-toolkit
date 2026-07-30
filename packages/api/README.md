# IMX Camera API

FastAPI application exposing a shared NVIDIA Jetson CSI camera as individual
JPEG snapshots and a live MJPEG stream. It combines the `camera` package with
the framework-neutral MJPEG iterator from the `stream` package.

## Requirements

- A Jetson-compatible camera setup supported by the `camera` package.
- Project dependencies installed through `uv`:

  ```bash
  uv sync --extra preview
  ```

- FastAPI and Uvicorn, installed through the optional `preview` dependency
  group, to expose the application over HTTP.

## Application lifecycle

The module exports `create_app()`, a FastAPI application factory. Its lifespan
handler starts one shared `Camera` during application startup and stops it
during shutdown. Endpoints share that camera; the API does not create a camera
per request or per connected MJPEG client.

For a local camera preview, run:

```bash
uv run python main.py
```

The application can also be served directly with Uvicorn:

```bash
uv run uvicorn --factory imx_camera_toolkit.api:create_app --host 0.0.0.0 --port 8000
```

Using the factory avoids creating an application, reading its configuration, or
initializing its dependencies merely by importing the package.

FastAPI exposes interactive documentation at `/docs` and the OpenAPI schema at
`/openapi.json`.

## Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /` | Customizable HTML camera preview. |
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

## Browser view customization

`GET /` serves one of two customizable bundled templates. The default
[`view/simple.html`](../../view/simple.html) provides the live preview without
a control panel. [`view/advanced.html`](../../view/advanced.html) adds runtime
camera controls. You may freely change HTML, CSS, JavaScript, title, layout,
and styling. The selected template is read for every request, so refreshing the
browser applies changes without restarting the server.

The live camera image is required. Keep this element in the template (it may
have additional classes, attributes, and surrounding markup):

```html
<img data-camera-stream src="{{ camera_stream_url }}" alt="Live camera feed">
```

The API validates the `data-camera-stream` marker and replaces
`{{ camera_stream_url }}` with `/api/camera/mjpeg` before serving the page. If
the required image element is absent or changed, the root endpoint returns an
error instead of serving a view without a live feed.

## Configuration

[config.yml](config.yml) controls the FastAPI title, description, version, and
the snapshot endpoint timeout. Missing, unreadable, malformed, or invalid
configuration falls back to built-in defaults.

To create an application with another configuration file or a custom camera,
use the factory:

```python
from imx_camera_toolkit.api import create_app
from imx_camera_toolkit.camera import Camera

camera = Camera(sensor_id=1)
app = create_app(
    camera,
    config_path="/etc/imx-camera/api.yml",
    view_mode="simple",
    view_path="/etc/imx-camera/index.html",
)
```

The resolved configuration is available as `app.state.config`, and the chosen
view path as `app.state.view_path`. The selected bundled mode is available as
`app.state.view_mode`. `view_path` takes precedence over `view_mode`.

## Security

The API does not implement authentication or authorization. Do not expose it
directly to an untrusted network. Bind it to a private interface or place it
behind an authenticated reverse proxy before remote use.
