# Camera Control

`camera_control` provides a framework-independent, validated runtime-control
layer for CSI cameras exposed through NVIDIA Argus. It keeps the desired camera
state separate from image capture and emits safe `nvarguscamerasrc` property
assignments for a caller-provided runtime handler.

The package is used by the FastAPI application, but it has no FastAPI
dependency and can be integrated with another UI, service, or application
loop.

## Internal architecture

The public [camera_control.py](camera_control.py) module is a compatibility
facade. The implementation is split by responsibility:

| Module | Responsibility |
| --- | --- |
| [models/](models) | Immutable settings, capabilities, modes, profiles, updates, and JSON serialization. |
| [capabilities/](capabilities) | Argus capability declarations and `gst-inspect-1.0` discovery. |
| [controls/](controls) | Validation and conversion of settings into `nvarguscamerasrc` properties. |
| [config/](config) | YAML loading, validation, and safe defaults. |
| [runtime/](runtime) | Thread-safe atomic state transitions, runtime-handler dispatch, and in-memory profiles. |

Applications can continue using imports such as
`from packages.camera_control.camera_control import CameraController`; the
public API remains unchanged.

## Supported controls

- Fixed exposure in microseconds, or automatic exposure.
- Fixed analog gain, or automatic gain.
- White-balance mode and auto-white-balance lock.
- Temporal denoise mode and optional strength.
- Explicit sensor-mode selection.
- Native sensor HDR mode when the configured sensor-mode metadata declares an
  HDR mode.
- Named in-memory profiles for application-level state management.

The controller validates a requested setting against the configured Argus
properties and sensor modes before passing it to the runtime handler. A failed
handler does not commit the requested state.

## Basic usage

Create a controller and use its returned update to inspect the requested
Argus properties:

```python
from packages.camera_control.camera_control import CameraController

controller = CameraController()
update = controller.update(
    exposure_us=5_000,
    gain=2.0,
    awb_mode="daylight",
    denoise_mode="high_quality",
)

print(update.source_properties)
```

To apply updates to the shared camera, provide a runtime handler. The camera
package decides whether a change can be applied live or requires a pipeline
restart:

```python
from packages.camera.camera import Camera
from packages.camera_control.camera_control import CameraController

camera = Camera()
controller = CameraController(
    runtime_handler=lambda update: camera.apply_argus_properties(
        update.source_properties,
        restart_required=update.restart_required,
    )
)

camera.start()

try:
    controller.set_exposure(5_000)
    controller.set_gain(2.0)

finally:
    camera.stop()
```

For JetPack 6 systems affected by the `nvarguscamerasrc` dynamic-range issue,
the camera package applies live exposure and gain through the V4L2 sensor
interface. Other supported properties continue through the Argus source.

## Configuration

Default settings are documented in [config.yml](config.yml). The file is read
automatically whenever `CameraController()` is created. It defines:

- `model`: an optional sensor identifier for API clients.
- `source_properties`: the Argus properties supported by the active driver.
- `sensor_modes`: optional mode metadata, including whether a mode is native
  HDR/WDR.
- `initial_settings`: the controller state before the first runtime update.

If the file is missing, unreadable, malformed, or invalid, validated built-in
defaults are used. The complete configuration falls back as one unit; a
partially invalid file is never applied.

Use another configuration file with `config_path`:

```python
from packages.camera_control.camera_control import CameraController

controller = CameraController(config_path="/etc/imx-camera/camera-control.yml")
```

Explicit constructor arguments have priority over the loaded configuration:

```python
from packages.camera_control.camera_control import (
    CameraCapabilities,
    CameraController,
)

capabilities = CameraCapabilities(
    source_properties=frozenset({"wbmode", "awblock", "tnr-mode"}),
    model="IMX sensor",
)

controller = CameraController(capabilities=capabilities)
```

Only declare properties and sensor modes that are actually supported by the
installed JetPack driver and the connected sensor. The
`discover_argus_capabilities()` helper can inspect the locally installed
`nvarguscamerasrc` element without opening the camera.

## Native HDR and software HDR

`CameraController.set_hdr(True)` selects a configured native HDR sensor mode.
It requires sensor-mode metadata with at least one entry whose `hdr` field is
`true`; it does not create HDR on an SDR sensor.

Software HDR is a separate feature of `packages.camera`. It captures exposure
brackets through the sensor control interface and fuses them on the Jetson.
Use the camera API or `Camera.configure_software_hdr()` for that mode. Do not
combine software HDR with manual exposure or gain updates from this controller
while the software HDR loop is active.

## Runtime state and profiles

`get_runtime_state()` returns JSON-ready settings, capabilities, source
properties, profiles, and a monotonically increasing revision number. This is
the representation used by the API layer.

Profiles are process-local and are not written to disk:

```python
controller.save_profile("indoor")
controller.set_awb("daylight")
controller.apply_profile("indoor")
```

Use `list_profiles()`, `get_profile()`, and `delete_profile()` to manage these
temporary settings snapshots. Persistent profile storage belongs to an
application configuration or database layer.

## Errors and restart behavior

`UnsupportedControlError` is raised when a requested setting needs an
undeclared Argus property or sensor mode. `ValueError` is raised for malformed
values, such as a non-positive exposure or an invalid enum value.

Each `CameraControlUpdate` exposes `restart_required`. Sensor-mode changes,
native HDR mode changes, and restoration of automatic exposure or gain may
require a capture-pipeline restart. The runtime handler is responsible for
honouring this flag.
