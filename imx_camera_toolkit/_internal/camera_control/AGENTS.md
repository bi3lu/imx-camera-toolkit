# Camera-control module guidance

This module validates desired Argus/V4L2 control state independently of camera
capture and exposes atomic updates through `CameraController`.

## Behavioral rules

- Validate requested settings against declared capabilities before invoking a
  runtime handler.
- A failing runtime handler must not commit the requested controller state or
  revision.
- Preserve the distinction between live-safe updates and changes marked
  `restart_required` (sensor mode, native HDR, and some automatic/manual
  transitions).
- Exposure values are microseconds and gain is linear. Validate types, finite
  bounds, enum values, and property availability before formatting Argus
  assignments.
- Profiles are in-memory process-local snapshots. Do not imply persistence.
- Native HDR selects a declared HDR sensor mode. Software HDR belongs to the
  CPU camera processing layer; never combine it silently with manual control.

## Configuration

`config.yml` and custom YAML documents describe source properties, sensor
modes, and initial settings. If a non-security control document is missing or
invalid, use the complete validated built-in configuration; do not merge a
partially invalid file. Only declare capabilities observed in the installed
`nvarguscamerasrc` and connected sensor.

Keep capability discovery read-only: `gst-inspect-1.0` probing must not open a
camera. Keep this module free of FastAPI dependencies.

## Validation

```bash
uv run pytest tests/unit/test_camera_control.py
```

Add unit coverage for invalid values, unsupported properties, handler
rollback, restart flags, serialization, and revision changes.
