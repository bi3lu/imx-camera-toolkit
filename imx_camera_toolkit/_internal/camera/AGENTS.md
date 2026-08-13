# Camera module guidance

This module owns Argus/GStreamer capture, frame publication, recovery, runtime
camera updates, optional JPEG, and optional encoded video branches.

## Non-negotiable contracts

- `Camera` is the compatible CPU path. It produces BGR host-memory `Frame`
  values and may use OpenCV/GStreamer fallback behavior.
- `GpuCamera` is the GPU-first path. It negotiates NV12 with
  `video/x-raw(memory:NVMM)` and must not map pixel planes into a NumPy/BGR
  image.
- Direct GPU reads are borrowed until the next publication. Subscriber frames
  have independent retained leases and must remain valid until released.
- Publishers retain only the latest value. All tee branches and appsinks stay
  bounded/leaky so preview, inference, and encoding cannot back-pressure
  capture or accumulate stale frames.
- Pipeline rebuilds, recovery, and `stop()` must invalidate/release frames,
  close subscriptions, wake waiters, and leave repeated cleanup safe.

## Module boundaries

- `config/`: static configuration loading and validation.
- `pipeline/`: construction of CPU, GPU, and encoder GStreamer descriptions.
- `backends/`: runtime capture and negotiated-cap validation.
- `models/`: public frame, metrics, video, and error contracts.
- `publishing/`: latest-frame/JPEG/video synchronization and ownership.
- `controls/`: live Argus and V4L2 application.
- `processing/`: CPU-only software HDR.
- `profiles/`: curated hardware configurations and tested/planned status.

Do not patch a camera's private pipeline string from application code. Add a
validated config field, a public pipeline factory, or a backend capability.
Normalize Argus properties before interpolating them into GStreamer syntax.

## Implementation practices

- Keep state transitions under lifecycle locks; never hold a lock while doing
  avoidable application work.
- Treat backend open as successful only after required negotiated state/first
  frame checks pass.
- Recovery retry budgets reset only after a successful publication.
- Sensor mode, native HDR, and preview/video topology changes may require a
  full bounded restart. Preserve or restore the prior valid configuration when
  a reconfiguration fails.
- Software HDR stays on `Camera`; do not add a hidden CPU conversion to
  `GpuCamera`.
- On Orin Nano, H.264 production encoding may select x264 because NVENC is not
  present. This affects only the encoder branch, not the inference branch.

## Validation

Run camera-focused host tests first:

```bash
uv run pytest tests/unit/test_camera_config.py \
  tests/unit/test_camera_read.py tests/unit/test_camera_recovery.py \
  tests/unit/test_camera_stats.py tests/unit/test_gpu_pipeline.py
```

Physical NVMM validation is opt-in and requires `IMX_CAMERA_SENSOR` set to
`IMX219` or `IMX477`; see `tests/hardware/test_gpu_capture.py`. Never infer a
hardware support claim from mocked pipeline tests.
