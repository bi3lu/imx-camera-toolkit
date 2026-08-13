# Public package guidance

This subtree contains both the stable import surface and `_internal`
implementations. Applications should see only the public surface.

## Public namespace

- `imx_camera_toolkit/__init__.py` exposes the most common capture, frame,
  consumer, inference, and preview symbols.
- Public subpackages (`camera`, `frames`, `consumers`, `inference`, `api`,
  `production_preview`, `stream`, `camera_control`, `controls`, and `testing`)
  are thin, explicit facades over implementations.
- `preview` and the FastAPI production application are optional. Preserve lazy
  imports so importing the core library does not require FastAPI/Uvicorn.

| Public module | Use it for |
| --- | --- |
| `camera` | CPU/GPU capture, frame/video contracts, profiles, and pipeline configuration. |
| `frames` | Minimal source protocols and the non-owning CPU camera adapter. |
| `consumers` | Latest-frame workers and asynchronous inference. |
| `inference` | Model-neutral TensorRT contracts, runner, and signed-model verification. |
| `camera_control` / `controls` | Validated runtime Argus settings. |
| `stream` | Framework-neutral MJPEG multipart formatting. |
| `api` / `preview` | Simple snapshot/MJPEG FastAPI applications. |
| `production_preview` | Shared WebRTC/HLS video delivery and GPU overlays. |
| `testing` | Deterministic camera and frame-source test doubles. |

When adding a public symbol:

1. Define and test the implementation in the responsible internal module.
2. Export it from the corresponding public subpackage with an explicit
   `__all__` entry.
3. Export it from the root only when it belongs in the common convenience API.
4. Extend `tests/unit/test_public_namespace.py` and user-facing documentation.

Do not expose an internal helper merely to make a test convenient. Prefer a
public protocol or immutable value object only when applications genuinely
need the contract.

## User-facing patterns

- Prefer context managers for cameras and consumers.
- Use `Camera`/`Frame` for CPU BGR work and `GpuCamera`/`GpuFrame` for NVMM.
- Use `frames` protocols to decouple application pipelines from camera classes.
- Use `consumers` for expensive latest-frame work; do not poll into a queue.
- Use `testing` test doubles in external test suites instead of importing
  private publishers or backends.
- Catch specific toolkit configuration/dependency/runtime errors where an
  application can recover; do not suppress capture or lease-expiry errors.

Preferred CPU capture pattern:

```python
from imx_camera_toolkit import Camera, CameraConfig

with Camera(CameraConfig(enable_preview=False)) as camera:
    frame = camera.read(timeout=1.0)
    if frame is not None:
        process_bgr(frame.image)
```

Preferred direct GPU lease pattern for a custom synchronous consumer:

```python
from imx_camera_toolkit import GpuCamera

camera = GpuCamera()
subscription = camera.subscribe_latest("custom-consumer")

with camera:
    frame = subscription.receive(timeout=1.0)
    if frame is not None:
        try:
            process_nvmm(frame.payload())
        finally:
            frame.release()
```

For ongoing expensive work, use `FrameConsumer` or `InferenceConsumer` rather
than writing a polling queue. For browser deployment, follow
`docs/GPU_CAMERA_YOLO_GUIDE.md`; do not copy a development bind into a field
service.

## Change rules

- Preserve aliases such as `HardwareVideoConfig`, `CameraFrame`,
  `CameraControls`, and `ExposureConfig` unless a deliberate compatibility
  change is requested.
- Keep public type annotations free of unnecessary concrete backend types.
- Public dataclasses and enums validate inputs at construction and should stay
  immutable where they represent configuration or snapshots.
- Public examples must import this namespace or a public subpackage, not
  `_internal`.
