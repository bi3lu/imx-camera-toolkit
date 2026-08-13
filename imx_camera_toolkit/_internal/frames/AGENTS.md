# Frame-contract module guidance

This module is the smallest application integration boundary for capture. Keep
it independent of inference models, browser transport, and concrete camera
backends.

## Contracts

- `FrameSource` is CPU/BGR compatible.
- `GpuFrameSource` yields borrowed NV12/NVMM frames.
- `CaptureFrameSource` allows model-neutral code to accept either explicit
  memory domain without converting between them.
- `CameraFrameSource` adapts an already-running camera and does not own its
  lifecycle.

The CPU adapter uses `copy=False` by default, so its image is shared read-only
host memory. `copy=True` is the explicit owned-copy option. This is not a GPU
zero-copy API.

Do not add queues, preprocessing, batching, model state, or automatic format
conversion here. Extend protocols only for metadata needed by generic capture
consumers and keep them runtime-checkable when the public contract requires it.

## Validation

```bash
uv run pytest tests/unit/test_frame_contracts.py \
  tests/unit/test_frame_source.py
```

Test CPU and GPU implementations through the public `testing` helpers and
verify lease expiry separately from normal `None` timeout behavior.
