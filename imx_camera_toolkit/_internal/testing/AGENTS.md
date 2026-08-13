# Test-double module guidance

These implementations are part of the stable public `imx_camera_toolkit.testing`
namespace for downstream application tests.

## Rules

- Keep doubles deterministic, thread-safe, lightweight, and independent of
  physical camera/GStreamer/CUDA availability.
- Match public lifecycle, latest-value, statistics, wait/notification, and
  error contracts closely enough for integration tests; do not mimic private
  backend details unnecessarily.
- `MockFrameSource` retains one frame only. Replacing/stopping a GPU frame must
  invalidate the prior lease.
- `mock_gpu_frame()` wraps an opaque object in a real `GpuBufferHandle`; tests
  using it must still honor release/expiry semantics.
- `MockCamera` may expose synthetic counters for diagnostics. Keep their units
  and names aligned with production snapshots.

When adding a public test double, export it through both internal and public
`testing/__init__.py`, document the ownership contract, and add namespace and
behavior tests.

## Validation

```bash
uv run pytest tests/unit/test_mock_camera.py \
  tests/unit/test_frame_contracts.py tests/unit/test_frame_source.py
```
