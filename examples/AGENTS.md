# Example application guidance

Examples demonstrate supported composition patterns; they are not alternate
implementations of library internals.

## Rules

- Import only from `imx_camera_toolkit` and its public subpackages.
- Keep model-specific decoding, output names, NMS, labels, masks, and overlay
  mapping in the example/application layer.
- Default network listeners to loopback. A remote bind must be explicit and
  field-ready examples must fail closed on missing token, Host, or TLS policy.
- Do not embed secrets, fixed production hostnames, model files, generated
  TensorRT engines, or cache contents.
- Build/prepare expensive runners before opening the camera when possible.
- Use context managers and order ownership so consumers stop before cameras;
  close overlays/native resources in `finally` when they are outside a context.
- Use named subscriptions that describe their consumer for meaningful drop
  metrics.
- Explain the expected model input/output contract in help text or the linked
  guide; do not imply compatibility with every YOLO/ONNX export.

Examples should remain small enough to read, but validate security-critical
CLI combinations and produce actionable parser errors.

## Validation

```bash
uv run ruff check examples
uv run black --check examples
uv run mypy examples/yolo_detection.py
uv run python examples/yolo_detection.py --help
```

Do not run a hardware example in automated host tests unless it has an
explicit mockable entry point and cannot open a sensor accidentally.
