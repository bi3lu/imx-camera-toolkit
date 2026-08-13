# Test-suite guidance

Tests are divided by the evidence they provide. Keep skips explicit and never
promote host simulation to a hardware support claim.

## Test classes

- `unit/`: deterministic behavior without a physical camera; use public test
  doubles or narrowly scoped fake runtimes.
- `integration/`: components working together. Some tests use local GStreamer;
  TensorRT parity is additionally hardware-gated.
- `hardware/`: physical Jetson CSI camera, NVMM, TensorRT, and encoder evidence.
- `benchmarks/`: opt-in performance smoke tests; thresholds are meaningful only
  with recorded target conditions.

All test functions use assertions, deterministic cleanup, and the markers
declared in `pyproject.toml`. Avoid arbitrary sleeps in unit tests; coordinate
threads with events/conditions and use bounded timeouts.

## Default host gate

```bash
uv run --frozen pytest tests/unit tests/integration \
  -m "not hardware and not benchmark" --no-cov
```

Tests must close cameras, workers, subscriptions, runners, clients, and temp
files even when assertions fail. A test must not bind a non-loopback interface
or open hardware unless its marker and opt-in configuration make that clear.

## Hardware gates

Physical GPU capture requires:

```bash
IMX_CAMERA_SENSOR=IMX219 IMX_CAMERA_SENSOR_ID=0 \
  uv run pytest tests/hardware/test_gpu_capture.py
```

Use `IMX477` only when that exact sensor/configuration is under test. Concurrent
TensorRT/production preview requires `IMX_PRODUCTION_PREVIEW_HARDWARE=1`, a
trusted local `IMX_TENSORRT_ONNX` path, and optionally
`IMX_VIDEO_ENCODER_BACKEND`. TensorRT parity requires
`IMX_TENSORRT_INTEGRATION=1` and the `tensorrt-test` dependencies.

Record JetPack/L4T, sensor, resolution, FPS, power mode, clocks, cooling,
TensorRT version, model hash, selected encoder, and duration with performance
results. Cached `.engine` files are disposable local artifacts.

## What to cover

- Public APIs: success, invalid input, timeout, lifecycle, and cleanup.
- Concurrency: independent subscribers, latest-frame drops, wake-on-close, and
  lease release after errors.
- Security: authentication, scopes, Host/TLS policy, body/rate limits, safe
  files/paths, and hidden diagnostics/docs.
- GPU code: negotiated caps, no CPU fallback, expiry, cache compatibility, and
  numerical parity on hardware.
- Transport: shared encoding, client isolation, SDP/caps correctness, safe HLS
  assets, and metrics observed at the correct media stage.
