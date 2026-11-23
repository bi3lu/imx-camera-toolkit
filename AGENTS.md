# IMX Camera Toolkit contributor guide

This is the repository-wide `AGENTS.md`. Run commands from the repository root.
The toolkit provides Argus/GStreamer capture and controls for CSI-connected IMX
sensors on NVIDIA Jetson, latest-frame consumers, optional TensorRT interop,
and browser delivery. The documented baseline is IMX219-77 on Jetson Orin Nano
with JetPack 6.2.3. Model decoding, NMS, labels, tracking, and model-specific
overlays belong in applications.

## Repository map

| Path | Responsibility |
| --- | --- |
| `imx_camera_toolkit/` | Stable public namespace and thin subpackage facades |
| `imx_camera_toolkit/_internal/` | Capture, frame contracts, consumers, inference, controls, and transport implementations |
| `native/` | Jetson NvBufSurface/EGL/CUDA bridge |
| `examples/` | Small integrations using public APIs |
| `view/` | Bundled simple, advanced, and production browser clients |
| `tests/` | Unit, integration, hardware, and benchmark suites |
| `docs/` | Architecture, deployment, and release evidence |

## Environment and checks

On a workstation with Python 3.10–3.12, install the locked host test environment:

```bash
uv sync --frozen --extra preview --group dev
```

On Jetson, give the virtual environment access to JetPack system packages:

```bash
uv venv --system-site-packages --allow-existing .venv
uv sync --frozen
```

Select only the extras needed by the change (`preview`, `production-preview`,
`tensorrt-build`, `model-security`, or `tensorrt-test`). Keep JetPack's CUDA,
TensorRT, OpenCV, GStreamer, and PyGObject compatible with the target image.
Use the JetPack-provided packages for those runtimes.

Run the narrowest relevant test first, then the host gates:

```bash
uv run --frozen ruff check .
uv run --frozen black --check .
uv run --frozen mypy imx_camera_toolkit tests
uv run --frozen pytest tests/unit tests/integration \
  -m "not hardware and not benchmark" --no-cov
```

Local Git hooks use the optional `hooks` group and require Python 3.11 or 3.12.
Run them on a workstation. The Black hook formats in place.

```bash
UV_PYTHON=3.11 uv run --frozen --extra preview --group hooks pre-commit run --all-files
```

Host tests use deterministic doubles or local runtimes, synchronize with
events or conditions, use bounded waits, and clean up resources on failure.
Use the markers in `pyproject.toml` and keep test listeners on loopback.

Hardware and benchmark gates are opt-in. See the
[release checklist](docs/RELEASE_CHECKLIST.md) and
[Jetson workflow](.github/workflows/jetson-hardware.yml) for commands and inputs.
Record the actual sensor, JetPack/L4T, resolution, FPS, and relevant power mode,
encoder, and model details when reporting results. State any skipped gates.

## Code and public API

- Write typed Python compatible with 3.10. Follow Ruff, Black, and strict mypy
  configuration in `pyproject.toml`. Use English Google-style docstrings that
  describe ownership, blocking, errors, and memory domains where applicable.
- Applications and examples import only from `imx_camera_toolkit` or its public
  subpackages. Preserve public aliases and behavior. For new public symbols,
  update the subpackage's explicit `__all__`, the root facade when appropriate,
  the [namespace tests](tests/unit/test_public_namespace.py), and relevant docs
  and examples.
- Load optional FastAPI, PyYAML, TensorRT, CUDA, and GStreamer integrations
  lazily. Report actionable dependency errors and preserve exception causes at
  subsystem boundaries.
- Validate external inputs before acquiring resources or formatting GStreamer
  properties. Invalid non-security YAML falls back to the complete validated
  defaults. Invalid security settings fail closed.
- Use monotonic nanoseconds for latency and ordering. Make conversions to or
  from Unix time explicit in the name and code. Preserve the meaning and units
  of public metrics and the shape of CLI `--json` output.
- Keep public test doubles deterministic and thread-safe. Match public
  lifecycle, latest-frame, lease expiry, and wait/notification behavior. Use
  `imx_camera_toolkit.testing` in downstream tests.
- Name example subscriptions for their consumer so drop metrics are readable.
  Document model input and output requirements and validate security-sensitive
  CLI combinations before starting capture.

## Capture, ownership, and concurrency

- `Camera` produces CPU BGR frames. Treat `read(copy=False).image` as shared
  read-only data. Use the owned copy when mutating it. Software HDR stays on
  the CPU capture path.
- `GpuCamera` negotiates NV12 in NVMM. The next direct publication invalidates
  the previous direct lease. Subscriptions retain independent leases until
  release. Release retained frames on success, drop, error, and shutdown.
  Keep NVMM pixel planes off the CPU in the GPU path.
- Keep one replaceable pending slot per subscriber and bound GStreamer branches
  and appsinks. Capture must neither run application code nor wait for a slow
  consumer. Preserve independent subscriber drop counts and wake blocked
  waiters on shutdown.
- Prefer context managers. Stop consumers before their camera when the
  application owns both. Transport and source adapters borrow running sources
  unless their public API explicitly transfers lifecycle ownership. Reuse a
  shared camera across requests and clients.
- Confirm negotiated caps and first-frame checks as required by the backend
  before reporting startup success. Keep recovery bounded and reset its retry
  budget after successful publication. Invalidate leases, close subscriptions,
  and make cleanup safe to repeat after stop or rebuild.
- Validate controls against declared capabilities before applying them. Preserve
  the previous valid configuration and controller revision on failure. Honor
  restart flags, exposure units in microseconds, and linear gain. Keep
  capability discovery read-only and control profiles process-local.

See the [capture](imx_camera_toolkit/_internal/camera/README.md),
[controls](imx_camera_toolkit/_internal/camera_control/README.md),
[frames](imx_camera_toolkit/_internal/frames/README.md), and
[consumers](imx_camera_toolkit/_internal/consumers/README.md) contracts.

## Inference and native interop

- Prepare TensorRT engines before starting capture when possible. Keep runners
  model-neutral and give each worker its own runner and CUDA stream. A worker
  timeout does not cancel an active TensorRT call.
- Retain the `Gst.Buffer` reference while native code uses its `NvBufSurface`.
  Map only the descriptor on the host. Validate surface layout before kernel
  launch, use the caller's CUDA stream, and release native resources in reverse
  order on all paths. Preserve compiler and linker hardening.
- Treat serialized engines as target-local executable caches. Preserve cache
  compatibility checks, atomic writes, owner/mode checks, symlink rejection,
  and rebuild on invalid cache. For signed models, verify the exact signed
  manifest bytes, ONNX digest, and declared tensor names before engine use.

See the [inference guide](imx_camera_toolkit/_internal/inference/README.md) for
interop, cache metadata, and TensorRT parity validation.

## Browser delivery and security

- MJPEG is the debug preview path. Keep multipart formatting framework-neutral,
  validate MIME boundaries, and preserve snapshot statuses (`200`, `204`, and
  `503`). Keep `data-camera-stream` and `{{ camera_stream_url }}` in the simple
  and advanced templates.
- WebRTC/HLS shares one encoded branch across bounded client pipelines.
  Preserve H.264 for WebRTC and H.265 for HLS only. Keep x264's CPU memory
  transition isolated to the encoder branch on Orin Nano. Use real SPS/caps
  for SDP and start late WebRTC clients on an IDR. Limit HLS paths and retained
  files to the rolling playlist and expected segments.
- Render GPU overlays on an isolated encoder surface. Synchronize renderer work
  before encoding and preserve the configured overlay failure policy. Count
  access units, RTP packets, bytes, and decoded frames at their actual stages.
- Default listeners to loopback. Remote deployments use field mode, scoped
  tokens, an exact Host allowlist, and TLS. Preserve limits, security headers,
  hidden API docs, minimal public health checks, and admin-only diagnostics.
  Keep middleware streaming-safe.
- Preserve field-mode scopes. `stream:read` covers media and signaling,
  `camera:read` covers control state, `camera:control` covers control updates,
  and `profiles:write` covers profile mutation. `admin` grants full access.
- Keep the field-mode login shell data-free before authentication. Exchange
  browser tokens for session-only HttpOnly cookies with the existing SameSite
  and Secure policy. Keep credentials out of client storage, URLs, and logs.
  Preserve hashed-token file validation, including ownership, mode, and
  symlink checks.
- Keep credentials, private keys, models, TensorRT engines, and generated
  caches and native binaries out of source control.

See the [HTTP API](imx_camera_toolkit/_internal/api/README.md),
[MJPEG transport](imx_camera_toolkit/_internal/stream/README.md),
[production preview](imx_camera_toolkit/_internal/production_preview/README.md),
and [deployment guide](docs/GPU_CAMERA_YOLO_GUIDE.md).

## Documentation

Write in English and link to detailed procedures through the
[documentation index](docs/README.md). For behavior changes, update the nearest
component README, affected user guides, and relevant release checks. Label
workstation and Jetson commands, verify links and CLI syntax, and claim hardware
support only after the relevant physical test.
