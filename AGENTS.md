# IMX Camera Toolkit repository guidance

These instructions are for human contributors, library users inspecting the
source, and coding agents. They apply to the whole repository. A more specific
`AGENTS.md` in a descendant directory adds module-level rules for that subtree.

## Project contract

IMX Camera Toolkit targets CSI-connected IMX sensors on NVIDIA Jetson, with
Jetson Orin Nano and JetPack 6.2.2 as the documented baseline. It provides
camera capture, controls, latest-frame consumers, optional TensorRT interop,
and browser delivery. It is not a model zoo, tracker, ROS/DeepStream wrapper,
or general video analytics framework.

Keep these stable paths distinct:

- `Camera` returns owned or shared BGR frames in CPU memory for OpenCV/NumPy.
- `GpuCamera` returns borrowed NV12/NVMM leases for CUDA/TensorRT consumers.
- MJPEG is a simple debug/browser path; WebRTC/HLS is the production browser
  path.
- Capture and inference keep only the newest pending frame. Do not introduce
  unbounded queues or let a slow consumer block capture.

## Using the library correctly

- Application code imports from `imx_camera_toolkit` or its public subpackages,
  never from `imx_camera_toolkit._internal`.
- Own camera lifecycle explicitly, preferably with `with Camera(...)` or
  `with GpuCamera(...)`. Do not create a camera per request or browser client.
- Treat `Camera.read(copy=False).image` as read-only shared CPU data.
- Treat direct `GpuFrame` values as short-lived borrowed leases. A newer direct
  publication invalidates the previous frame. Retained subscription frames
  must be released; toolkit consumers release them automatically.
- Prepare expensive TensorRT engines before opening capture when possible.
- Keep model decoding, NMS, labels, tracking, and model-specific overlays in
  application code.
- Use field mode, scoped tokens, an exact Host allowlist, and TLS for remote
  deployments. Never put credentials, models, private keys, TensorRT engines,
  or generated caches in source control.

## Environment and dependencies

On Jetson, create the environment with access to JetPack system packages:

```bash
uv venv --system-site-packages --allow-existing .venv
uv sync
```

Select only the optional groups needed by the change: `preview`,
`production-preview`, `tensorrt-build`, `model-security`, or `tensorrt-test`.
CUDA, TensorRT, OpenCV, GStreamer, and PyGObject must remain compatible with
the target JetPack image. Do not replace JetPack-provided CUDA, TensorRT,
OpenCV, or PyGObject with unrelated PyPI wheels.

## Code and compatibility rules

- Python support is 3.10 through 3.12; write typed Python compatible with 3.10.
- Ruff, Black, and strict mypy configuration in `pyproject.toml` are canonical.
- Docstrings are English, follow the Google Python style, and describe
  ownership, blocking, errors, and memory domains where those details matter.
- Keep the core importable without optional FastAPI, PyYAML, TensorRT, CUDA, or
  GStreamer Python dependencies. Optional integrations load lazily and fail
  with actionable dependency errors.
- Preserve public exports and behavior unless the task explicitly changes the
  public contract. Update public `__all__`, namespace tests, examples, and docs
  together when adding an API.
- Prefer monotonic nanosecond timestamps for latency and ordering. Do not mix
  them with Unix time without an explicit conversion and name.
- Keep configuration validation fail-closed for security settings. Existing
  non-security YAML loaders intentionally fall back to complete built-in
  defaults rather than accepting a partially invalid document.

## Validation

Run the narrowest relevant test first, then the repository gates:

```bash
uv run --frozen ruff check .
uv run --frozen black --check .
uv run --frozen mypy imx_camera_toolkit tests
uv run --frozen pytest tests/unit tests/integration \
  -m "not hardware and not benchmark" --no-cov
```

`pre-commit run --all-files` executes the same host gates, except its Black hook
formats in place. Hardware and benchmark tests are opt-in; never report them as
passing unless they ran on the stated Jetson, sensor, power mode, and software
stack. See `tests/AGENTS.md` and `docs/RELEASE_CHECKLIST.md` for those commands.

## Documentation discipline

- Keep documentation in English and link detailed procedures from
  `docs/README.md` instead of duplicating them.
- Distinguish tested hardware from planned support. Do not turn an inference,
  sensor, encoder, or network assumption into a support claim.
- Commands must state whether they run on a workstation or the Jetson and must
  not expose secrets through URLs, JavaScript, or committed configuration.
- When behavior changes, update the nearest component README, the public guide
  if applicable, and troubleshooting or release checks affected by the change.

## Repository map

| Path | Responsibility |
| --- | --- |
| `imx_camera_toolkit/` | Stable public Python namespace and private implementation. |
| `native/` | Jetson NvBufSurface/EGL/CUDA pybind11 bridge. |
| `examples/` | Small application-owned integration examples using public APIs. |
| `view/` | Bundled simple, advanced, and production browser clients. |
| `tests/` | Unit, integration, hardware, and benchmark validation. |
| `docs/` | Architecture, deployment, and release documentation. |
