# GPU release checklist

Use this checklist before tagging a release that contains the experimental GPU
path. A host-only green workflow is not sufficient evidence for Jetson support.

## Host gate

```bash
uv lock --check
uv run ruff check .
uv run mypy imx_camera_toolkit packages tests
uv run pytest tests/unit tests/integration -m "not hardware and not benchmark"
uv build
```

## Jetson gate

Run the manual `Jetson hardware validation` GitHub Actions workflow on a
self-hosted runner labeled `jetson`. Select the connected IMX219 or IMX477 and
provide a local ONNX model path. The workflow must verify:

- NVMM capture at 1280x720 and 1920x1080 at 30 FPS;
- simultaneous TensorRT and H.264 hardware preview;
- TensorRT/ONNX Runtime output parity;
- native CUDA interop build against the installed JetPack headers;
- benchmark JSON containing CPU, GPU, FPS, drops, mean latency, and p95 latency.

Archive the benchmark together with `nvpmodel -q`, `jetson_clocks --show`,
JetPack/L4T version, sensor, cooling state, model hash, and TensorRT version.

## Release gate

- Confirm `GpuCamera` still requires `experimental=True`.
- Update the tested hardware matrix without promoting untested sensors.
- Review `CHANGELOG.md` and package version consistency.
- Build wheel and source distribution from a clean checkout.
- Inspect wheel contents for native sources, browser assets, and public modules.
- Create a signed tag only after both CI workflows pass.
- Never publish or share cached TensorRT `.engine` files as portable assets.
