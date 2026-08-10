# GPU release checklist

Use this checklist before tagging a release that contains the experimental GPU
path. A host-only green workflow is not sufficient evidence for Jetson support.

## Host gate

```bash
uv lock --check
uv run ruff check .
uv run mypy imx_camera_toolkit packages tests
uv run pytest tests/unit tests/integration -m "not hardware and not benchmark"
uv audit --frozen
uv build
uv export --all-extras --format cyclonedx1.5 --frozen \
  --output-file dist/sbom.cdx.json
```

## Jetson gate

Run the manual `Jetson hardware validation` GitHub Actions workflow on a
self-hosted runner labeled `jetson`. Select the connected IMX219 or IMX477 and
provide a local ONNX model path. The workflow must verify:

- NVMM capture at 1280x720 and 1920x1080 at 30 FPS;
- simultaneous TensorRT and H.264 production preview using the resolved
  NVENC/x264 backend;
- WebRTC decode smoke test with matching SDP fmtp, real RTP counters, a
  late-joining peer, and a subscription beginning mid-GOP;
- TensorRT/ONNX Runtime output parity;
- native CUDA interop build against the installed JetPack headers;
- benchmark JSON containing CPU, GPU, FPS, drops, mean latency, and p95 latency.

Keep this runner dedicated and preferably ephemeral or reset between jobs. It
must not hold deploy keys, release credentials, or unrelated private data: the
manual workflow executes repository code directly on physical hardware and
must never be changed to run automatically for untrusted pull requests.

Archive the benchmark together with `nvpmodel -q`, `jetson_clocks --show`,
JetPack/L4T version, sensor, cooling state, model hash, and TensorRT version.

## Release gate

- Confirm `GpuCamera` still requires `experimental=True`.
- Update the tested hardware matrix without promoting untested sensors.
- Review package version consistency and the release notes in the pull request.
- Build wheel and source distribution from a clean checkout.
- Inspect wheel contents for native sources, browser assets, and public modules.
- Confirm CodeQL, dependency review, Ruff security rules, and the dependency
  audit pass with no known vulnerability.
- Confirm the release build emits a CycloneDX SBOM and every third-party GitHub
  Action remains pinned to a full commit SHA.
- Build native CUDA interop with hardening enabled and verify `GNU_RELRO`,
  `BIND_NOW`, and a non-executable stack; exercise the opt-in ASan/UBSan build.
- Create a signed tag only after both CI workflows pass.
- Never publish or share cached TensorRT `.engine` files as portable assets.
