# GPU release checklist

Use this checklist before tagging a release that contains the experimental GPU
path. A host-only green workflow is not sufficient evidence for Jetson support.

## Host gate

```bash
uv lock --check
uv run ruff check .
uv run black --check .
uv run mypy imx_camera_toolkit tests
uv run pytest tests/unit tests/integration -m "not hardware and not benchmark" \
  --cov=imx_camera_toolkit/_internal --cov-report=term-missing \
  --cov-fail-under=68
uv audit --frozen
uv build
uv export --all-extras --format cyclonedx1.5 --frozen \
  --output-file dist/sbom.cdx.json
```

## Jetson gate

Run the `Jetson hardware validation` GitHub Actions workflow on a self-hosted
runner labeled `jetson`. Manual runs select the connected IMX219 or IMX477 and
provide a local ONNX model path. The weekly trusted-branch run reads
`JETSON_SENSOR`, `JETSON_SENSOR_ID`, `JETSON_ONNX_MODEL_PATH`, and
`JETSON_BENCHMARK_SECONDS` repository variables. The workflow must verify:

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
workflow executes repository code directly on physical hardware and must never
run automatically for untrusted pull requests. The scheduled run executes only
the repository default branch.

Archive the benchmark together with `nvpmodel -q`, `jetson_clocks --show`,
JetPack/L4T version, sensor, cooling state, model hash, and TensorRT version.

## Release gate

- Confirm `GpuCamera` still requires `experimental=True`.
- Update the tested hardware matrix without promoting untested sensors.
- Review package version consistency and the release notes in the pull request.
- Build wheel and source distribution from a clean checkout.
- Inspect wheel contents for native sources, browser assets, public modules,
  and the absence of the legacy top-level `packages` namespace.
- Configure the `pypi` GitHub environment and the matching PyPI Trusted
  Publisher before publishing the first release. Do not add a long-lived PyPI
  token.
- Confirm Python CodeQL, Ruff security rules, and the dependency audit pass
  with no known vulnerability. Re-enable dependency review only after the
  repository dependency graph is enabled.
- Confirm the release build emits a CycloneDX SBOM and every third-party GitHub
  Action remains pinned to a full commit SHA.
- Build native CUDA interop with hardening enabled and verify `GNU_RELRO`,
  `BIND_NOW`, and a non-executable stack; exercise the opt-in ASan/UBSan build.
- Create a signed tag only after both CI workflows pass.
- Never publish or share cached TensorRT `.engine` files as portable assets.
