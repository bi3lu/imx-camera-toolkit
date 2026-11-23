# CPU/GPU release checklist

Use this checklist before tagging a release that changes either stable capture
path. A host-only green workflow is not sufficient evidence for Jetson support.

## Host gate

```bash
uv lock --check
uv run ruff check .
uv run black --check .
uv run mypy imx_camera_toolkit tests
uv run pytest tests/unit tests/integration -m "not hardware and not benchmark" \
  --cov=imx_camera_toolkit/_internal --cov-report=term-missing \
  --cov-fail-under=65.5
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

- JetPack 6.2.3 / Jetson Linux 36.5.2 is installed on the runner;
- NVMM capture at 1280x720 and 1920x1080 at 30 FPS;
- simultaneous TensorRT and H.264 production preview using the resolved
  NVENC/x264 backend, with engine preparation on the main thread, inference on
  its consumer thread, and CUDA overlay rendering on the GStreamer thread;
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

- Before tagging v0.9.0, copy the release candidate checkout to a physical
  Jetson Orin Nano Super with JetPack 6.2.3 and IMX219-77. Run the MCP smoke
  and native interop checks, record the results, and accept the runtime
  decision in [ADR-MCP-001](adr/ADR-MCP-001-python-sdk-runtime.md#validate-on-jetson-orin-nano-super).
- Confirm `GpuCamera()` remains usable without a feature flag and reports the
  stable NV12/NVMM contract.
- Confirm `preview`, `snapshot`, and `test` accept both `--backend cpu` and
  `--backend gpu`.
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

## MCP release gates

- Before v0.9.0, run the selected SDK and stdio profile checks in
  [ADR-MCP-004](adr/ADR-MCP-004-local-sdk-and-transport.md#workstation-interoperability-evidence)
  against the packaged host. Record supported client and protocol versions,
  JPEG decoding and metadata, unsupported revision handling, cancellation,
  clean stdout, EOF and relaunch. Record the independent client outcome.
- Before v0.9.0, verify tool registration against the scope names and admin
  behavior in [ADR-MCP-003](adr/ADR-MCP-003-threat-model-and-access-policy.md).
  Record protocol tests for observer defaults, explicit snapshot opt-in,
  direct-call denials, disabled tools and the selected local transport boundary.
- On the physical Jetson before v0.9.0, confirm that `camera_status` reports
  the configured CPU or GPU path and reaches `ready` after that run's first
  raw frame, including with JPEG preview disabled. Check the stopped state
  and confirm a restart requires a new publication.
- Before releasing inference tools, verify separate inference grants and the
  response, provenance and payload tests from ADR-MCP-003.
- Before enabling the remote profile, record MCP-027 review and passing
  evidence for issuer and audience validation, scope enforcement, trusted
  proxy and origin policy, request limits, redaction and audit retention.
  Include control and preview authorization tests for any such tools shipped.
