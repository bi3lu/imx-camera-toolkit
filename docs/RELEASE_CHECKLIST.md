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

- The local MCP SDK decision is accepted for the tested Jetson configuration.
  Verify the final release candidate against the recorded baseline and
  collect evidence for the outstanding Jetson checks.
  [SDK decision](adr/ADR-MCP-001-python-sdk-runtime.md) ·
  [Physical baseline](mcp/jetson-validation.md#remaining-release-checks)
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

The [November 25, 2026 Jetson run](mcp/jetson-validation.md) passed the
packaged CPU smoke, SDK and native coexistence, local protocol and client
checks, CPU/GPU capabilities, shared ownership, JPEG timing and matched
performance trials at commit `0a22d91`. See the evidence page for the
remaining production workflow checks and the limits of the tested profile.

When validating the final release candidate, use the following checks to
compare against that baseline.

- Run the packaged CLI smoke with the `mcp` extra and JetPack system packages
  available. The test starts the observer and snapshot launches in sequence
  and emits protocol, status, capability and JPEG metadata as JSON.

  ```bash
  IMX_CAMERA_SENSOR=IMX219 IMX_CAMERA_SENSOR_ID=0 \
    .venv/bin/python -m pytest -m hardware -s --no-cov \
    tests/hardware/test_mcp_jetson.py
  ```

- Compare `camera_capabilities` with the installed sensor, driver mode
  descriptors and read-only Argus inspection. Verify the active CPU and GPU
  frame formats after publication with preview off and on. Record operating
  resolution, FPS, observed control support and JPEG timing support.
  [MCP capability contract](mcp/README.md#camera-capabilities)
- Run the selected SDK and stdio profile checks in
  [ADR-MCP-004](adr/ADR-MCP-004-local-sdk-and-transport.md#workstation-interoperability-evidence)
  against the packaged host. Record supported client and protocol versions,
  JPEG decoding and metadata, unsupported revision handling, cancellation,
  clean stdout, EOF and relaunch. Verify partial startup cleanup, the 64 KiB
  input limit and eight outstanding request limit, and shutdown with a
  blocked output pipe. Repeat the
  [independent Go client probe](adr/ADR-MCP-004-local-sdk-and-transport.md#synthetic-camera-host-results)
  and record the negotiated revision and decoded JPEG dimensions.
  [Stdio transport tests](../tests/integration/test_mcp_stdio.py)
- Verify the wheel and source distribution in clean Python 3.10–3.12
  environments. With the base install, check core imports, MCP help and the
  missing-runtime message. With the `mcp` extra, check SDK and types versions,
  invalid launch arguments and a synthetic stdio exchange through the CLI.
  [Local installation and launch](mcp/README.md#local-stdio-launch)
- Ask a reader who did not write the [local observer guide](MCP_GUIDE.md) to
  run both example commands on a working Jetson with dependencies already
  installed. Record the elapsed time from opening the guide to verifying
  ready status and opening the saved JPEG, along with any unclear steps.
  The target is 15 minutes or less.
- Verify tool registration against the scope names and admin
  behavior in [ADR-MCP-003](adr/ADR-MCP-003-threat-model-and-access-policy.md).
  Record protocol tests for observer defaults, explicit snapshot opt-in,
  direct-call denials, disabled tools and the selected local transport boundary.
  Confirm the local rate and concurrency limits, slot release after
  cancellation or deadline, and the inclusive UTF-8 response limit.
  [Camera tool protocol tests](../tests/integration/test_mcp_camera_tools.py)
- Verify that the packaged tools return the documented domain
  errors for invalid arguments, exhausted tool budgets, expired deadlines and
  unexpected failures. Verify that diagnostics stay private and malformed
  requests use JSON-RPC errors. [MCP error contract](mcp/README.md#errors-and-mcp-representation)
- Confirm that `camera_status` reports the configured CPU or GPU path and
  reaches `ready` after that run's first raw frame. Check the stopped state
  and confirm a restart requires a new publication.
- Overlap two calls on one stdio connection and record one Argus session
  and one standalone shutdown. Verify a borrowed camera continues
  publishing after host shutdown.
  [Camera ownership smoke checks](adr/ADR-MCP-002-camera-ownership.md#verification)
- Decode a CPU MCP snapshot and confirm acquisition age, fixed-source
  constraints and matching metadata. Verify GPU HTTP JPEG delivery and the
  built-in GPU MCP snapshot rejection `timing_unavailable`.
  [ADR-MCP-006](adr/ADR-MCP-006-bounded-jpeg-snapshots.md#verification)
- Compare the matched capture and preview profiles with the recorded
  [performance baseline](mcp/jetson-validation.md#matched-performance-trials).
- Before releasing inference tools, verify separate inference grants and the
  response, provenance and payload tests from ADR-MCP-003.
- Before enabling the remote profile, record MCP-027 review and passing
  evidence for issuer and audience validation, scope enforcement, trusted
  proxy and origin policy, request limits, redaction and audit retention.
  Include control and preview authorization tests for any such tools shipped.
