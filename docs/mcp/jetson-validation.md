# Local MCP validation on Jetson

The November 25, 2026 run passed the MCP-001 runtime checks and the MCP-014
physical smoke and performance checks on the following configuration.

| Item | Tested configuration |
| --- | --- |
| Source | Clean checkout at `0a22d91a6f5f36e2d1264859e0dbd4cc7e3335cb`. Wheel and source distribution built from that checkout identify as `0.8.3`. |
| Device | Jetson Orin Nano Super, aarch64, JetPack 6.2.3 / L4T 36.5.2, IMX219-77 at Argus sensor ID 0. |
| Runtime | JetPack CPython 3.10.12 with system packages, uv 0.11.14, GStreamer 1.20.3, PyGObject 3.42.1, OpenCV 4.5.4 with GStreamer, CUDA toolkit 12.6.68, TensorRT 10.3.0. |
| Power and clocks | 25 W mode. Power and clock settings were left as found. Trial logs contain CPU and GPU clocks and thermal readings from sysfs. `jetson_clocks --show` required privileges unavailable to the test process. |

## Runtime, protocol and camera behavior

- Clean wheel and source-distribution installs verified base imports and CLI
  help before adding the optional runtime. The base wheel reported a missing
  SDK before camera acquisition. With the `mcp` extra, both distributions
  installed `mcp` and `mcp-types` at 2.3.0. All 14 synthetic protocol cases
  passed on Jetson, covering observer grants, typed errors, wire limits,
  revision retry, cancellation, EOF and relaunch.
- The packaged CLI passed both physical CPU smoke cases. Python MCP 2.3.0 and
  Go MCP SDK 1.8.0 each negotiated `2026-07-28`, listed the three tools with
  snapshots enabled, and decoded a 1280×720 JPEG with matching metadata. Go
  also passed the independent synthetic 1×1 JPEG probe. See the
  [transport ADR](../adr/ADR-MCP-004-local-sdk-and-transport.md#synthetic-camera-host-results)
  for workstation results and the TypeScript transport limitation.
- CPU and NVMM capture at 720p and 1080p, native CUDA interop and TensorRT/ONNX
  parity passed. The native build verified `GNU_RELRO`, `BIND_NOW` and a
  non-executable stack. Parity also passed with SDK 2.3.0 and with the isolated
  SDK 1.30.0 comparison. The spike decoded physical 1280×720 JPEGs with
  SDK 2.3.0 on both tested revisions and with SDK 1.30.0 on `2025-11-25`.
  [Runtime decision](../adr/ADR-MCP-001-python-sdk-runtime.md)
- The embedded fixture verified `ready` after publication, stopped state,
  publication after restart, and `BGR_CPU` and `NV12_NVMM` with preview on and
  off. With cached Argus evidence, it reported the attached sensor's five
  modes and declared exposure and gain support with unknown numeric bounds.
  The standalone launcher leaves sensor declarations unknown.
- The ownership checks recorded one Argus open and close per capture run
  and continued publication after closing the borrowed host. Two requests
  overlapped on one stdio pipe. The fixture simulated a pending snapshot on
  live capture to verify cancellation and disconnect. Observed GPU leases in
  the checks were released. [Ownership ADR](../adr/ADR-MCP-002-camera-ownership.md)
- On CPU, a same-publication probe matched the returned server-monotonic age
  of 25.20 ms to the acquisition timestamp. The 1280×720, quality-65 image was
  93,869 bytes, and the response was 125,964 UTF-8 bytes before framing. The
  probe found no EXIF and verified the fixed-quality rejection. The built-in
  GPU branch served HTTP JPEGs and returned `timing_unavailable` for MCP
  snapshots. [Snapshot ADR](../adr/ADR-MCP-006-bounded-jpeg-snapshots.md)

## Matched performance trials

Argus mode 2 captured 1920×1080 at 30 FPS with 720p or 1080p output. Trials
used 5 s of warmup and 30 s of measurement. Both sides of each pair used the
same capture and preview settings, a raw observer, and an open private stdio
pipe. The loaded side added five status calls per second and, on profiles
with JPEG preview, one snapshot attempt per second. The regression compares
successful capture reads per second. Raw-observer latency measures local
frame handoff, and tool latency measures the client round trip.

| Suite | Valid pairs within the provisional 5% target | Largest capture FPS drop |
| --- | ---: | ---: |
| CPU/GPU capture and JPEG preview | 24/24 | 0.157% |
| GPU capture with application-owned TensorRT worker and status calls | 6/6 | 0.110% |

One pair that overlapped a Go build was excluded and replaced. All four
bounded overload trials returned `rate_limited`, continued capture and
recovered normal tool calls. At JPEG quality 65, CPU snapshots succeeded in
90 of 90 attempts at 720p and 88 of 90 at 1080p. The other two exceeded a
payload budget. GPU snapshot attempts returned the expected timing rejection.
The inference results cover the tested GPU worker and model.

## Remaining release checks

- The standard production-preview test requests a dynamic TensorRT shape
  profile. The available fixed 640×640 ONNX model failed engine preparation
  under that profile. An adapted fixed-profile run passed concurrent H.264
  preview, CUDA overlay, and TensorRT using x264. Run the standard workflow
  with a compatible model for the final release candidate.
- A physical WebRTC fixture passed late-join decode, SDP fmtp and RTP checks
  using native GObject ownership for loopback ICE. A fixture run using
  PyGObject for that step exited with `SIGSEGV`. Reproduce the successful
  path in the release validation and retain the failure in the evidence.
- The core capture benchmark completed, and its metrics were extracted from
  stdout. Native Argus diagnostics shared the stream with `--json` output.
  Verify that the release workflow archives a standalone JSON document.
- The ASan/UBSan native build passed. Instrumented runtime execution and a
  long-duration RSS soak were outside this run. Desktop MCP UI rendering
  and Jetson Python 3.11–3.12 were also outside the tested configuration.

Use the [release checklist](../RELEASE_CHECKLIST.md#mcp-release-gates) and the
[physical SDK procedure](../adr/ADR-MCP-001-python-sdk-runtime.md#validate-on-jetson-orin-nano-super)
to reproduce these checks. Keep the dated results tied to their source
revision when recording evidence for the final release candidate.
