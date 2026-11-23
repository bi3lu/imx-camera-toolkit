# ADR-MCP-001 — Python SDK runtime on Jetson

- **MCP-001 status** DONE/BLOCKED
- **Runtime decision** Proposed, pending physical Jetson validation
- **Workstation evidence** 2026-11-23

The workstation spike is complete. Development can proceed on `develop` with
SDK v2 as the working candidate. Before the v0.9.0 release, copy the repository
to the physical Jetson Orin Nano Super and run the validation below to accept
the runtime decision.

## Proposed runtime

Test MCP SDK `2.3.0` in the toolkit process using the JetPack system interpreter
first. The published SDK and its `mcp-types` companion both declare Python
`>=3.10`, so this candidate can use the existing Python 3.10 JetPack bindings if
the hardware smoke passes. Keep SDK `1.30.0` as the maintenance-line comparison.
Both pins belong to the experiment. The toolkit currently has zero required
dependencies and supports Python 3.10–3.12.
[MCP SDK 2.3.0](https://pypi.org/project/mcp/2.3.0/) ·
[mcp-types 2.3.0](https://pypi.org/project/mcp-types/2.3.0/) ·
[MCP SDK 1.30.0](https://pypi.org/project/mcp/1.30.0/)

| Candidate | Published support | Evidence |
| --- | --- | --- |
| SDK 2.3.0 in-process on Python 3.10–3.12 | Current stable line. Implements the 2026-07-28 discovery protocol and the 2025-11-25 handshake. | Passes both revisions on macOS with a synthetic JPEG. Pending JetPack bindings, physical capture and native interop. |
| SDK 1.30.0 in-process on Python 3.10–3.12 | Maintenance line for critical and security fixes. Latest advertised revision is 2025-11-25. | Passes the 2025-11-25 handshake on macOS with a synthetic JPEG. The v2 client's automatic discovery probe also falls back successfully, with a validation diagnostic on the v1 server. Pending the same Jetson checks. |
| SDK v2 in a sidecar | Separate interpreter and process. Requires an authenticated, bounded local camera IPC contract. | Deferred. Revisit if Jetson evidence shows a dependency or process-isolation requirement. Measure IPC round-trip latency and validate peer access and deadlines if this path is prototyped. |

The minimum-Python assumption in the RFC predates these released packages.
Python 3.11 and 3.12 host installs establish Python package compatibility.
Any proposed Jetson interpreter must separately pass the binding checks on
the target image. MCP-004 can use the two releases above as protocol candidates
while this decision is pending.
[Upstream releases](https://github.com/modelcontextprotocol/python-sdk/releases) ·
[SDK protocol versions](https://py.sdk.modelcontextprotocol.io/protocol-versions/) ·
[Upstream support policy](https://github.com/modelcontextprotocol/python-sdk/security)

## Workstation evidence

The experiment serves one application-owned camera over stdio and uses the
official v2 client for both server versions. The client checks discovery,
`tools/list`, `camera_status` and one image response. It checks the returned
bytes against a 619-byte synthetic JPEG and emits only metadata. The final
check on Jetson decodes the received JPEG in memory.

| macOS 27.2.1, ARM64 | Core-only install and imports | v1 server | v2 server |
| --- | --- | --- | --- |
| CPython 3.10.20 | Pass | 2025-11-25 pass | 2026-07-28 and 2025-11-25 pass |
| CPython 3.11.13 | Pass | 2025-11-25 pass | 2026-07-28 and 2025-11-25 pass |
| CPython 3.12.0 | Pass | 2025-11-25 pass | 2026-07-28 and 2025-11-25 pass |

- `uv 0.12.20` resolved the two mutually exclusive SDK groups alongside the
  toolkit's preview extra. The root and experiment lock checks passed. All
  six separate SDK installs passed `uv pip check` and the core imports.
- The Python 3.10 wheel-only resolver dry run for `aarch64-manylinux_2_35`
  succeeded for both groups. JetPack supplies its own GStreamer, PyGObject,
  OpenCV, TensorRT and CUDA components.
- The upstream advisory list and the two `uv audit --locked --only-group`
  checks reported no known vulnerabilities affecting the tested pins or
  audited dependency sets at the time of the spike.
  [Upstream advisories](https://github.com/modelcontextprotocol/python-sdk/security/advisories)
- Ruff, Black and mypy passed. The host suite reported 250 passed, 4 skipped
  and 2 deselected. The skips covered TensorRT parity and the three GStreamer
  integration tests. Physical CSI capture and real CUDA interop are pending.

## Reproduce on a workstation

Run from the repository root on a POSIX workstation with Python 3.10, 3.11 and
3.12 available.

```bash
spike_dir=examples/mcp_runtime_spike
spike_envs=$(mktemp -d)
uv lock --check
uv lock --project "$spike_dir" --check

for py in 3.10 3.11 3.12; do
  UV_PROJECT_ENVIRONMENT="$spike_envs/$py-core" \
    uv sync --locked --no-default-groups --python "$py"
  "$spike_envs/$py-core/bin/python" -c '
import importlib.util
from imx_camera_toolkit import Camera, GpuCamera
from imx_camera_toolkit.inference import TensorRTRunner
assert importlib.util.find_spec("mcp") is None'
  for sdk in v1 v2; do
    UV_PROJECT_ENVIRONMENT="$spike_envs/$py-$sdk" \
      uv sync --project "$spike_dir" --locked --python "$py" --group "sdk-$sdk"
    uv pip check --python "$spike_envs/$py-$sdk/bin/python"
    "$spike_envs/$py-$sdk/bin/python" -c '
from importlib.metadata import version
from imx_camera_toolkit import Camera, GpuCamera
from imx_camera_toolkit.inference import TensorRTRunner
print(version("mcp"))'
  done
  "$spike_envs/$py-v2/bin/python" "$spike_dir/client.py" \
    --server-python "$spike_envs/$py-v1/bin/python" --legacy
  "$spike_envs/$py-v2/bin/python" "$spike_dir/client.py"
  "$spike_envs/$py-v2/bin/python" "$spike_dir/client.py" --legacy
done

for sdk in v1 v2; do
  uv audit --project "$spike_dir" --locked --only-group "sdk-$sdk"
  uv export --project "$spike_dir" --locked --group "sdk-$sdk" \
    --no-emit-project --no-emit-package imx-camera-toolkit \
    --output-file "$spike_envs/$sdk-requirements.txt"
  uv pip install --dry-run --target "$spike_envs/$sdk-linux-arm64" \
    --python-version 3.10 --python-platform aarch64-manylinux_2_35 \
    --only-binary :all: -r "$spike_envs/$sdk-requirements.txt"
done
```

## Validate on Jetson Orin Nano Super

Run on JetPack 6.2.3 from the repository root with an IMX219-77
available as Argus sensor 0. The existing TensorRT parity test uses sensor 0
and generates its own ONNX fixture. Record the exact sensor, L4T, interpreter,
camera settings, power mode, JetPack package versions and outcomes. Resolve
any failed system imports in the JetPack environment before testing an SDK.
[Jetson prerequisites](../GPU_CAMERA_YOLO_GUIDE.md) ·
[Jetson hardware gate](../RELEASE_CHECKLIST.md)

```bash
spike_dir=examples/mcp_runtime_spike
spike_envs=$(mktemp -d)
jetpack_python=/usr/bin/python3
uname -sm
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack nvidia-l4t-core python3-gi \
  python3-opencv python3-libnvinfer
nvpmodel -q
"$jetpack_python" -VV
gst-inspect-1.0 nvarguscamerasrc nvvidconv appsink
"$jetpack_python" -c '
import cv2, gi, tensorrt
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
print(Gst.version_string(), gi.__version__, cv2.__version__, tensorrt.__version__)
print(next(line.strip() for line in cv2.getBuildInformation().splitlines()
           if "GStreamer:" in line))'

export IMX_CAMERA_SENSOR=IMX219 IMX_CAMERA_SENSOR_ID=0 IMX_TENSORRT_INTEGRATION=1
UV_PROJECT_ENVIRONMENT="$spike_envs/baseline" \
  uv venv --python "$jetpack_python" --system-site-packages "$spike_envs/baseline"
UV_PROJECT_ENVIRONMENT="$spike_envs/baseline" \
  uv sync --locked --python "$jetpack_python" --extra production-preview \
    --extra tensorrt-test --group dev
"$spike_envs/baseline/bin/imx-camera" diagnose --hardware --json
"$spike_envs/baseline/bin/imx-camera" test --backend cpu --sensor-id 0 --json
"$spike_envs/baseline/bin/imx-camera-build-interop" --cuda-architecture 87
"$spike_envs/baseline/bin/python" -m pytest -m hardware \
  tests/hardware/test_gpu_capture.py tests/integration/test_tensorrt_onnx_parity.py

for sdk in v1 v2; do
  uv venv --python "$jetpack_python" --system-site-packages "$spike_envs/$sdk"
  UV_PROJECT_ENVIRONMENT="$spike_envs/$sdk" \
    uv sync --project "$spike_dir" --locked --python "$jetpack_python" --group "sdk-$sdk"
  "$spike_envs/$sdk/bin/imx-camera" test --backend gpu --sensor-id 0 --json
done
"$spike_envs/v2/bin/python" "$spike_dir/client.py" \
  --server-python "$spike_envs/v1/bin/python" --jetson --legacy
"$spike_envs/v2/bin/python" "$spike_dir/client.py" --jetson
"$spike_envs/v2/bin/python" "$spike_dir/client.py" --jetson --legacy

for sdk_pin in mcp==1.30.0 mcp==2.3.0; do
  UV_PROJECT_ENVIRONMENT="$spike_envs/baseline" \
    uv run --frozen --extra production-preview --extra tensorrt-test --group dev \
    --with "$sdk_pin" python -m pytest -m hardware \
    tests/hardware/test_gpu_capture.py tests/integration/test_tensorrt_onnx_parity.py
done
```

Record the observed protocol, backend, JPEG byte count, checks that passed or
were skipped, and the first failure for each candidate. Record any alternate
Python interpreter and the source of each JetPack binding if it is evaluated.
Accept the ADR after selecting one runtime with passing physical evidence.
If every candidate fails, leave the MVP runtime decision open and document
the concrete incompatibilities.
