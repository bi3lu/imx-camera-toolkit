# MCP ADR — Python SDK runtime on Jetson

- **Runtime decision** Accepted for the local MCP profile
- **Workstation evidence** 2026-11-23
- **Physical Jetson evidence** 2026-11-25

Use MCP SDK 2.3.0 in the toolkit process with JetPack system Python and access
to the system packages. The physical check passed on Orin Nano Super with
IMX219-77, JetPack 6.2.3 / L4T 36.5.2 and Python 3.10.12.
[Jetson validation evidence](../mcp/jetson-validation.md)

## Runtime decision

The optional `mcp` extra selects SDK `2.3.0` and its matching `mcp-types`
package. Both declare Python `>=3.10`. The experiment retains SDK `1.30.0`
for comparison. The toolkit has zero required dependencies and supports
Python 3.10–3.12.
[Local SDK and transport ADR](ADR-mcp-local-sdk-and-transport.md) ·
[MCP SDK 2.3.0](https://pypi.org/project/mcp/2.3.0/) ·
[mcp-types 2.3.0](https://pypi.org/project/mcp-types/2.3.0/) ·
[MCP SDK 1.30.0](https://pypi.org/project/mcp/1.30.0/)

| Candidate | Published support | Evidence |
| --- | --- | --- |
| SDK 2.3.0 in-process on Python 3.10–3.12 | Implements the 2026-07-28 discovery protocol and the 2025-11-25 handshake. | Passed both revisions with a synthetic JPEG on macOS and a physical JPEG in the Jetson spike. The installed wheel passed physical CPU smoke, NVMM capture and TensorRT/ONNX parity on JetPack Python 3.10.12. |
| SDK 1.30.0 in-process on Python 3.10–3.12 | Maintenance-line comparison with the 2025-11-25 handshake. | Passed the legacy handshake with a synthetic JPEG on macOS and a physical JPEG on Jetson. The isolated Jetson environment passed NVMM capture and TensorRT/ONNX parity. |
| SDK v2 in a sidecar | Separate interpreter and process. Requires an authenticated, bounded local camera IPC contract. | Deferred. Revisit if Jetson evidence shows a dependency or process-isolation requirement. Measure IPC round-trip latency and validate peer access and deadlines if this path is prototyped. |

The minimum-Python assumption in the RFC predates these released packages.
Python 3.11 and 3.12 host installs establish Python package compatibility.
The physical result applies to the tested JetPack system Python 3.10.12.
Evaluate any alternate interpreter against the bindings on its target image.
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
  integration tests. The physical results are recorded in the
  [Jetson validation evidence](../mcp/jetson-validation.md).

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
    uv sync --project "$spike_dir" --frozen --python "$jetpack_python" --group "sdk-$sdk"
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

The physical run used the archived spike lock with `--frozen` after `--locked`
requested a lock update. When repeating the comparison, record negotiated
protocols, backend, JPEG byte counts, skips and the first failure for each
candidate. Record the source of any alternate interpreter's JetPack bindings.
