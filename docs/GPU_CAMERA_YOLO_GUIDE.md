# GPU Camera and YOLO deployment guide

This guide brings an NVIDIA Jetson Orin Nano running JetPack 6.2.2 from a fresh
repository checkout to GPU-first YOLO inference with a low-latency WebRTC
preview. It also describes a fail-closed field deployment with scoped
authentication and TLS.

The runnable reference is
[`examples/yolo_detection.py`](../examples/yolo_detection.py). It deliberately
keeps YOLO decoding in application code: the toolkit remains model-neutral and
only owns capture, CUDA preprocessing, TensorRT execution, latest-frame
scheduling, overlays, and transport.

## Resulting data path

```text
IMX CSI sensor
  -> nvarguscamerasrc
  -> NV12 in NVMM
  -> leaky latest-frame branch
  -> CUDA NV12-to-NCHW preprocessing
  -> TensorRT YOLO engine
  -> application-owned Nx6 decoder
  -> CUDA rectangle overlay
  -> shared H.264 encoder
  -> WebRTC browser client
```

Camera pixels do not become a BGR or NumPy host image in this path. A
`GpuFrame` is a borrowed, short-lived lease over an NVMM buffer. The
`InferenceConsumer` retains one lease while processing it and replaces an
unread input when inference is slower than capture. This bounds latency and
memory instead of accumulating stale frames.

Jetson Orin Nano has no hardware NVENC block. The toolkit therefore keeps
capture, inference, and overlay in NVMM, then uses one shared x264 fallback
branch for H.264 preview. This affects preview CPU usage; it does not move the
YOLO input path to the CPU.

## 1. Confirm the target

The tested baseline is Jetson Orin Nano, JetPack 6.2.2 / Jetson Linux 36.5,
CUDA 12.6, TensorRT 10.3, and Python 3.10. Run these commands on the Jetson over
SSH:

```bash
cat /etc/nv_tegra_release
nvcc --version
python3 --version
python3 -c "import tensorrt; print(tensorrt.__version__)"
nvpmodel -q
jetson_clocks --show
```

Do not install CUDA, TensorRT, PyGObject, or OpenCV wheels from PyPI over the
JetPack packages. Native ABI compatibility is part of the camera and inference
contract.

Confirm that the sensor is visible before changing the application:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=60 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
  fakesink sync=false
```

If Argus reports that the camera provider is unavailable, stop other camera
processes and restart its daemon before retrying:

```bash
sudo systemctl restart nvargus-daemon
```

## 2. Install system prerequisites

Install the build headers and GStreamer elements used by CUDA interop and the
WebRTC/x264 preview:

```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-gi python-gi-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  gstreamer1.0-nice gstreamer1.0-libav cmake ninja-build v4l-utils
```

Verify every required runtime element. `gst-inspect-1.0` exits nonzero and
names the missing plugin when the image is incomplete:

```bash
gst-inspect-1.0 nvarguscamerasrc nvvidconv nvjpegenc x264enc h264parse \
  rtph264pay webrtcbin nicesrc nicesink appsink
test -f /usr/src/jetson_multimedia_api/include/nvbufsurface.h
```

## 3. Create the project environment

From the repository root, create an environment that can import JetPack's
system packages. The production preview and native TensorRT build extras are
both required by the YOLO example:

```bash
uv venv --system-site-packages --allow-existing .venv
uv sync --extra production-preview --extra tensorrt-build
uv run python -c "import cv2, gi, tensorrt; print(cv2.__version__, tensorrt.__version__)"
```

Build the repository's CUDA/NvBufSurface bridge for Orin compute capability
8.7:

```bash
uv run imx-camera-build-interop --cuda-architecture 87
uv run python -c \
  "from imx_camera_toolkit._internal.inference.interop import NativeCudaInterop; NativeCudaInterop(); print('CUDA interop: OK')"
```

The extension is built against the currently installed JetPack headers. Rebuild
it after changing the JetPack image, Python minor version, or repository native
sources.

## 4. Validate capture independently

Do not debug the model and sensor at the same time. First run diagnostics and a
bounded GPU capture test:

```bash
uv run imx-camera diagnose --hardware
uv run imx-camera info --hardware
uv run imx-camera test --backend gpu --sensor-id 0 \
  --width 1280 --height 720 --fps 30 --frames 60
```

Then measure the target configuration:

```bash
uv run imx-camera benchmark jetson --backend gpu --resolution 720p \
  --sensor-id 0 --frames 300 --fps 30 --json
```

For IMX219, 1280x720 commonly maps to Argus sensor mode 4 and 1920x1080 to
mode 2. Treat those values as sensor/driver-specific, not universal defaults.
Use `--sensor-mode` only after `imx-camera info --hardware` or Argus output has
confirmed the mode on the actual device.

## 5. Export a compatible YOLO model

Export the ONNX model on a development workstation, not on the deployed
Jetson. The example expects one end-to-end output tensor whose rows are:

```text
[x1, y1, x2, y2, confidence, class_id]
```

Coordinates must refer to the letterboxed model input. The example uses the
runner's exact resize scale and padding to map them back to camera pixels.
Non-max suppression must already be present in the graph. If a model emits raw
YOLO heads, add an application decoder/NMS stage instead of treating them as
Nx6 rows.

One Ultralytics export starting point is:

```bash
uvx --from ultralytics yolo export model=yolo11n.pt format=onnx \
  imgsz=640 dynamic=True simplify=True nms=True opset=17
```

Export options and output names vary between model families and exporter
versions. Inspect the artifact rather than assuming the example's default
output name `boxes`:

```bash
python - <<'PY'
import onnx

model = onnx.load("yolo11n.onnx")
print("inputs:", [value.name for value in model.graph.input])
print("outputs:", [value.name for value in model.graph.output])
onnx.checker.check_model(model)
PY
```

Copy only the ONNX model to the Jetson. TensorRT engines are executable,
target-local cache artifacts and must not be copied between JetPack,
TensorRT, GPU, model, or shape-profile combinations.

```bash
scp yolo11n.onnx jetson:/opt/imx-camera/models/yolo11n.onnx
```

For field deployments, use the signed-model support described in
[GPU inference integration](../imx_camera_toolkit/_internal/inference/README.md#engine-cache-safety)
and provision the public trust anchor independently from the model payload.

## 6. Run local GPU-first YOLO

Start with loopback-only HTTP and an SSH tunnel. Replace `output0` with the
name printed in the previous step:

```bash
uv run python examples/yolo_detection.py \
  /opt/imx-camera/models/yolo11n.onnx \
  --output output0 --score 0.50 \
  --sensor-id 0 --width 1280 --height 720 --fps 30
```

From the workstation, forward the loopback port and open
`http://127.0.0.1:8000/`:

```bash
ssh -L 8000:127.0.0.1:8000 jetson
```

The first launch can take several minutes because TensorRT builds an FP16
engine before opening the camera. Later launches reuse it only when the ONNX
hash, TensorRT version, compute capability, precision, input name, and complete
shape profile all match.

The example exposes model-neutral health at `/debug/health`. Check these
fields during commissioning:

- `capture.capture_fps` and capture drop counters;
- `components.inference.processed_frames`, `failed_frames`, and
  `latest_inference_time_ns`;
- `components.overlay.healthy` and its last error;
- `encode_fps`, encoder backend, active clients, RTP counters, and WebRTC
  connection state.

## 7. Run in field mode

Field mode enables scoped bearer authentication, disables OpenAPI endpoints,
validates `Host`, limits request size and rate, adds browser hardening headers,
and restricts detailed diagnostics to an `admin` grant. Use separate random
tokens for preview and administration.

Generate tokens in an interactive shell and store only their SHA-256 digests.
Keep the plaintext values in the device's secret-management system and give
the preview token to authorized operators. The following commissioning example
creates a process-owned `0600` file for the current unprivileged user:

```bash
umask 077
read -rsp 'Preview token: ' STREAM_TOKEN; echo
read -rsp 'Admin token: ' ADMIN_TOKEN; echo
STREAM_DIGEST="$(printf %s "$STREAM_TOKEN" | sha256sum | cut -d' ' -f1)"
ADMIN_DIGEST="$(printf %s "$ADMIN_TOKEN" | sha256sum | cut -d' ' -f1)"
printf '{"schema_version":1,"tokens":[{"sha256":"%s","scopes":["stream:read"]},{"sha256":"%s","scopes":["admin"]}]}\n' \
  "$STREAM_DIGEST" "$ADMIN_DIGEST" > /tmp/imx-camera-tokens.json
install -d -m 0700 "$HOME/.config/imx-camera"
install -m 0600 /tmp/imx-camera-tokens.json \
  "$HOME/.config/imx-camera/tokens.json"
rm /tmp/imx-camera-tokens.json
unset STREAM_TOKEN ADMIN_TOKEN STREAM_DIGEST ADMIN_DIGEST
```

The application accepts only root-owned or process-owned regular token files
with mode `0600` or `0640`; it rejects symlinks and looser permissions.

### Direct TLS

For a direct listener, bind to the Jetson interface, provide the exact DNS name
or IP sent in the HTTP `Host` header, and provide both certificate files:

```bash
uv run python examples/yolo_detection.py \
  /opt/imx-camera/models/yolo11n.onnx \
  --output output0 --field-mode \
  --host 0.0.0.0 --allowed-host camera.example.com \
  --token-file "$HOME/.config/imx-camera/tokens.json" \
  --tls-certfile /etc/imx-camera/tls/fullchain.pem \
  --tls-keyfile /etc/imx-camera/tls/private.key
```

Open `https://camera.example.com:8000/`, enter the `stream:read` token in the
public login shell, and let the application exchange it for a session-only,
HttpOnly, SameSite cookie. Do not embed a token in JavaScript, a URL, or a
repository file.

### TLS reverse proxy

The recommended topology terminates TLS or mTLS at a reverse proxy and keeps
Uvicorn on loopback. Preserve the original host and forward the HTTPS scheme:

```bash
uv run python examples/yolo_detection.py \
  /opt/imx-camera/models/yolo11n.onnx \
  --output output0 --field-mode \
  --host 127.0.0.1 --allowed-host camera.example.com \
  --token-file "$HOME/.config/imx-camera/tokens.json" --behind-tls-proxy
```

Example NGINX location:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto https;
    proxy_read_timeout 75s;
}
```

Expose only the proxy port through the firewall. A client outside the local
network may also require `--stun-server` and `--turn-server`; TURN credentials
are deployment secrets and should come from protected configuration rather
than source code.

Verify the public and protected routes:

```bash
curl --fail https://camera.example.com/healthz
curl --fail -H "Authorization: Bearer <admin-token>" \
  https://camera.example.com/debug/health
test "$(curl --silent --output /dev/null --write-out '%{http_code}' \
  https://camera.example.com/openapi.json)" = 404
```

The first two commands should return liveness and authenticated diagnostics.
The final command should return `404` in field mode.

## 8. Operate it as a service

Use a dedicated unprivileged account with access to the camera devices. Do not
run the inference service as root merely to read its root-owned configuration.
The exact supplementary groups depend on the JetPack image; commonly they
include `video`. Create the account once if it is not already managed by the
device provisioning system:

```bash
sudo useradd --system --user-group --create-home \
  --home-dir /var/lib/imx-camera --shell /usr/sbin/nologin imx-camera
sudo usermod --append --groups video imx-camera
```

Provision the field credentials for that account before installing the unit.
This keeps the file root-owned but readable by the service group:

```bash
sudo install -d -o root -g imx-camera -m 0750 /etc/imx-camera
sudo install -o root -g imx-camera -m 0640 \
  "$HOME/.config/imx-camera/tokens.json" /etc/imx-camera/tokens.json
sudo install -d -o imx-camera -g imx-camera -m 0750 \
  /opt/imx-camera-toolkit/.cache
```

Example systemd unit for the reverse-proxy topology:

```ini
[Unit]
Description=IMX GPU Camera YOLO
After=network-online.target nvargus-daemon.service
Wants=network-online.target

[Service]
Type=simple
User=imx-camera
Group=imx-camera
SupplementaryGroups=video
WorkingDirectory=/opt/imx-camera-toolkit
ExecStart=/opt/imx-camera-toolkit/.venv/bin/python examples/yolo_detection.py /opt/imx-camera/models/yolo11n.onnx --output output0 --field-mode --host 127.0.0.1 --allowed-host camera.example.com --token-file /etc/imx-camera/tokens.json --behind-tls-proxy
Restart=on-failure
RestartSec=3
TimeoutStopSec=20
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/imx-camera-toolkit/.cache

[Install]
WantedBy=multi-user.target
```

Review systemd hardening against the device nodes and JetPack libraries present
on the target; add narrowly scoped `DeviceAllow` rules only after confirming
all camera/CUDA devices used by that image.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now imx-camera-yolo.service
sudo systemctl status imx-camera-yolo.service
journalctl -u imx-camera-yolo.service -f
```

Record `nvpmodel -q`, `jetson_clocks --show`, cooling state, model hash, sensor
mode, resolution, FPS, and client count with every performance result. Select a
power mode appropriate to the installed supply and cooling. Locking clocks can
improve repeatability but increases power and thermal load; validate throttling
under the actual enclosure and ambient temperature.

## Troubleshooting

| Symptom | Check | Corrective action |
| --- | --- | --- |
| `nvarguscamerasrc` cannot open | Another process, ribbon orientation, sensor overlay, `nvargus-daemon` logs | Stop competing clients, verify hardware configuration, restart Argus, then rerun the bounded GStreamer test. |
| `No module named cv2`, `gi`, or `tensorrt` | `.venv/pyvenv.cfg` and the import command from step 3 | Recreate the environment with `--system-site-packages`; do not replace JetPack packages with PyPI wheels. |
| CUDA interop build cannot find headers | `python-gi-dev`, GStreamer development packages, `/usr/src/jetson_multimedia_api/include/nvbufsurface.h` | Install the missing JetPack/development package and rebuild the extension. |
| TensorRT rejects ONNX | TensorRT parser log, model opset, input count/type, raw versus end-to-end output | Re-export with a TensorRT-compatible opset and one float32 NCHW image input; keep YOLO decoding application-owned. |
| Output tensor is not found | ONNX output names and `--output` | Pass the actual tensor name. |
| Boxes are scaled incorrectly | Nx6 coordinate convention and letterbox metadata | Ensure detections use model-input `xyxy` coordinates and NMS is already applied. |
| Preview works but inference is slow | Inference health, shape, precision, power/thermal state | Use FP16, choose an appropriate inference shape, inspect throttling, and expect input drops rather than queue growth. |
| WebRTC page loads without video | Browser console, ICE state, RTP counters, firewall, STUN/TURN | Verify GStreamer WebRTC plugins, proxy timeouts, UDP reachability, and ICE configuration. |
| Field mode returns `400 Invalid host header` | Browser host versus `--allowed-host` | Add the exact external DNS name or address; never use `*`. |
| Field mode redirects repeatedly | `X-Forwarded-Proto` and proxy trust boundary | Forward `https`, keep the app on loopback, and do not expose the untrusted proxy port. |
| Token file is rejected | Owner, file type, mode, JSON schema, lowercase digest | Use a regular non-symlink file owned by root/process with `0600` or `0640`. |

## Commissioning checklist

- GPU capture passes for the deployed sensor mode, resolution, and FPS.
- The native interop extension was built on the target JetPack image.
- ONNX input/output names and the Nx6 schema were inspected explicitly.
- TensorRT built a local engine and a restart produced a validated cache hit.
- Inference and overlay health stay clean under the expected client load.
- Capture and inference drops are understood; no unbounded frame queue exists.
- Field mode uses distinct least-privilege tokens and a protected token file.
- TLS or mTLS terminates before any non-loopback network path.
- The exact external host is allowlisted and Swagger/OpenAPI returns `404`.
- WebRTC connectivity was tested from the real operator network, including
  STUN/TURN where required.
- Power, cooling, sustained clocks, and thermal throttling were validated in
  the final enclosure.
- The service stops cleanly and recovers after an intentional restart of the
  application and Argus daemon.

## Related documentation

- [CPU, GPU, and browser mode guide](GPU_PATH_GUIDE.md)
- [GPU inference integration](../imx_camera_toolkit/_internal/inference/README.md)
- [Production browser preview](../imx_camera_toolkit/_internal/production_preview/README.md)
- [Camera capture architecture](../imx_camera_toolkit/_internal/camera/README.md)
- [NVIDIA JetPack 6.2.2 release page](https://developer.nvidia.com/embedded/jetpack-sdk-622)
- [NVIDIA accelerated GStreamer guide](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html)
- [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
