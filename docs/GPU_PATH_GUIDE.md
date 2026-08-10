# CPU, GPU, and browser mode guide

The stable and experimental paths deliberately solve different problems. Do
not choose a camera class from the desired model name; choose it from the
memory domain required by the next consumer.

| Requirement | Capture API | Browser path | Cost and status |
| --- | --- | --- | --- |
| OpenCV, NumPy, CPU processing, snapshots | `Camera` | MJPEG | Stable. NV12 is converted to BGR and copied into host RAM. |
| Simple diagnostics in any desktop browser | `Camera` or GPU JPEG branch | MJPEG | Debug-oriented. Easy to deploy, but JPEG and multipart delivery cost CPU/network bandwidth. |
| TensorRT or custom CUDA consumer | `GpuCamera(experimental=True)` | Optional WebRTC | Experimental. The borrowed NV12 frame remains in NVMM. |
| DeepStream application | `GpuCamera(experimental=True)` or a native DeepStream source | WebRTC/HLS outside capture | Experimental toolkit interop. Prefer a native DeepStream pipeline when DeepStream owns the full graph. |
| Low-latency production browser preview | `GpuCamera(experimental=True)` | H.264 WebRTC | One shared NVENC encoder where available, or shared CPU x264 on Orin Nano. |
| Reverse-proxy-friendly segmented delivery | `GpuCamera(experimental=True)` | H.264/H.265 HLS | Simpler HTTP deployment, with more latency than WebRTC. H.265 requires NVENC. |

`Camera.read(copy=False)` only avoids another Python-side array copy. It does
not turn BGR host memory into CUDA or NVMM memory. Conversely, `GpuFrame` is a
short-lived borrowed lease when returned directly by `read()`. A
`subscribe_latest()` consumer gets its own retained lease, which stays valid
while that consumer processes it and must then be released.

## Supported baseline

The release baseline is NVIDIA JetPack 6.2.2 on Jetson Orin. NVIDIA lists the
following bundled versions:

- Jetson Linux 36.5 on Ubuntu 22.04;
- CUDA Toolkit 12.6.10;
- TensorRT 10.3.0;
- cuDNN 9.3.0 and VPI 3.2;
- DeepStream 7.1 support.

See NVIDIA's [JetPack 6.2.2 release page](https://developer.nvidia.com/embedded/jetpack-sdk-622).
The toolkit also requires Argus, PyGObject GStreamer bindings, NVMM-capable
GStreamer elements, and the JetPack Multimedia API development headers when
building the CUDA interop extension.

Install a repository environment that can see JetPack system packages:

```bash
uv venv --system-site-packages --allow-existing .venv
uv sync --extra production-preview --extra tensorrt --group dev
sudo apt-get install python-gi-dev libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev gstreamer1.0-libav \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-nice \
  cmake ninja-build
uv run imx-camera-build-interop
```

TensorRT and CUDA are intentionally not installed from PyPI. Verify the active
target rather than assuming an image contains the expected stack:

```bash
cat /etc/nv_tegra_release
nvcc --version
uv run python -c "import tensorrt; print(tensorrt.__version__)"
gst-inspect-1.0 nvarguscamerasrc nvvidconv x264enc h264parse \
  rtph264pay rtph264depay webrtcbin nicesrc nicesink avdec_h264 \
  videoconvert appsink hlssink2
```

## TensorRT engine compatibility

Treat every `.engine` as a trusted, target-local executable artifact. By
default TensorRT records and checks both its exact build version and the GPU
compute capability. NVIDIA also states that general hardware compatibility
mode is not supported on JetPack. See [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html).

The reference `TensorRTRunner` therefore caches an engine only beside metadata
matching all of the following:

- SHA-256 of the ONNX source;
- complete TensorRT version;
- CUDA compute capability;
- FP16 or FP32 precision;
- input tensor name;
- dynamic min/opt/max shape profile.

Any mismatch rebuilds the engine. Do not copy an engine between Jetsons,
JetPack images, TensorRT versions, or model revisions. Never deserialize an
engine from an untrusted source.

## Examples

- [`examples/yolo_detection.py`](../examples/yolo_detection.py) demonstrates
  an end-to-end YOLO export producing `[x1, y1, x2, y2, score, class]`, with a
  model-owned decoder and CUDA rectangle overlay feeding WebRTC.
- [`examples/segmentation.py`](../examples/segmentation.py) consumes an opaque
  segmentation tensor without adding a mask schema to the toolkit.
- [`examples/custom_tensorrt_engine.py`](../examples/custom_tensorrt_engine.py)
  builds or reloads a dynamic custom ONNX engine and enumerates named outputs.
- [`examples/parallel_raw_preview.py`](../examples/parallel_raw_preview.py)
  runs a raw CPU consumer and MJPEG preview from one stable `Camera` pipeline.

The example decoders are application code. Output names, NMS, class labels,
mask interpretation, and overlay policy are intentionally not part of capture.

## Deployment benchmark

Run the complete stable CPU and experimental GPU matrix on the target Jetson:

```bash
uv run imx-camera benchmark jetson --resolution all --backend all \
  --frames 300 --fps 30 --json
```

The command executes 1280x720 and 1920x1080 separately and reports:

- observed consumer FPS and capture/drop counters;
- mean and p95 publication-to-consumer latency;
- process CPU utilization, where 100% means one fully occupied CPU core;
- average Jetson GR3D GPU utilization sampled with `tegrastats`.

Use `--backend cpu` or `--backend gpu` to isolate one path. `--sensor-id` and
`--sensor-mode` select the connected sensor. Keep the Jetson power mode,
clocks, cooling state, camera mode, model, and client count in the benchmark
artifact; results are not portable across those conditions. NVIDIA documents
the sampler format in the [tegrastats guide](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).

## Experimental API policy

`GpuCamera` requires `experimental=True` on every construction. Until the API
is promoted, minor releases may change NVMM handle details, native build
requirements, encoder properties, or supported JetPack versions. `Camera`,
`Frame`, BGR `raw_frame`, and the MJPEG debug path retain their existing stable
semantics.

Promotion requires successful unit CI plus the separate Jetson hardware
workflow for the supported sensor/resolution matrix, TensorRT parity, 720p/30
production encode, clean shutdown/recovery, and a documented compatibility
matrix update.
