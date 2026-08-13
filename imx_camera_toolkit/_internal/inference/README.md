# GPU inference integration

The inference package defines a model-agnostic boundary between camera capture
and optional GPU consumers. Core camera APIs do not depend on TensorRT, ONNX,
YOLO, or a particular output schema.

## Public contract

`InferenceRunner` has three operations:

```python
runner.prepare(frame_spec)
result = runner.infer(gpu_frame)
runner.close()
```

`InferenceResult` contains the input sequence, monotonic input-frame timestamp,
optional hardware capture timestamp, elapsed inference time, named
`TensorOutput` values, immutable metadata, and optional opaque overlays
supplied by the application. It does not define boxes, classes, masks, NMS, or
YOLO-specific fields. The monotonic timestamp lets a preview or UI calculate
the age of the result without guessing which frame was evaluated.

## JetPack 6.2.2 interoperability

The selected interop path is:

```text
Gst.Buffer (NV12/NVMM)
  -> NvBufSurface (NVBUF_MEM_SURFACE_ARRAY)
  -> EGLImage
  -> CUDA CUeglFrame
  -> NV12-to-NCHW CUDA kernel
  -> TensorRT input binding on the same cudaStream_t
```

The pybind11 extension obtains the boxed `Gst.Buffer` through the official
PyGObject C API, retains it for the synchronous `infer()` call, and maps only
the small GStreamer descriptor needed to locate `NvBufSurface`. It never maps
the NV12 pixel planes to a CPU address. `NvBufSurfaceMapEglImage` and CUDA EGL
registration expose those planes directly to the preprocessing kernel.

The kernel performs resize, NV12 color conversion, channel ordering,
normalization, and NCHW layout directly into runner-owned CUDA memory. There is
no BGR image, NumPy camera input, host upload, or implicit CPU fallback.

## Installation and native build

TensorRT, CUDA, and a compatible NumPy must come from the target JetPack
installation. They are not installed from PyPI because engine, runtime, OpenCV,
and NumPy compatibility is platform specific. Install only the native-extension
build dependency with:

```bash
uv sync --extra tensorrt-build
```

The legacy `tensorrt` extra remains an alias for the same build dependency.
Applications that verify signed model manifests can independently install
`--extra model-security`. The `tensorrt-test` extra is intended for isolated
test environments and permits NumPy 1.21 or newer.

The native module additionally needs JetPack's Multimedia API and development
headers:

```bash
sudo apt-get install python-gi-dev libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev cmake ninja-build
uv run imx-camera-build-interop
```

The default CUDA architecture is `87` for Jetson Orin. Override it when
building for another Jetson GPU:

```bash
uv run imx-camera-build-interop --cuda-architecture 72
```

## TensorRT runner

`TensorRTRunner` parses ONNX with TensorRT, builds FP16 or FP32 explicit-batch
engines, binds all outputs by tensor name, and supports a true dynamic
`ShapeProfile` rather than assuming 640×640:

```python
from pathlib import Path

from imx_camera_toolkit import FrameFormat, GpuCamera, MemoryType
from imx_camera_toolkit.inference import FrameSpec, ShapeProfile, TensorRTRunner

runner = TensorRTRunner(
    "detector.onnx",
    cache_dir=Path(".cache/tensorrt"),
    precision="fp16",
    shape_profile=ShapeProfile(
        minimum=(1, 3, 320, 320),
        optimum=(1, 3, 640, 640),
        maximum=(1, 3, 1280, 1280),
    ),
    inference_shape=(1, 3, 640, 640),
    resize_mode="letterbox",
    padding_value=(114, 114, 114),
)

frame_spec = FrameSpec(
    width=1280,
    height=720,
    format=FrameFormat.NV12_NVMM,
    memory_type=MemoryType.NVMM,
)
runner.prepare(frame_spec)

with GpuCamera() as camera:
    frame = camera.read(timeout=2.0)
    if frame is not None:
        result = runner.infer(frame)
        camera.record_stage_latency("inference", result.inference_time_ns)

runner.close()
```

Prepare before opening Argus when an engine may need to be built. TensorRT
engine compilation is synchronous and cannot be interrupted; holding a borrowed
NVMM frame during a multi-minute build unnecessarily delays both capture and
shutdown. Build logging reports cache misses, ONNX parsing, FP16/FP32 build
start and elapsed time, cache storage, and cached-engine deserialization.

`resize_mode="letterbox"` preserves the source aspect ratio and fills centered
padding with `padding_value`. Every result includes `preprocessing.transform`
with `scale`, `pad_x`, `pad_y`, `source_shape`, and `model_shape`, and the same
data is available as `runner.resize_transform`. The default `stretch` mode
retains the previous behavior.

With the default `input_name=None`, the runner selects the model's only ONNX
input regardless of its name. Pass an explicit `input_name` for models with
multiple inputs; ambiguous automatic selection raises a configuration error.

For live operation, prefer the asynchronous latest-frame adapter over polling
`read()`. It provides one input slot and one worker per expensive consumer, so
model throughput cannot accumulate a capture backlog:

```python
from imx_camera_toolkit import FrameFormat, GpuCamera, MemoryType
from imx_camera_toolkit.consumers import InferenceConsumer
from imx_camera_toolkit.inference import FrameSpec

frame_spec = FrameSpec(1280, 720, FrameFormat.NV12_NVMM, MemoryType.NVMM)
runner.prepare(frame_spec)

with GpuCamera() as camera:
    with InferenceConsumer(
        camera.subscribe_latest("primary-inference"),
        runner,
        prepared_spec=frame_spec,
    ) as inference:
        result = inference.latest_result
```

Create a separate `TensorRTRunner` for each independent inference consumer.
Each runner owns its CUDA stream, while each `InferenceConsumer` owns its worker
thread. Slow consumers overwrite only their own unread slot and automatically
contribute named drop counters to camera health metrics.

`InferenceConsumer` also discovers `runner.prepared_frame_spec`, so the explicit
`prepared_spec` is optional for `TensorRTRunner`. Its `health()` mapping reports
worker state, successful/failed/dropped counts, the latest error, inference
time, and named output shapes. `stop(timeout=...)` cannot interrupt a TensorRT
builder or an active handler; a cleanup timeout is logged without masking an
exception already propagating from a context-manager body.

The reference runner supports one float32 NCHW image input and arbitrary named
output tensors. Model-specific decoding, NMS, labels, masks, and overlays stay
in the consuming application.

## Engine cache safety

Each `.engine` has an adjacent JSON metadata file containing:

- SHA-256 of the ONNX file;
- TensorRT version;
- CUDA compute capability;
- FP16/FP32 precision;
- input tensor name;
- complete dynamic min/opt/max shape profile.
- SHA-256 of the serialized engine itself.

The runner deserializes a cache entry only when every field matches. Missing,
corrupt, empty, or incompatible entries cause an ONNX rebuild and atomic local
replacement. Engine files must not be copied between Jetsons or TensorRT
versions without this validation.

Cache directories must be owned by root or the process and use `0700`/`0750`;
engine and metadata files use `0600`/`0640`. Symlinks, unexpected ownership,
permissions, metadata, or engine digests make the entry untrusted and cause a
rebuild. The engine remains a disposable local artifact rather than a model
trust anchor.

For deployed models, set `require_signed_model=True` and provide an Ed25519 PEM
`public_key_path`. The runner verifies `model.manifest.sig` over the exact bytes
of `model.manifest.json`, checks the manifest SHA-256 against `model.onnx`, and
requires the discovered input/output tensor names to match the signed contract.
The schema-versioned manifest contains `model_sha256`, `model_version`,
`inputs`, and `outputs`. Model, manifest, signature, and trust anchor must be
root/process-owned regular files that are not group/world writable.

## Parity validation

The opt-in integration test preprocesses one retained IMX camera frame through
the CUDA interop layer, evaluates the same dynamic ONNX graph with ONNX Runtime
and TensorRT FP16, and compares its generic `boxes` tensor numerically:

```bash
uv sync --extra preview --extra tensorrt-test
uv run imx-camera-build-interop
IMX_TENSORRT_INTEGRATION=1 \
  uv run pytest tests/integration/test_tensorrt_onnx_parity.py
```

ONNX Runtime is a test-only reference and is not used by `TensorRTRunner`.
