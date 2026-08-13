# Inference module guidance

This module provides model-neutral TensorRT execution and Jetson NVMM/CUDA
interop. It does not define YOLO, detection, segmentation, NMS, labels, or
application result schemas.

## Runner contract

- `prepare(FrameSpec)` builds or loads resources for an explicit NV12/NVMM
  layout.
- `infer(GpuFrame)` must finish while the borrowed lease is valid and returns
  named `TensorOutput` values plus portable metadata.
- `close()` is idempotent and releases runner context, buffers, and streams.
  Per-frame native registrations must be released before `infer()` returns.
- The reference runner supports one float32 NCHW image input, dynamic
  min/opt/max profiles, FP16/FP32, stretch or letterbox preprocessing, and
  arbitrary named outputs.

Build or load an engine before opening Argus when compilation may take minutes.
TensorRT build and active inference calls cannot be interrupted safely; do not
pretend a worker timeout cancels them.

## CUDA interop

The intended path is `Gst.Buffer` -> `NvBufSurface` -> EGLImage -> CUDA EGL
frame -> NV12-to-NCHW kernel -> TensorRT binding on one runner-owned stream.
Mapping the small GStreamer descriptor is permitted; mapping camera pixel
planes into a CPU image or adding a hidden host upload is not.

Keep Python optional-runtime imports lazy. CUDA and TensorRT come from JetPack,
not PyPI. Native ABI errors must surface as actionable `InferenceDependencyError`
or `CudaInteropError` messages.

## Engine and model security

- Cache acceptance requires matching ONNX SHA-256, serialized engine digest,
  TensorRT version, compute capability, precision, input name, and complete
  shape profile.
- Engines are target-local executable artifacts, not distributable model
  files or trust anchors.
- Preserve atomic cache writes, strict owner/mode checks, symlink rejection,
  and rebuild-on-invalid behavior.
- Signed deployments verify the exact Ed25519-signed manifest bytes, ONNX
  digest, and declared input/output names before building or loading an engine.

## Validation

```bash
uv run pytest tests/unit/test_inference_contracts.py
```

The parity test requires JetPack, a physical NVMM frame, ONNX Runtime, and
`IMX_TENSORRT_INTEGRATION=1`:

```bash
IMX_TENSORRT_INTEGRATION=1 \
  uv run pytest tests/integration/test_tensorrt_onnx_parity.py
```

Do not weaken numerical tolerances or security checks merely to make a host
mock pass.
