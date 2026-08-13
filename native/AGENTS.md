# Native CUDA interop guidance

This directory builds the Jetson-specific pybind11 bridge used by GPU
preprocessing and CUDA overlays.

## ABI and memory rules

- Build against the target JetPack CUDA, GStreamer, PyGObject, and Jetson
  Multimedia API headers/libraries. The default CUDA architecture `87` is for
  Jetson Orin; override it explicitly for another target.
- Retain the Python boxed `Gst.Buffer` with `gst_buffer_ref` while native code
  uses its `NvBufSurface`, then unmap/unregister/unref in reverse order.
- Mapping the GStreamer descriptor to locate `NvBufSurface` is allowed. Never
  map NV12 pixel planes into CPU memory for the inference path.
- Validate surface memory type, dimensions, supported NV12 color formats,
  plane count, pitches, EGL mapping, and CUDA registration before launching a
  kernel.
- Keep every CUDA operation on the caller/runner-owned stream and synchronize
  at explicit ownership boundaries, not globally by accident.
- Native destructors and error paths must be no-throw, idempotent, and release
  partially acquired resources.
- Preserve compiler/linker hardening. Sanitizers are opt-in host-side aids and
  do not replace Jetson hardware validation.

## Build and validation

```bash
uv sync --extra tensorrt-build
uv run imx-camera-build-interop --cuda-architecture 87
```

Run the real overridden-GStreamer-buffer integration test on a JetPack host:

```bash
uv run pytest tests/integration/test_gstreamer_cuda_interop_buffer.py
```

Changes to preprocessing must also run TensorRT/ONNX Runtime parity. Changes
to rectangle drawing must run production overlay and hardware encode tests.
Do not commit the generated `_cuda_interop*.so` artifact.
