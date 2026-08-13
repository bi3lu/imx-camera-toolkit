# Production-preview module guidance

This module delivers one shared encoded camera branch through WebRTC or HLS.
It is separate from MJPEG and must not alter CPU/OpenCV capture behavior.

## Media architecture

- One `GpuCamera` encoder branch serves all clients. Each client receives a
  bounded latest encoded slot and leaky downstream queue.
- WebRTC uses H.264 for broad browser compatibility. H.265 is an HLS-only
  option here.
- `AUTO` selects NVENC when present and otherwise x264 for H.264. Orin Nano has
  no NVENC, so only the isolated encoder branch crosses to I420 system memory;
  capture, inference, and CUDA overlays remain NVMM.
- Wait for real SPS/caps before producing H.264 SDP. Do not hardcode a
  `profile-level-id` that may disagree with the active encoder.
- A late WebRTC client starts on an IDR and uses a private timestamp origin.
- HLS paths accept only the rolling playlist and expected segment names; keep
  path traversal impossible and file retention bounded.

## Overlay and metrics

GPU overlays operate on an isolated encoder surface so drawing cannot mutate
the surface used by inference. The mapper remains application/model-owned.
Preserve explicit fail-open/fail-closed error policy and synchronize renderer
CUDA work before encode consumes the surface.

Metrics must describe where data was observed. An accepted appsrc buffer is a
push, not proof of RTP delivery. Keep access-unit, RTP packet, byte, decode,
loss, jitter, RTT, bus, parser, payloader, signaling, ICE, and connection
counters distinct.

## HTTP and security

The camera lifecycle is always application-owned. The FastAPI lifespan may own
only transport workers/peer pipelines. In field mode the root page is a public
data-free login shell; `stream:read` protects media/signaling and `admin`
protects full diagnostics. Exchange browser tokens for session-only HttpOnly
SameSite cookies; never store credentials in JavaScript storage or media URLs.

## Validation

```bash
uv run pytest tests/unit/test_production_preview.py \
  tests/integration/test_gstreamer_h264_roundtrip.py \
  tests/integration/test_gstreamer_webrtc_roundtrip.py
```

The concurrent TensorRT/encode acceptance test is opt-in with
`IMX_PRODUCTION_PREVIEW_HARDWARE=1` and `IMX_TENSORRT_ONNX`; see
`tests/hardware/test_production_video_hardware.py`.
