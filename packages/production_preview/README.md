# Production browser preview

The production preview path shares one Jetson hardware encoder between browser
clients. It is an explicit addition to `GpuCamera`; the existing OpenCV/JPEG
snapshot and MJPEG APIs remain available as the simple debug path.

```text
NV12/NVMM camera tee
  +-> gpu_sink -> TensorRT consumer
  +-> nvjpegenc -> MJPEG debug (optional)
  +-> nvv4l2h264enc/nvv4l2h265enc -> encoded latest slots
       +-> RTP -> webrtcbin per client
       +-> h264parse/h265parse -> hlssink2 rolling segments
```

Raw production-preview pixels do not enter OpenCV, NumPy, or host RAM. The
encoder branch copies only the compressed access unit at its appsink, then
shares that output rather than creating one encoder per browser.

## Runtime installation

Install the optional HTTP layer:

```bash
uv sync --extra production-preview
```

JetPack supplies the accelerated encoder. WebRTC additionally needs the
GStreamer WebRTC/RTP plugins and libnice; HLS needs the bad-plugins HLS sink:

```bash
sudo apt-get install nvidia-l4t-gstreamer gstreamer1.0-plugins-bad \
  gstreamer1.0-nice
gst-inspect-1.0 nvv4l2h264enc nvv4l2h265enc webrtcbin nicesrc hlssink2
```

NVIDIA documents direct NVMM input for both
[`nvv4l2h264enc` and `nvv4l2h265enc`](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html).
GStreamer's [`webrtcbin`](https://gstreamer.freedesktop.org/documentation/webrtc/)
implements peer connection and RTC statistics, while
[`hlssink2`](https://gstreamer.freedesktop.org/documentation/hls/hlssink2.html)
writes the rolling playlist and segments served by this package.

## WebRTC

WebRTC is the preferred low-latency mode. H.264 is required because H.265
WebRTC decoding is not consistently available across browsers.

```python
from imx_camera_toolkit import GpuCamera, HardwareVideoConfig, VideoCodec
from imx_camera_toolkit.production_preview import (
    ProductionPreviewConfig,
    ProductionPreviewServer,
    create_production_preview_app,
)
import uvicorn

camera = GpuCamera(
    enable_preview=False,
    video_config=HardwareVideoConfig(
        codec=VideoCodec.H264,
        bitrate_bps=4_000_000,
        keyframe_interval=30,
    ),
)
transport = ProductionPreviewServer(camera, ProductionPreviewConfig())
app = create_production_preview_app(transport)

with camera:
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

The bundled browser view performs server-offer SDP negotiation and trickle ICE
over small REST messages. Configure `stun_server` and `turn_server` for clients
outside the local network. Each peer receives its own single encoded-frame
slot and leaky RTP queue; slow peers cannot delay capture, inference, encoding,
or another peer.

## HLS

HLS is simpler to deploy behind an ordinary reverse proxy and supports either
H.264 or H.265. It trades latency for rolling one-second segments. The bundled
page uses native browser HLS; deployments targeting browsers without native
HLS should put their preferred JavaScript HLS player in front of the same
playlist endpoint:

```python
from pathlib import Path

from imx_camera_toolkit.production_preview import (
    PreviewTransport,
    ProductionPreviewConfig,
)

config = ProductionPreviewConfig(
    transport=PreviewTransport.HLS,
    hls_directory=Path("/var/lib/imx-camera/hls"),
    hls_target_duration=1,
    hls_playlist_length=3,
    hls_max_files=5,
)
```

The application writes only into the configured directory. `hlssink2` removes
old segments beyond `hls_max_files`. The FastAPI route accepts only
`playlist.m3u8` and `segmentNNNNN.ts`, preventing path traversal.

## GPU overlay and CPU fallback

`CudaOverlayRenderer` is the reference production overlay. It consumes the
newest model-neutral `InferenceResult`, maps application outputs to
`OverlayRectangle` values, and draws directly into an isolated NV12/NVMM
surface on a renderer-owned CUDA stream:

```python
inference = InferenceConsumer(
    camera.subscribe_latest("primary-inference"),
    runner,
)
overlay = CudaOverlayRenderer(inference, mapper=decode_rectangles)
camera.set_video_overlay(overlay)

with camera:
    with inference:
        run_application()

overlay.close()
```

Rebuild the native interop module after installation so it includes the CUDA
rectangle kernel:

```bash
uv run imx-camera-build-interop
```

The encoder branch inserts a device-side `nvvidconv` copy before the overlay
hook. In-place drawing therefore cannot mutate the capture surface being read
by TensorRT. The application-owned mapper keeps boxes, masks, labels, and model
decoding outside capture.

For environments without CUDA overlay support, retain
`InferencePreviewSource`: it overlays JPEGs through an application CPU renderer
and feeds the existing MJPEG debug server. Selecting that fallback is explicit;
production `GpuCamera` rejects a host-memory renderer on its NVMM branch.

## Metrics and validation

`GET /api/preview/health` reports:

- recent hardware `encode_fps` and actual encoded `bitrate_bps`;
- cumulative encoded frames and bytes;
- active browser client count;
- frames/segments sent, bytes sent, dropped frames, and drop rate per client.

WebRTC counts both overwritten per-peer slots and leaky RTP queue overruns. HLS
counts segment gaps observed by each registered browser.

The no-camera tests validate pipeline caps, encoder selection, transport
sharing, metrics, safe HLS paths, and CUDA overlay dispatch. Run the opt-in
Jetson acceptance test with an ONNX model to measure 720p/30 hardware encoding
while TensorRT is active:

```bash
IMX_PRODUCTION_PREVIEW_HARDWARE=1 \
IMX_TENSORRT_ONNX=/opt/models/model.onnx \
uv run pytest tests/hardware/test_production_video_hardware.py
```

It requires at least 25 encode FPS, successful TensorRT inference, and less
than half of one CPU core consumed by the process during the sample window.
