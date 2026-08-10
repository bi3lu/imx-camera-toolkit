# Frame Sources

`frames` is the complete integration boundary between IMX Camera Toolkit and an
application-owned image-processing pipeline. It provides frame acquisition
only; it does not implement inference, overlays, batching, tracking, CUDA
streams, multiprocessing, ROS 2, DeepStream, multi-camera synchronization, or
external data transport.

The public contracts distinguish host-memory `Frame` values from borrowed
`GpuFrame` values. `FrameSource` remains the compatible CPU protocol;
`GpuFrameSource` covers NV12/NVMM sources, and `CaptureFrameSource`/
`CaptureFrame` let model-agnostic consumers accept either mode. Direct GPU
reads are borrowed and a successor invalidates the preceding lease. Camera
subscriptions create independent retained leases that remain valid until the
consumer or worker releases them.

## Contract

```python
from imx_camera_toolkit.frames import FrameSource


class DetectionPipeline:
    def __init__(self, source: FrameSource, detector: object) -> None:
        self.source = source
        self.detector = detector

    def process_once(self) -> None:
        frame = self.source.read(timeout=1.0)

        if frame is not None:
            self.detector(frame.image)
```

`FrameSource.read()` returns the newest available `Frame` or `None`. Sources do
not create an unbounded queue and do not prescribe how applications schedule or
process frames.

## Camera adapter

`CameraFrameSource` adapts an already running toolkit camera without taking
over its lifecycle:

```python
from imx_camera_toolkit import Camera
from imx_camera_toolkit.frames import CameraFrameSource

with Camera(enable_preview=False) as camera:
    source = CameraFrameSource(camera)
    frame = source.read(timeout=1.0)
```

The adapter uses `copy=False` by default and therefore exposes the shared raw
image payload as read-only without another Python API copy. The compatible
camera backend has already converted NV12/NVMM to BGR and materialized the
frame in host RAM; this adapter does not provide GPU zero-copy. Pass
`copy=True` when the application requires an independent image buffer.
