# Vision Pipeline Core

`vision` is a framework-independent core for real-time AI Vision workloads.
It separates frame acquisition, inference, optional overlay rendering, and
event delivery without depending on a specific model runtime. The package can
therefore be used with OpenCV, TensorRT, ONNX Runtime, PyTorch, CUDA-backed
buffers, or custom inference implementations.

`CameraFrameSource` adapts the toolkit's existing CSI camera to `FrameSource`
and delivers its raw processed BGR frames directly. `SyntheticFrameSource` and
`FileFrameSource` provide deterministic development and local-preview sources.

## Design

`VisionPipeline` uses two background threads:

```text
FrameSource
    |
    v
Capture thread
    |
    v
Single latest-frame slot
    |
    v
Processing thread
    |
    +--> InferenceResult
    |
    +--> optional OverlayFrame
```

The capture thread and processing thread are intentionally decoupled. The
pending-work slot stores at most one frame. If frame acquisition is faster than
processing, a newly captured frame replaces the waiting frame. The replaced
frame is counted as dropped and is not processed later.

This latest-frame policy is intended for live vision systems where fresh output
is more valuable than complete processing of a growing queue of stale images.
It also bounds in-memory pending-frame storage to one source image.

## Core interfaces

### `FrameSource`

A source supplies opaque image payloads and owns its own resources:

```python
from packages.vision import FrameSource


class MySource(FrameSource):
    @property
    def exhausted(self) -> bool:
        ...

    def open(self) -> None:
        ...

    def read(self) -> object | None:
        ...

    def close(self) -> None:
        ...
```

`read()` returns one image payload. Returning `None` with `exhausted == False`
means that a live source currently has no frame; the pipeline waits briefly and
tries again. Returning `None` with `exhausted == True` ends capture cleanly.

The payload type is deliberately `object`. A source may return a NumPy BGR
array, a CUDA buffer, a decoded tensor, or another image container accepted by
the selected processor.

The pipeline retains payload references rather than copying image buffers.
`Frame` is therefore shallowly immutable: its metadata cannot be changed, but
the image object itself may be mutable. The pipeline never modifies
`Frame.image`; sources and processors must treat a published payload as
read-only after handing it to the pipeline. `OpenCVOverlay` preserves this
contract by drawing on an image copy.

### `FrameProcessor`

A processor receives a `Frame` and returns an `InferenceResult`:

```python
from packages.vision import Frame, FrameProcessor, InferenceResult


class MyProcessor(FrameProcessor):
    def process(self, frame: Frame) -> InferenceResult:
        return InferenceResult(
            frame_sequence=frame.sequence,
            values={"class": "person"},
        )
```

The returned `frame_sequence` must match `frame.sequence`. A mismatched result
is rejected as a processing error.

`InferenceResult` intentionally does not contain the source image. It contains
only structured output:

- `frame_sequence`;
- `detections` as `Detection` objects;
- model-specific `values`;
- completion timestamp.

This separation lets applications publish inference data, events, or telemetry
without retaining or serializing the image buffer.

### Detection coordinate contract

`Detection.box` always uses pixel coordinates of the original `Frame.image`
provided to `FrameProcessor.process()`. The box never refers to model input,
letterboxed input, cropped input, or preview coordinates.

For example, when a `1920×1080` camera frame is letterboxed to `640×640` for a
model, the processor must map each model output back to source-frame pixels
before creating a detection:

```python
Detection(
    label="person",
    confidence=0.98,
    box=BoundingBox(x=320, y=180, width=400, height=500),
)
```

This gives `OpenCVOverlay` an unambiguous coordinate space and lets it render
directly over the source image. Preview layers that resize the image should
resize the overlay image together with the source image rather than reinterpret
the detection coordinates.

## Quick start

The bundled synthetic source and no-op processor make it possible to exercise
the lifecycle without a Jetson, camera, OpenCV, or model runtime:

```python
from packages.vision import (
    NoopFrameProcessor,
    SyntheticFrameSource,
    VisionPipeline,
)

pipeline = VisionPipeline(
    SyntheticFrameSource(max_frames=100),
    NoopFrameProcessor(),
)

pipeline.start()
pipeline.wait_until_stopped(timeout=2.0)

print(pipeline.latest_result)
print(pipeline.stats)
```

For a continuous source, call `stop()` when the application is shutting down:

```python
pipeline = VisionPipeline(
    SyntheticFrameSource(),
    NoopFrameProcessor(),
)

pipeline.start()

try:
    # Run application work here.
    pass

finally:
    pipeline.stop()
```

`start()` is idempotent while the pipeline is running. A new lifecycle resets
the latest frame, result, overlay, error, and counters.

## Bundled frame sources

### Synthetic source

`SyntheticFrameSource` generates deterministic payloads without a camera or
decoder. By default, it yields dictionaries containing an increasing frame
index. Provide a factory to produce suitable test images or model inputs:

```python
from packages.vision import SyntheticFrameSource

source = SyntheticFrameSource(
    frame_factory=lambda index: {"input": index},
    max_frames=500,
    interval=1 / 30,
)
```

Set `max_frames=None` for an unbounded source. The source resets its index when
opened for a new pipeline lifecycle.

### File source

`FileFrameSource` reads an image or video file with the locally installed
OpenCV build:

```python
from packages.vision import FileFrameSource

image_source = FileFrameSource("example.jpg")
video_source = FileFrameSource("example.mp4", loop=True)
```

An image produces one frame unless `loop=True`. A non-looping video ends when
OpenCV reaches its final frame. The source raises `FileNotFoundError` for a
missing path and `RuntimeError` when OpenCV cannot decode or open the file.

On NVIDIA Jetson, use the JetPack-provided OpenCV environment with the project
virtual environment configured through `uv venv --system-site-packages`.

Video playback defaults to `PlaybackMode.UNBOUNDED`, which decodes as quickly
as OpenCV can provide frames. This preserves deterministic benchmark behavior.
For real-time local playback, pace video output using its declared source FPS:

```python
from packages.vision import FileFrameSource, PlaybackMode

video_source = FileFrameSource(
    "example.mp4",
    loop=True,
    playback=PlaybackMode.SOURCE_FPS,
)
```

If the video does not declare a usable FPS value, the source logs a warning and
uses unbounded playback. Static images are unaffected by playback policy.

### Camera source

`CameraFrameSource` is the direct integration with `packages.camera.Camera`:

```python
from packages.camera.camera import Camera
from packages.vision import CameraFrameSource, VisionPipeline

camera = Camera()
source = CameraFrameSource(camera)
pipeline = VisionPipeline(source, processor)
```

By default, the adapter owns the camera lifecycle: opening the vision pipeline
starts the camera and stopping it stops the camera. If a preview API already
owns the shared camera, use `manage_lifecycle=False` and start that camera
before starting Vision Pipeline:

```python
source = CameraFrameSource(camera, manage_lifecycle=False)
```

The adapter reads `Camera.wait_for_raw_frame()` and receives a processed BGR
frame directly. The camera publishes this same raw frame to Vision Pipeline and
JPEG encoding, eliminating an inefficient `BGR → JPEG → BGR` round trip.

## Detections and overlays

Inference implementations may populate `InferenceResult.detections` with
`Detection` objects:

```python
from packages.vision import BoundingBox, Detection, InferenceResult

result = InferenceResult(
    frame_sequence=frame.sequence,
    detections=(
        Detection(
            label="person",
            confidence=0.98,
            box=BoundingBox(x=120, y=80, width=220, height=400),
        ),
    ),
)
```

`OpenCVOverlay` renders these detections onto a copy of the source image:

```python
from packages.vision import OpenCVOverlay, VisionPipeline

pipeline = VisionPipeline(
    source,
    processor,
    overlay=OpenCVOverlay(color=(0, 255, 0), thickness=2),
)
```

The rendered image is available through `pipeline.latest_overlay`. Overlay
failures do not invalidate a successful inference result; they are recorded in
pipeline statistics and emitted as events.

`OpenCVOverlay` requires an OpenCV-compatible image that provides `copy()`.
It is optional, so non-OpenCV inference pipelines can omit it entirely.

## Events

Subscribe to pipeline events before calling `start()`:

```python
from packages.vision import PipelineEvent, PipelineEventType


def on_event(event: PipelineEvent) -> None:
    if event.type is PipelineEventType.RESULT_AVAILABLE:
        print(event.result)


unsubscribe = pipeline.subscribe(on_event)
pipeline.start()

# Remove the listener when it is no longer needed.
unsubscribe()
```

Available event types are:

- `STARTED`
- `FRAME_CAPTURED`
- `FRAME_DROPPED`
- `RESULT_AVAILABLE`
- `OVERLAY_AVAILABLE`
- `SOURCE_EXHAUSTED`
- `SOURCE_ERROR`
- `PROCESSING_ERROR`
- `OVERLAY_ERROR`
- `STOPPED`

Events contain frame identifiers, structured results, optional errors, and
metadata. They never include source image buffers.

Handlers are invoked synchronously by the thread that emits the event. Keep
handlers short and non-blocking; applications that perform I/O or expensive
work must hand events to their own queue or worker. In particular,
`FRAME_CAPTURED` handlers execute on the capture thread and
`RESULT_AVAILABLE` handlers execute on the processing thread. A slow HTTP,
disk, or database call in one of these handlers directly lowers capture or
inference throughput and changes latest-frame drop behavior.

The currently explicit and supported mode is synchronous dispatch:

```python
from packages.vision import EventBus

events = EventBus(mode="synchronous")
```

`VisionPipeline` exposes the selected policy through `pipeline.event_mode`.
A bounded queued dispatcher is intentionally deferred to a later layer, where
its backpressure policy can be chosen deliberately.

## State and diagnostics

The pipeline provides thread-safe snapshots of its latest data:

| Property | Description |
| --- | --- |
| `state` | `STOPPED`, `RUNNING`, or `STOPPING`. |
| `latest_frame` | Most recently acquired source `Frame`, including its image payload. |
| `latest_result` | Most recently completed `InferenceResult`, without image data. |
| `latest_overlay` | Most recently rendered optional `OverlayFrame`. |
| `last_error` | Most recent source, processor, or overlay exception. |
| `stats` | Immutable counters for captured, processed, dropped, and failed work. |

Processor and overlay errors are isolated: capture continues and later frames
can still produce valid results. A source error ends capture, after which the
processing worker completes the newest already-pending frame before stopping.

## Lifecycle and shutdown

`VisionPipeline` opens an optional managed processor, then opens the source
during `start()`. It closes both during normal completion or shutdown.

Use `request_stop()` for non-blocking shutdown requests, especially from an
event handler, signal callback, or API callback:

```python
pipeline.request_stop()
```

`request_stop()` signals workers and closes the source to unblock capture where
possible, but does not join worker threads. Use `stop()` from an application
thread when shutdown must wait for completion:

```python
pipeline.stop(timeout=5.0)
```

When `stop()` is called from a synchronous pipeline event handler, it safely
degrades to a non-blocking stop request. It does not attempt to join the
emitting worker or another worker that may depend on it.

If a source cannot unblock its `read()` operation after `close()`, `stop()` can
raise `RuntimeError` when the timeout expires. Custom live sources should make
`close()` safe to call repeatedly and use it to unblock any pending read.
Source-close failures are recorded as `SOURCE_ERROR` events and in
`pipeline.last_error`, but they never bypass the remaining shutdown sequence.

### Optional processor lifecycle

Simple processors only implement `process()`. Model-backed processors that need
to allocate an engine, CUDA context, or device buffers may additionally satisfy
`ManagedFrameProcessor`:

```python
from packages.vision import Frame, InferenceResult, ManagedFrameProcessor


class TensorRTProcessor(ManagedFrameProcessor):
    def open(self) -> None:
        # Load engine and allocate device resources.
        ...

    def process(self, frame: Frame) -> InferenceResult:
        ...

    def close(self) -> None:
        # Release device resources.
        ...
```

Vision Pipeline opens a managed processor before camera capture starts and
closes it after workers finish. Processor-close failures are reported as
`PROCESSING_ERROR` events without interrupting shutdown.

## Jetson integration guidance

This core intentionally does not select a model framework or expose an HTTP
endpoint. A Jetson integration layer should:

1. Use `CameraFrameSource` for the toolkit camera, or implement `FrameSource`
   for another acquisition backend.
2. Implement `FrameProcessor` with TensorRT, ONNX Runtime, PyTorch, or another
   accelerator-aware runtime.
3. Use `latest_result` or `RESULT_AVAILABLE` events to publish structured
   inference output.
4. Use `OpenCVOverlay` only when a rendered visual stream is required.

This keeps capture, model inference, visual presentation, and transport
independent and replaceable.
