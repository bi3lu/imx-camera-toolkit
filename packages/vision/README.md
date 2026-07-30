# Vision Pipeline Core

`vision` is a framework-independent core for real-time AI Vision workloads.
It separates frame acquisition, inference, optional overlay rendering, and
event delivery without depending on a specific model runtime. The package can
therefore be used with OpenCV, TensorRT, ONNX Runtime, PyTorch, CUDA-backed
buffers, or custom inference implementations.

The package does not own a CSI camera directly. A camera adapter can implement
`FrameSource` in a later integration layer, while `SyntheticFrameSource` and
`FileFrameSource` provide immediately usable development sources.

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
work should hand events to their own queue or worker.

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

`VisionPipeline` opens the source during `start()` and closes it during normal
completion or `stop()`. `stop()` requests termination, closes the source to
unblock capture where possible, and waits for both workers.

```python
pipeline.stop(timeout=5.0)
```

If a source cannot unblock its `read()` operation after `close()`, `stop()` can
raise `RuntimeError` when the timeout expires. Custom live sources should make
`close()` safe to call repeatedly and use it to unblock any pending read.

## Jetson integration guidance

This core intentionally does not select a model framework or expose an HTTP
endpoint. A Jetson integration layer should:

1. Adapt the selected camera backend to `FrameSource`.
2. Implement `FrameProcessor` with TensorRT, ONNX Runtime, PyTorch, or another
   accelerator-aware runtime.
3. Use `latest_result` or `RESULT_AVAILABLE` events to publish structured
   inference output.
4. Use `OpenCVOverlay` only when a rendered visual stream is required.

This keeps capture, model inference, visual presentation, and transport
independent and replaceable.
