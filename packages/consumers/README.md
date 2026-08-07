# Asynchronous consumers

The consumer layer fans each camera publication into one replaceable slot per
subscriber. Publication only replaces references and signals conditions; it
never calls application code, waits for inference, or builds a frame queue.

```text
capture 30 FPS
  +-> preview slot   -> preview worker, newest JPEG/result
  +-> inference slot -> inference worker 10 FPS, stale inputs overwritten
  +-> telemetry slot -> independent worker
```

## Generic frame worker

Both `Camera` and `GpuCamera` expose `subscribe_latest(name)`. A
`LatestFrameSubscription` reports how many unread frames its private slot has
replaced. Camera health includes the same counter under the subscription name.

```python
from imx_camera_toolkit.consumers import FrameConsumer

subscription = camera.subscribe_latest("histogram")
consumer = FrameConsumer(subscription, calculate_histogram)
consumer.start()
```

`FrameConsumer` runs the callback on its own thread. Stop the consumer before
stopping its camera. Camera shutdown also closes every remaining subscription
and wakes blocked workers. A consumer cannot retain a borrowed `GpuFrame`
beyond the next capture publication; GPU work must import/retain the native
buffer during the callback according to the `GpuFrame` lifetime contract.

## Inference worker

`InferenceConsumer` accepts a GPU subscription and any `InferenceRunner`. It
prepares the runner when the public `FrameSpec` changes, calls `infer()` only on
its worker, retains one newest `InferenceResult`, and optionally fans results
out through `subscribe_results()`.

Use one runner per expensive consumer. The reference `TensorRTRunner` owns one
CUDA stream, so two consumers with two runners have separate Python workers and
CUDA streams. Sharing one runner concurrently is not supported.

## Inference preview

`InferencePreviewSource` combines a JPEG preview source with the newest result
without coupling capture to a model schema:

```python
from imx_camera_toolkit.consumers import InferencePreviewSource

def render(jpeg, result, context):
    return draw_application_overlays(jpeg, result, context.detection_age_ns)

overlay_preview = InferencePreviewSource(camera, inference, render)
overlay_preview.start()
```

The adapter has its own worker and one encoded output slot compatible with the
toolkit's MJPEG source contract. A 10 FPS inference result is reused on each
fresh 30 FPS preview until a newer result arrives. The renderer decides how to
interpret overlays, boxes, masks, labels, or model outputs. The context and
adapter expose the monotonic inference-frame timestamp and detection age for UI
telemetry.
