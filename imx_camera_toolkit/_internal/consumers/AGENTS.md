# Consumer module guidance

Consumers isolate expensive work from capture with one replaceable slot per
subscriber and one worker per consumer.

## Scheduling and ownership

- Publishing replaces references and signals waiters; it never calls user code
  or waits for a worker.
- Each subscriber has an independent latest slot and drop counter. Do not
  replace this with FIFO/backlog behavior.
- A worker releases every retained `GpuFrame` in success, drop, error, and
  shutdown paths.
- Stop consumers before their camera when the application owns both. Camera
  shutdown must still close subscriptions and wake blocked workers.
- Error callbacks and logging are bounded; repeated failures use bounded
  backoff. `GpuFrameExpiredError` before inference is a dropped input, not a
  failed model invocation.

## Inference consumers

`InferenceConsumer` accepts the model-neutral `InferenceRunner` protocol. One
consumer should own one runner/CUDA stream. Do not share a `TensorRTRunner`
concurrently. Prepare on frame-spec changes, retain only the newest result, and
report health without assuming boxes, masks, YOLO, or labels.

`InferencePreviewSource` may reuse a slower result on newer preview frames. Its
renderer and freshness policy remain application-owned.

## Validation

```bash
uv run pytest tests/unit/test_consumers.py \
  tests/unit/test_inference_contracts.py
```

Use deterministic synchronization rather than sleeps where possible. Cover
drop accounting, release on exceptions, stop timeouts, recovered health, and
independent subscribers.
