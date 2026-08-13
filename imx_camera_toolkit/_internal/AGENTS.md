# Internal implementation guidance

This package is private, but its behavior implements stable public contracts.
Private status permits refactoring; it does not permit silent ownership,
threading, memory-domain, or error-semantics changes.

## Layering

- Camera modules own capture resources and publish frames.
- Frame contracts describe what consumers may receive.
- Consumers schedule application work without blocking publishers.
- Inference owns CUDA/TensorRT resources but remains model-neutral.
- Stream and preview modules transport already-produced frames.
- API modules compose lifecycle, controls, security, and transport.

Avoid circular ownership. A transport must not start a camera unless its public
factory explicitly owns that lifecycle; a consumer must not stop its source;
capture must not call application inference code on its worker thread.

## Shared implementation rules

- Keep imports of optional runtimes inside constructors/factories or explicit
  runtime loaders. Host unit tests must remain usable without Jetson hardware.
- Validate external paths, enums, sizes, timeouts, shapes, GStreamer property
  fragments, and security inputs before acquiring expensive resources.
- Use locks/conditions around shared latest values and wake waiters on shutdown.
- Cleanup must be idempotent and preserve an exception already raised by the
  caller whenever possible.
- Never silently convert between CPU/BGR and GPU/NVMM contracts.
- Metrics names and units are API: distinguish frames, access units, RTP
  packets, bytes, nanoseconds, and percentages.
- Raise toolkit-specific errors at public subsystem boundaries and retain the
  original exception with `raise ... from error` when useful.

Top-level internal files such as `cli.py`, `diagnostics.py`, `benchmarks.py`,
and `telemetry.py` are governed here. CLI changes must keep commands bounded,
machine-readable `--json` output stable, and physical-camera actions explicit.
