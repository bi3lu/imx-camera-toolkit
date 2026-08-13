# MJPEG stream guidance

This module formats latest JPEG frames as framework-neutral
`multipart/x-mixed-replace` data.

## Rules

- A stream never opens, starts, stops, or otherwise owns its camera/source.
- One running camera may serve multiple stream iterators.
- Yield only frames newer than the iterator's last frame number. Slow clients
  skip frames; do not add per-client frame queues.
- Keep this module independent of FastAPI and other HTTP frameworks.
- Validate MIME boundaries as safe ASCII tokens and compute accurate
  `Content-Length`; never interpolate untrusted header fragments.
- A missing/invalid non-security YAML file falls back to the complete built-in
  stream config. Explicit constructor values take precedence.

Use `stream.content_type` for an HTTP response rather than rebuilding the MIME
type in another layer.

## Validation

```bash
uv run pytest tests/unit/test_stream.py
uv run pytest tests/benchmarks/test_streaming.py -m benchmark
```

Benchmark results cover formatting throughput only, not network or browser
performance.
