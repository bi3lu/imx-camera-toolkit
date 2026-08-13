# Generic preview guidance

`PreviewServer` converts application-owned images into a latest JPEG snapshot
and MJPEG transport. It is model-neutral and does not own an optional source.

## Rules

- `start()`/`stop()` manage only the forwarding worker, never the source.
- Retain one encoded JPEG and apply the configured maximum FPS; do not build a
  queue for slow clients.
- Accept opaque CPU image payloads or `Frame.image`; do not interpret boxes,
  masks, labels, tracking IDs, or inference outputs.
- Source reads use `copy=False` where supported and treat the returned CPU
  image as read-only.
- Keep optional FastAPI imports inside `create_app()` so the core preview
  transport remains importable without HTTP dependencies.
- Reuse the shared `SecurityConfig` and scope semantics from the API module.

The standalone app owns the preview worker through its lifespan. It does not
own a source camera, even when forwarding from one.

## Validation

```bash
uv run pytest tests/unit/test_preview_server.py tests/unit/test_preview.py
```

Cover manual publication, source forwarding, rate limiting, startup/shutdown,
source errors, view validation, and authentication without physical hardware.
