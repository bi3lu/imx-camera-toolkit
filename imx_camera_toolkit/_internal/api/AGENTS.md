# Camera HTTP API guidance

This module composes a shared camera, controls, MJPEG transport, FastAPI
lifecycle, validated configuration, and deployment security. It is the simple
debug/browser API; production WebRTC/HLS lives in `production_preview`.

## Lifecycle and streaming

- `create_app()` owns camera startup/shutdown only when `manage_camera=True`.
  Applications attaching an existing camera keep lifecycle ownership.
- Every request shares one camera. Never open a camera per endpoint or client.
- Snapshot and MJPEG handlers consume latest JPEGs and must not buffer an
  entire stream or block the capture thread.
- Keep `/healthz` minimal and public. Detailed health belongs behind `admin`.
- Preserve documented snapshot status codes: `200` for an image, `204` when a
  requested newer frame did not arrive, and `503` when no image is available.

## Security

Field mode is fail-closed: it requires token grants and valid configuration,
disables OpenAPI/Swagger, validates Host, applies body/rate limits and security
headers, and optionally requires HTTPS. Preserve scope boundaries:

- `stream:read`: preview streams and signaling;
- `camera:read`: control state;
- `camera:control`: camera updates;
- `profiles:write`: profile mutation;
- `admin`: detailed diagnostics and superuser access.

Token files contain SHA-256 digests, not plaintext credentials. Keep regular
file, ownership, mode, schema, and symlink checks. Middleware must remain
streaming-safe and must not buffer response bodies.

## Views and configuration

Simple/advanced templates require the `data-camera-stream` marker and
`{{ camera_stream_url }}` placeholder. Reject an invalid template instead of
serving a page without a live stream. Development config may use complete
built-in fallback; field mode must fail startup on an invalid supplied config.

## Validation

```bash
uv run pytest tests/unit/test_api_security.py tests/unit/test_preview.py \
  tests/integration/test_api_with_mock_camera.py
```

Test routes with `MockCamera`; hardware behavior belongs in camera tests.
