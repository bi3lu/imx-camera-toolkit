# Browser-view guidance

These HTML files are shipped package assets, not development-only pages:

- `simple.html`: MJPEG preview without controls;
- `advanced.html`: MJPEG preview with camera controls;
- `production.html`: WebRTC/HLS client and field-mode login shell.

## Rules

- Preserve the required `data-camera-stream` element and
  `{{ camera_stream_url }}` placeholder in simple/advanced templates.
- Keep endpoint paths and payload schemas synchronized with the FastAPI route
  implementations and tests.
- Production field mode may render `/` publicly, but the page must be data-free
  until authentication succeeds.
- Never persist bearer tokens in `localStorage`, `sessionStorage`, URLs, HLS
  playlist URLs, logs, or source. Exchange them for the server's session-only
  HttpOnly cookie and clear application-held values promptly.
- Do not weaken SameSite/Secure-cookie assumptions or add third-party scripts
  without a deliberate security review and documentation update.
- WebRTC statistics must preserve units and meaning expected by the server
  (`packets`, `bytes`, `frames`, `milliseconds`, and loss counters).
- Show recoverable connection/authentication errors without leaking tokens,
  SDP secrets, internal paths, or full administrative diagnostics.
- Keep layouts usable on desktop and mobile and maintain accessible labels,
  controls, status text, and keyboard behavior.

Validate template changes with preview/API unit tests and, for
`production.html`, the GStreamer WebRTC round-trip plus a real browser on the
target deployment network.
