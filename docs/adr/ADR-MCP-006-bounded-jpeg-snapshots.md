# ADR-MCP-006 — Bounded JPEG snapshots

**Status** Accepted

## Context

The [camera snapshot contract](../mcp/README.md#camera-snapshot) requires a
complete, detached JPEG, its publication sequence, and an age measured from
acquisition on the server's monotonic clock. The application owns capture
under the [local stdio profile](ADR-MCP-004-local-sdk-and-transport.md).

| Source | Encoding | Configuration and lifecycle | Timing evidence |
| --- | --- | --- | --- |
| CPU `Camera` | JetPack OpenCV `imencode` on the existing rate-limited JPEG path | Uses the configured output size and quality. The application can enable or disable preview without reopening capture. | The capture loop records monotonic acquisition when its backend returns the BGR frame, then passes that timestamp to the publisher encoding that frame. Software HDR uses the final contributing read, as the CPU frame contract does. |
| `GpuCamera` | JetPack `nvjpegenc` on a dedicated NVMM tee branch with a bounded queue and appsink | The application enables preview in camera configuration. Changing topology while running requires a camera restart. | The raw appsink and JPEG appsink drop independently. The raw sample's monotonic time marks its own read. The JPEG backend returns detached bytes and establishes neither a shared acquisition time nor a matching raw sequence. |
| Application-provided preview | Application chooses an existing isolated encoder | A source may supply a fixed JPEG profile through the internal adapter interface, with the application retaining lifecycle. | A source must supply the server-monotonic acquisition timestamp for the exact encoded frame. |

## Decision

Reuse one latest JPEG slot per publisher. A publication carries immutable
bytes, a JPEG-local sequence, configured dimensions and quality, and
acquisition time when established. The CPU snapshot view filters out
publications acquired before the current capture or preview activation,
including across recovery. Legacy HTTP/MJPEG consumers keep their byte and
sequence interface.

The adapter borrows one configured source. Requests do not enable preview,
reopen capture, read raw frames, acquire NVMM leases, or encode. Built-in
GPU snapshot requests fail with `timing_unavailable` when the preview is
otherwise eligible. The existing GPU HTTP preview continues to consume
detached JPEG bytes. The v1 MCP age field retains its approved acquisition
clock semantics.

Validate input and fixed constraints before JPEG processing. Enforce
configured byte limits before parsing, verify JPEG markers, frame
dimensions, scan boundaries and EXIF absence without decompressing pixels,
and check freshness again before constructing the result. Entropy correctness
is the encoder's responsibility. The SDK boundary measures the serialized
success response including base64, both metadata copies, and the JSON-RPC
envelope before sending an image.
Requests that require a different size or quality fail with
`constraints_unsupported`. Resizing and re-encoding are outside this
implementation.

Positive waits poll the same latest slot at intervals up to 10 ms, shortened
by the effective deadline. They use no worker thread, retain no publication
across an await, and stop on cancellation or terminal source state. The
application/transport work in MCP-007 and MCP-010 owns admission, host
shutdown and final wire enforcement. MCP-008 covers validation shared across
tools.

## Verification

Workstation tests exercise the coherent CPU publication, GPU fail-closed
timing, preview compatibility, freshness boundaries, overlapping readers,
cancellation, shutdown, stale arrivals, malformed JPEG and payload budgets.
A temporary stdio harness used the pinned SDK 2.3.0 client and a synthetic
preview source feeding the adapter. The client received matching JPEG and
metadata, a typed stale error and a cancellable wait. The JPEG decoded to
1 × 1 pixels with macOS `sips`.

The physical Jetson run decoded the packaged CPU snapshot and matched the
returned age to the acquisition time of the same JPEG. The GPU branch
served HTTP JPEGs while the MCP adapter returned `timing_unavailable`.
At JPEG quality 65, two of 90 physical 1080p CPU snapshot attempts exceeded
a payload budget. All 90 attempts at 720p succeeded.
[Snapshot evidence and trial conditions](../mcp/jetson-validation.md)

Before enabling built-in GPU MCP snapshots, establish acquisition
correlation across both branches under drops, recovery and restart, and
verify conversion into the server's monotonic clock domain.
