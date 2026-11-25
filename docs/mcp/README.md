# MCP tool contracts

The approved camera tool contracts target v0.9.0. The application-bound
inference status tool targets v0.9.1. The remaining inference and control
sections describe draft contracts.

## Contract and compatibility

[JSON Schema 2020-12 definitions](../../imx_camera_toolkit/_internal/mcp/tool-contracts.schema.json) specify the
logical objects. [Fixtures](tool-contracts.examples.json) provide tool
arguments and expected outcomes. Each fixture gives the names of the schema
definitions to validate, the input validity, and the expected outcome kind.

| Tool | Input definition | Success definition |
| --- | --- | --- |
| `camera_status` | `CameraInput` | `CameraStatus` |
| `camera_capabilities` | `CameraInput` | `CameraCapabilities` |
| `camera_snapshot` | `SnapshotInput` | `SnapshotMetadata` plus one JPEG |
| `inference_status` | `InferenceStatusInput` | `InferenceStatus` |

The camera ID is case-sensitive. Omitting it selects `primary`, the single
registered camera. Explicit arguments must form an object. Unknown keys,
explicit nulls, and coercion from booleans or strings are invalid. An omitted
arguments object is equivalent to `{}`. Schema defaults document what the
server applies. All numeric values must be finite JSON numbers.
The SDK-independent adapters validate decoded arguments before sampling the
camera, capability provider, or preview source. Snapshot arguments use the
effective server limits, which are validated when the adapter is constructed.
Integer arguments must be decoded as integers, so a JSON value such as
`1.0` is rejected even if an offline JSON Schema validator accepts it.

Successful metadata and domain errors carry integer `schema_version: 1`,
independent of the toolkit version and MCP protocol revision. Nullable
required fields use `null` when information is unknown. For known
collections, an empty array or object means there are zero entries. Servers
emit only declared, approved fields. Clients tolerate new optional output
fields. New optional inputs are usable when advertised by `tools/list`.
Removing required fields or changing meaning, units, types, closed enums,
or accepted arguments incompatibly requires a new major contract version.
Advertise versioned tool names when serving incompatible versions together.
When publishing a tool schema, inline the named object definition and include
the referenced local `$defs` so a client can validate it offline.

All calls in one host share the application's `primary` binding. Capture
normally belongs to the application, and a standalone launch can explicitly
transfer its camera to the host. Host startup enables dispatch, while
capture readiness follows the status contract below.
[Camera ownership and host lifecycle](../adr/ADR-MCP-002-camera-ownership.md)

## Camera status

The observation time is RFC 3339 UTC with a `Z` suffix. It is sampled on the
server and represents the observation, including when there is no camera.
`toolkit_version` is the installed toolkit version. `backend` identifies the
toolkit capture path (`cpu` or `gpu`); it can be known while that path is
stopped. Use `null` when the path cannot be identified.

| State | Source condition |
| --- | --- |
| `unavailable` | No registered source or a stopped source. |
| `starting` | Starting, or running before its first valid publication. |
| `ready` | Running with a valid publication in the current capture run. |
| `recovering` | Recovering, even if an earlier frame is still retained. |
| `error` | Terminal failure. |

Use a coherent camera statistics sample, with failed, recovering, or stopped
lifecycle taking precedence over any retained frame. `capture_fps` is the
recent successful source-read rate in frames per second. `capture_dropped_frames`
is the camera's count of failed reads or frames that were not published as raw
frames. `subscriber_dropped_frames` contains separate per-subscriber frame
counts for application-approved public aliases only. Capture counts apply to
the current camera instance. Subscriber counts apply to the current
subscription instance. Counters can reset when the corresponding instance
changes. An unregistered camera has null metrics.

The SDK-independent read adapter uses one `CameraStats` sample for lifecycle,
publication readiness, FPS, and capture drops. It reads subscriber drops from
the specific subscriptions bound by the application. Bind the replacement
subscription when its instance changes. Use an empty object when the
application has bound no public aliases. If a count exceeds the schema's
integer range, report null for the capture count or omit the affected
subscriber entry.

## Camera capabilities

The camera owner can bind the explicitly selected curated profile,
sensor-specific controller declarations, and a cached read-only Argus
inspection for the same installation. MCP requests reuse these values and
never run discovery or apply controls.

| Field | Evidence |
| --- | --- |
| `sensor_model` | A public model identifier from confirmed metadata, a sensor declaration, or the selected profile, in that order. Pipeline defaults cannot identify an attached sensor. |
| `sensor_modes` | Confirmed mode descriptors when available, then declarations for the matching sensor. Width and height are in pixels and the rate is maximum FPS. Profile capture settings represent one requested operating point. |
| `frame_format` | `BGR_CPU` or `NV12_NVMM` once the corresponding capture path reports a publication in its ready state. Stopped, starting, recovering, and failed paths report `null`. |
| `controls` | Sensor-specific declarations establish support for fixed exposure in microseconds and linear analog gain. A missing property in the cached inspection establishes lack of support for that installation. Property presence alone leaves sensor support unknown. |
| `snapshot` | The host's actual snapshot adapter supplies preview configuration, available evidence of a timed JPEG source, and effective request limits. |

Control values use `false` for known lack of support and `null` for unknown
support. A bounds object with null endpoints means support is declared and
the sensor limits are unknown. The current controller and Argus inspection do
not provide per-sensor numeric bounds. Write tools and permissions follow the
release profile.

`snapshot.configured` is true when the host has bound a source whose preview
is enabled. `snapshot.source_supported` becomes true when that source has
published a JPEG with acquisition timing in the running state. Before that
evidence it is unknown, and a source with no acquisition timing reports false.
For example, the built-in GPU JPEG branch can be configured and usable by HTTP
preview while reporting `source_supported: false` for MCP. Freshness, payload
size and requested dimensions are evaluated by the snapshot tool. Video
encoder availability is outside this contract. GStreamer factory presence
alone cannot establish NVENC capability.

An unregistered camera returns null source fields. An application may supply
its own complete cached projection in place of the default mapping. Camera
dependency or capture errors during a read return `camera_unavailable`, and a
camera timeout returns `request_timeout`.
Unexpected read failures return `internal_error`. The adapter keeps the
original exception as an internal cause.

The advertised snapshot limits are the effective server limits and can be
lower than the schema ceilings. They describe the contract accepted by the
server. An individual source may have fixed dimensions or quality within
those limits. The input rules below still apply.

## Camera snapshot

The server returns one complete JPEG with matching metadata. The following
ceilings apply even when a caller omits the corresponding argument.

| Input or budget | Default | Hard limit |
| --- | --- | --- |
| `max_width`, `max_height` | Effective server bounds | 1–1920 px wide, 1–1080 px high |
| `jpeg_quality` | Source default quality | Integer 1–95 when requested |
| `max_age_ms` | Smaller of 2,000 ms and the server maximum | 1–10,000 ms |
| `wait_ms` | 0 ms, read the latest slot once | 0–1,000 ms, further bounded by the request deadline |
| JPEG bytes before base64 | Server budget | 512 KiB (524,288 bytes) |
| Serialized MCP JSON-RPC response | Transport budget | 768 KiB (786,432 UTF-8 bytes), including base64 and metadata, before transport framing |

An explicit input above an effective limit fails with `invalid_argument`.
Width and height are upper bounds. An explicit quality requests the exact
setting on the encoder's 1–95 scale. If a fixed source cannot satisfy the
request, including when its quality is unknown and an exact quality was
requested or its known encoder setting lies outside the effective bounds, return `snapshot_unavailable` with reason
`constraints_unsupported`. Use a configured bounded preview transform if
available. Keep processing outside capture, reuse detached JPEG bytes, and
keep the response in memory. The image must have no EXIF metadata. An image
over either byte budget yields `payload_too_large`.

`frame_sequence` is the JPEG publication sequence, scoped to this source
instance and its `jpeg_publication` sequence domain. It is independent of
raw capture and inference sequences. The age is sampled after any wait and
transform, just before returning the result. It is elapsed time from the
frame's acquisition in the same server monotonic clock domain. Accept age
equal to the limit. Preserve or correlate the acquisition timestamp with
that exact JPEG. The CPU adapter uses the time the backend returned the
frame being encoded. A source without verified acquisition timing returns
`snapshot_unavailable` with reason `timing_unavailable`. This includes the
built-in GPU preview branch, which supplies detached JPEGs from a separate
appsink. [Snapshot source ADR](../adr/ADR-MCP-006-bounded-jpeg-snapshots.md)

Freshness and waiting are independent. With `wait_ms: 0`, an empty slot
yields `snapshot_unavailable` with reason `no_frame`, and an old frame
yields `snapshot_stale`. A positive wait can use a fresh existing image or
wait for a qualifying publication within the effective deadline. Expiry of
that wait yields `request_timeout`. Source lifecycle failure and known
configuration or format failures can end the call earlier. Cancellation
propagates through the cooperative wait so the host can drain the operation.
The absolute request deadline comes from the host's monotonic clock. An
invalid server-supplied deadline yields `internal_error`. Reject malformed
source JPEG data as a sanitized `internal_error`. The fixed-source adapter
checks JPEG structure, dimensions, EXIF absence, and raw byte size. The SDK
server checks the UTF-8 size of the complete success response, including both
metadata copies, base64, the request ID, and SDK protocol metadata. The
newline used by stdio to frame the JSON-RPC response is outside this budget.

## Errors and MCP representation

Domain failures carry `schema_version`, a stable `code`, a safe `message`
and a `retryable` flag. The flag describes retrying the same arguments after
the indicated transient condition clears. Clients branch on the code and
details instead of the human-readable message. Use allowlisted details only.

| Code | When to use it | Retry unchanged |
| --- | --- | --- |
| `camera_unavailable` | Source unregistered, stopped or failed. | No |
| `camera_starting` | Source starting or recovering. | Yes |
| `snapshot_unavailable` | JPEG source missing or disabled (`no_source`), constraints unsupported, timing unavailable, or latest slot empty (`no_frame`). | Only for `no_frame` |
| `snapshot_stale` | Latest JPEG exceeds the requested capture age in a nonwaiting call. | Yes, after a new publication |
| `invalid_argument` | Valid MCP call with invalid tool arguments. | No |
| `permission_denied` | Tool policy rejects a call that reaches the handler. | No |
| `rate_limited` | Tool budget is exhausted. Optional `retry_after_ms` gives a delay. | Yes, after capacity returns |
| `request_timeout` | Effective tool deadline expired. | Yes |
| `payload_too_large` | Response exceeds a byte budget. | No |
| `internal_error` | Unexpected tool execution failure. | No |
| `inference_not_configured` | Inference tool has no configured result source. | No |
| `no_inference_result` | A configured source has yet to publish a result. | Yes |
| `stale_inference_result` | Latest result fails the age constraint. | Yes, after a new result |

The canonical codes for missing preview and inference data are
`snapshot_unavailable` and `no_inference_result`. Stale errors include the
observed `age_ms` and the applied `max_age_ms`. Keep credentials, paths, raw
exceptions, private configuration and image bytes out of error messages.
The domain layer uses fixed messages and validates detail values before
serialization. It retains original exception causes for internal diagnostics.
A typed camera timeout maps to `request_timeout`. An unclassified exception,
including an unexpected timeout from application code, maps to `internal_error`.

For successful tools, return the logical object in `structuredContent` and
its serialized JSON in a text block for consumers that only read `content`.
Snapshot success adds one MCP image block with MIME type `image/jpeg` and
base64 JPEG data. Metadata carries no image data. The selected local
profile uses the SDK's explicit tool result type and advertises
`outputSchema`. Clients that only read `content` can still read the same
metadata from the JSON text block.
[Local SDK and transport ADR](../adr/ADR-MCP-004-local-sdk-and-transport.md)

Return tool failures as completed MCP results with `isError: true` and one
text block containing the error JSON. Clients read the error block before
applying a success schema. The transport handles malformed protocol requests,
method or tool lookup, and protocol-level server failures as JSON-RPC errors.
Disabled and unknown tool names share the same lookup error. An HTTP auth or
transport rejection before tool execution uses the applicable transport
response. Local calls reaching an enabled tool with insufficient scope get
`permission_denied`. Once authorized, validate the domain arguments before
allocating work capacity. An exhausted tool budget yields `rate_limited`.
The scope and discovery policy is in
[ADR-MCP-003](../adr/ADR-MCP-003-threat-model-and-access-policy.md).
[MCP 2026-07-28 tool specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) ·
[MCP 2025-11-25 tool specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

A workstation stdio probe used the SDK's explicit tool result type with
the published output schemas. A Python SDK 2.3.0 client received the three
success results and a typed stale error from an SDK 1.30.0 server on
`2025-11-25` and an SDK 2.3.0 server on both `2025-11-25` and `2026-07-28`.
On all three runs the JPEG bytes matched the fixture and the structured
metadata matched the JSON text. The supported local profile uses the
`2026-07-28` revision. The Go SDK independently exercised the synthetic
camera host and decoded the JPEG. Desktop UI rendering remains a separate
check.
[Workstation results and Go probe](../adr/ADR-MCP-004-local-sdk-and-transport.md#synthetic-camera-host-results)

## Local tool registration

The internal SDK 2.3.0 server factory binds an application-supplied `MCPHost`.
The default tool set exposes `camera_status` and `camera_capabilities` with
`mcp:camera:status`. A launcher must both enable `camera_snapshot` and grant
`mcp:snapshot:read` for it to appear in discovery and return images. The tool
set and grants are fixed for the lifetime of the server. The application
enters and closes the host around its transport.

Authorized and valid calls share a token bucket that starts with eight
tokens and refills at 30 calls per second. Two snapshot reads can be active
per host. A third read receives `rate_limited` immediately. A snapshot has a
one-second absolute deadline, which also bounds any positive `wait_ms`.
On timeout or disconnect the host requests cancellation and keeps the
slot occupied until the snapshot task exits. Preview polling is cooperative
and cached diagnostics and size-bounded JPEG serialization run inline.

Run the camera tool and stdio protocol tests on a workstation.

```bash
uv sync --frozen --extra preview --extra mcp --group dev
uv run --no-sync pytest \
  tests/integration/test_mcp_stdio.py tests/integration/test_mcp_camera_tools.py
```

These tests use a synthetic camera and an SDK subprocess. They exercise
observer defaults, disabled and unknown names, missing grants, successful
calls, argument validation, JPEG freshness, rate limits, and the exact UTF-8
response ceiling. The stdio checks also cover unsupported revision retry,
overlapping snapshot requests sharing one capture, slot release after timeout
or cancellation, and shutdown before relaunch.

## Local stdio launch

The `mcp` extra installs MCP SDK 2.3.0 and its matching types package
on Python 3.10–3.12. For the v0.9.0 release, install from the built wheel
or source distribution. On Jetson, create the environment with the
JetPack system interpreter so capture can use the system packages:

```bash
uv venv --python /usr/bin/python3 --system-site-packages .venv
uv pip install --python .venv/bin/python './dist/<toolkit-wheel>.whl[mcp]'
```

For a checkout on a POSIX workstation, install the locked test environment
and inspect the command's options:

```bash
uv sync --frozen --extra preview --extra mcp --group dev
uv run --frozen --extra preview --extra mcp --group dev imx-camera mcp --help
```

Configure the trusted client to launch the installed executable by absolute
path, with private stdin and stdout pipes and a separate stderr stream.
Use the environment that also has access to the Jetson capture runtime.

```text
/absolute/path/to/.venv/bin/imx-camera mcp --sensor-id 0
```

The standalone command owns one CPU camera and exposes status and
capabilities by default. Add `--enable-snapshot` to start its fixed JPEG
preview and enable and grant the snapshot tool. Capture starts before the
server accepts requests. Close the write end of the child's stdin pipe
to shut down, then launch a new subprocess to reconnect. The base
installation provides CLI help. A valid launch with a missing or incompatible
SDK reports the extra to install on stderr before accessing capture.

An application already managing capture can call the internal async
`serve_camera_stdio(host)` entry from its subprocess lifespan with a
borrowed host. Start capture before serving and close it from the parent
lifespan after serving returns. The entry claims the process's standard
descriptors once. Ordinary stdin reads return EOF and ordinary stdout
writes go to stderr through process exit. The [transport ADR](../adr/ADR-MCP-004-local-sdk-and-transport.md#hosting-and-lifecycle)
details the pipe and shutdown limits.

Run the transport and camera tool tests on a POSIX workstation with the
pinned SDK and pytest installed.

```bash
uv run --frozen --extra preview --extra mcp --group dev pytest \
  tests/integration/test_mcp_stdio.py tests/integration/test_mcp_camera_tools.py
```

The subprocess tests use synthetic capture to exercise ownership,
partial startup failure, protocol-clean stdout, cancellation, disconnect
and relaunch, and bounded input and output. The installed CLI also passed
physical CPU smoke with Python and Go clients on JetPack 6.2.3. See the
[Jetson validation results](jetson-validation.md) for tested versions,
snapshot limits and CPU/GPU behavior.

## Inference status

An application can bind its existing `InferenceConsumer` using the internal
`InferenceReadAdapter.from_worker(worker)` and pass the adapter to `MCPHost`
as `inference_adapter`. Enable `inference_status` and grant
`mcp:inference:read` in the trusted launcher. The default local tool set
and the standalone camera CLI leave it disabled. Omit arguments or send
`{}`. The host uses one application-bound inference source.

| State | Source condition |
| --- | --- |
| `stopped` | The worker is inactive, even if it retains a previous result. |
| `initializing` | The worker is active and has no result or current failure. |
| `running` | The worker is active and has a result and no current failure. |
| `degraded` | The active worker has consecutive failures. |
| `unknown` | The adapter uses a result-only source without worker diagnostics. |

An enabled and authorized call without a bound source returns
`inference_not_configured`. The adapter samples the latest result once and
reads the worker's cached counters separately. The counters and lifecycle
can change while sampling. The response exposes processed, failed, dropped
and consecutive failure counts, with `null` for unavailable or out-of-range
counts. The current failure indicator is `worker_failure` while consecutive
failures are positive, then `null` after recovery. Only the failure
indicator reaches the client.

`inference_time_ms` measures runner duration. `age_ms` measures elapsed
time from the result's input frame timestamp on the server monotonic clock.
Both may describe a retained result while the worker is stopped or degraded.
Missing or out-of-range timings are `null`. A zero or future frame timestamp
also yields a null age. Status age is diagnostic and applies no freshness
filter. The application owns worker startup, inference and shutdown.
Polling reads cached data without waiting for a new result. Each response is
detached before transport writes.

## Future inference results and controls

`InferenceLatest` defines a draft envelope for the latest result published
by an application-owned worker. The application supplies `source_id`,
`model_id`, a versioned `payload_schema` identifier and a JSON object
matching that schema. The identifier is descriptive and requires no network
lookup. The result age is elapsed server monotonic time since result
publication. `inference_time_ms` measures runner duration when available.
`frame_sequence` identifies the input in `raw_capture` when known. There is
no implied correspondence to a JPEG sequence. The host can already read
one cached result for internal processing. The SDK tool follows application
serialization and freshness enforcement. The future adapter must bound
serialized bytes and collection sizes according to the application schema
and return `payload_too_large` on overflow. The envelope excludes tensors,
device buffers and overlays.

For `camera_set_controls`, the proposed input is `camera_id` and a
nonempty `controls` object with an explicit, server-advertised allowlist of
public settings fields. For example, `{"controls":{"exposure_us":5000}}`
requests a fixed exposure in microseconds. Gain, if enabled, is linear.
Reject unknown or disallowed settings, enforce types and per-camera ranges
through `CameraController`, and reject operations requiring a restart in
the initial control profile. A successful call reports the applied
settings and controller revision. Control authorization and audit use the
later release policy.
