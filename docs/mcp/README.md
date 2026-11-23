# MCP tool contracts

The approved camera tool contracts target v0.9.0. The inference and control
sections record extension points for later releases.

## Contract and compatibility

[JSON Schema 2020-12 definitions](tool-contracts.schema.json) specify the
logical objects. [Fixtures](tool-contracts.examples.json) provide tool
arguments and expected outcomes. Each fixture gives the names of the schema
definitions to validate, the input validity, and the expected outcome kind.

| Tool | Input definition | Success definition |
| --- | --- | --- |
| `camera_status` | `CameraInput` | `CameraStatus` |
| `camera_capabilities` | `CameraInput` | `CameraCapabilities` |
| `camera_snapshot` | `SnapshotInput` | `SnapshotMetadata` plus one JPEG |

The camera ID is case-sensitive. Omitting it selects `primary`, the single
registered camera. Explicit arguments must form an object. Unknown keys,
explicit nulls, and coercion from booleans or strings are invalid. An omitted
arguments object is equivalent to `{}`. Schema defaults document what the
server applies. All numeric values must be finite JSON numbers.

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

## Camera capabilities

Project only information from the registered source, using existing
configuration, negotiated state and discovery results. Reads reuse those
results. `sensor_model` is a public model identifier. `sensor_modes` lists
known applicable mode indices, dimensions in pixels, maximum FPS and HDR
support. A mode field can be unknown. `frame_format` reports the active
negotiated frame format. `controls` describes fixed exposure in microseconds
and linear gain. A bounds object indicates known support and may have
unknown endpoints. Use `false` for known lack of support and `null` when
support is unknown. Write tools and permissions follow the release profile.
`snapshot.source_supported` indicates whether a JPEG source is supported
by this installation. `snapshot.configured` indicates
whether one is configured for MCP. These describe the source, while discovery
and calling permissions are evaluated for each caller. A supported source may
be disabled. Successful capabilities responses can represent an unregistered
camera using null source data.

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
requested, return `snapshot_unavailable` with reason
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
that exact JPEG. A source with only publication timing returns
`snapshot_unavailable` with reason `timing_unavailable`. GPU capture timing
correlation is to be verified in MCP-006.

Freshness and waiting are independent. With `wait_ms: 0`, an empty slot
yields `snapshot_unavailable` with reason `no_frame`, and an old frame
yields `snapshot_stale`. A positive wait can use a fresh existing image or
wait for a qualifying publication within the effective deadline. Expiry of
that wait yields `request_timeout`. Source lifecycle failure and known
configuration or format failures can end the call earlier. Reject malformed
source JPEG data as a sanitized `internal_error`.

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
| `inference_not_configured` | Future inference tool has no configured result source. | No |
| `no_inference_result` | A configured source has yet to publish a result. | Yes |
| `stale_inference_result` | Latest result fails the age constraint. | Yes, after a new result |

The canonical codes for missing preview and inference data are
`snapshot_unavailable` and `no_inference_result`. Stale errors include the
observed `age_ms` and the applied `max_age_ms`. Keep credentials, paths, raw
exceptions, private configuration and image bytes out of error messages.

For successful tools, return the logical object in `structuredContent` and
its serialized JSON in a text block for consumers that only read `content`.
Snapshot success adds one MCP image block with MIME type `image/jpeg` and
base64 JPEG data. Metadata carries no image data. If the selected SDK
convenience layer cannot compose this result, use the SDK's explicit tool
result type. A transport profile that can only send the image and text
blocks must omit `outputSchema` at discovery and supply the same JSON
metadata text. MCP-004 establishes the supported profile.

Return tool failures as completed MCP results with `isError: true` and one
text block containing the error JSON. Clients read the error block before
applying a success schema. Reserve JSON-RPC errors for malformed protocol
requests, method or tool lookup, and protocol-level server failures. An HTTP
auth or transport rejection before tool execution uses the applicable
transport response. MCP-003 defines the scope and discovery policy.
[MCP 2026-07-28 tool specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) ·
[MCP 2025-11-25 tool specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

A workstation stdio probe used the SDK's explicit tool result type with
the published output schemas. A Python SDK 2.3.0 client received the three
success results and a typed stale error from an SDK 1.30.0 server on
`2025-11-25` and an SDK 2.3.0 server on both `2025-11-25` and `2026-07-28`.
On all three runs the JPEG bytes matched the fixture and the structured
metadata matched the JSON text. Client rendering and the supported local
profile are part of MCP-004. The probe used synthetic data.

## Future inference and controls

`InferenceLatest` defines a draft envelope for the latest result published
by an application-owned worker. The application supplies `source_id`,
`model_id`, a versioned `payload_schema` identifier and a JSON object
matching that schema. The identifier is descriptive and requires no network
lookup. The result age is elapsed server monotonic time since result
publication. `inference_time_ms` measures runner duration when available.
`frame_sequence` identifies the input in `raw_capture` when known. There is
no implied correspondence to a JPEG sequence. The read uses an existing
result. The future adapter must bound serialized bytes and collection sizes
according to the application schema and return `payload_too_large` on
overflow. The envelope does not publish tensors, device buffers or overlays.

For `camera_set_controls`, the proposed input is `camera_id` and a
nonempty `controls` object with an explicit, server-advertised allowlist of
public settings fields. For example, `{"controls":{"exposure_us":5000}}`
requests a fixed exposure in microseconds. Gain, if enabled, is linear.
Reject unknown or disallowed settings, enforce types and per-camera ranges
through `CameraController`, and reject operations requiring a restart in
the initial control profile. A successful call reports the applied
settings and controller revision. Control authorization and audit use the
later release policy.
