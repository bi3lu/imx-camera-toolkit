# ADR-MCP-004 — Local MCP SDK and transport

**Status** Accepted for v0.9.0. The physical runtime gate remains in
[ADR-MCP-001](ADR-MCP-001-python-sdk-runtime.md#validate-on-jetson-orin-nano-super).

## Release profile

| Concern | Decision |
| --- | --- |
| SDK | Pin the optional MCP runtime to `mcp==2.3.0`. The SDK requires `mcp-types==2.3.0` transitively. |
| Python | Toolkit support is Python 3.10–3.12. The SDK and its types package both require Python 3.10 or later. On Jetson, use the JetPack system Python 3.10 with access to system packages, subject to physical validation. |
| Protocol | Target `2026-07-28`. Use `server/discover` to verify capabilities and carry protocol version and client capabilities in each request's `_meta`. |
| Local transport | The client launches one application subprocess with an absolute executable path, fixed argument vector, minimal environment and private stdio pipes. The launcher supplies grants under ADR-MCP-003. There is no local MCP HTTP listener. |
| Hosting | Run the SDK in the camera-owning application's Python process. The application owns capture and optional producers; the injected MCP adapters borrow those sources. |

The SDK v2 API supports discovery, per-request metadata, explicit tool results
and stdio cancellation using the selected interpreter range. The exact pin
bounds the tested SDK API, types and wire behavior. The lockfile bounds the
resolver output for transitive dependencies on supported targets. MCP-012 adds
the optional dependency and tests built artifacts and SDK-free core imports.
SDK 1.30.0 remains the spike's maintenance-line comparison. Before changing
the pin, review upstream release notes, support policy, dependencies and
advisories. Regenerate the relevant locks and rerun the SDK-free core checks,
the protocol matrix and the Jetson gate with the proposed pin. Recheck
cancellation, typed errors, result encoding and cleanup before widening any
support claim.
[SDK v2.3.0 release](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.3.0) ·
[SDK package metadata](https://pypi.org/project/mcp/2.3.0/) ·
[Types package metadata](https://pypi.org/project/mcp-types/2.3.0/) ·
[Protocol versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)

## Hosting and lifecycle

The launch target is an application entry point. Its concrete invocation is
implemented in MCP-011/MCP-012 and documented in MCP-015. The client owns the
child process and closes its stdin to request graceful shutdown. The
application validates configuration and grants before opening capture, then
starts its sources before serving requests. Shutdown stops dispatch and
releases MCP waits before the application closes its owned producers and
camera. For borrowed sources, the MCP host only closes its own resources.
Relaunch starts a fresh process and rediscovers tools. An application with a
physical sensor gives that sensor one owner. Starting a second subprocess
against the same sensor is outside this profile; multiple in-flight calls on
a single host share its sources. MCP-007 owns the injected lifecycle contract.

Keep stdout exclusively for UTF-8 JSON-RPC messages, one per line. Route
operational logs and native library output to stderr, including at capture
startup. An EOF or shutdown closes request work within the host's cleanup
budget. A client cancels a cooperative wait with `notifications/cancelled`.
Cancellation stops response delivery and releases the wait promptly. Work
already inside a non-interruptible call retains its admission slot until it
actually finishes. A deadline protects both the tool and shutdown paths.
[stdio binding](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio) ·
[Local scopes and grants](ADR-MCP-003-threat-model-and-access-policy.md#local-profile)

Loopback Streamable HTTP would let several clients attach to one long-lived
application, including a camera already owned by a web application. It adds
a listener restricted to loopback, per-launch credentials distinct from
field-mode credentials, an exact Host/port allowlist, validation of any
supplied Origin, and network request admission. A native client can connect
without an Origin if it authenticates. An ASGI mount would also need the parent
lifespan to enter the SDK transport manager exactly once and coordinate its
teardown with the camera owner. At `2026-07-28`, HTTP requests are
stateless at the protocol layer. Supporting the earlier HTTP revision adds
initialization and compatible handling of session state. Stdio fits the
trusted local client-launch model and avoids these additional hosting
requirements.
[Streamable HTTP binding](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http) ·
[SDK ASGI mounting example](https://py.sdk.modelcontextprotocol.io/run/asgi/#mounting-it)

## Workstation interoperability evidence

The [fixture probe](../../examples/mcp_runtime_spike/probe_profile.py) uses
real stdio subprocesses and the approved synthetic contract fixtures. It
verifies discovery, `tools/list`, the three camera calls, SDK validation of
structured output, a 619-byte JPEG and matching JSON metadata, a completed
`snapshot_stale` error and a JSON-RPC `-32602` lookup error. The client
cancels after a progress notification and lists tools again after the
server's wait releases. A fresh subprocess serves status after relaunch.
The wire check requests `2099-01-01`, receives `-32022` with the requested
revision and `["2026-07-28"]` in `supported`, then retries discovery on the
same pipe with the supported revision. Closing stdin makes the child exit
cleanly with diagnostics on stderr.

| Client | Server | Revision and outcome |
| --- | --- | --- |
| SDK 2.3.0, auto-discovery | SDK 2.3.0 fixture probe | `2026-07-28`. Full probe passes on macOS ARM64 with CPython 3.10.20, 3.11.13 and 3.12.0. The received JPEG also decoded to a 1×1 PNG with macOS Image I/O. |
| SDK 2.3.0, forced legacy mode | SDK 2.3.0 and SDK 1.30.0 runtime spike | `2025-11-25`. Handshake, list, status and synthetic JPEG pass. Legacy compatibility evidence only. |
| Raw JSON-RPC probe | SDK 2.3.0 fixture probe | Unsupported revision rejected, discovery retry succeeds on `2026-07-28`. |

The Python client exposes typed image blocks and structured metadata.
A client UI needs to handle both blocks to display the image and metadata.
Visual behavior in desktop clients and a second independent client are
pending MCP-028. Handshake-only clients and HTTP-only clients are outside
the local release profile. SDK 2.3.0 can accept legacy stdio initialization,
so dispatch must enforce the same grants on any request it admits.

Run from the repository root on a workstation. The optional macOS commands
below render the JPEG received by the client to a PNG and report its dimensions.

```bash
spike_dir=examples/mcp_runtime_spike
uv run --project "$spike_dir" --locked --group sdk-v2 \
  python "$spike_dir/probe_profile.py"

render_dir=$(mktemp -d)
uv run --project "$spike_dir" --locked --group sdk-v2 \
  python "$spike_dir/probe_profile.py" --jpeg-output "$render_dir/received.jpg"
sips -s format png "$render_dir/received.jpg" --out "$render_dir/received.png"
sips -g pixelWidth -g pixelHeight "$render_dir/received.png"
```

## Remote profile follow-up

MCP-021 must select the deployed client/SDK/revision matrix and recheck
the standards before implementing remote Streamable HTTP. Verify POST
handling, Accept negotiation, request metadata headers and header/body
agreement, modern errors and HTTP status mapping, SSE proxy buffering,
disconnect cancellation and retry semantics. Modern HTTP has no standalone
GET SSE stream or protocol session ID. If a separate legacy profile is
approved, verify its handshake and session, GET/DELETE and resumption
behavior under that revision. Check actual client limits against the
snapshot envelope. Enter the transport lifecycle from the parent ASGI
lifespan if mounted into the camera application.

Apply ADR-MCP-003's TLS, proxy, exact Host and Origin policy and OAuth
resource-server requirements. Recheck protected resource metadata and
challenge behavior, audience and scope on every protected request, and
non-shared caching for principal-specific results. Complete the remote
release gates before enabling the profile.
[MCP HTTP authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) ·
[Release checklist](../RELEASE_CHECKLIST.md#mcp-release-gates)
