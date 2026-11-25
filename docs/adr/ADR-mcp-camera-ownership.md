# MCP ADR — Camera ownership and host lifecycle

**Status** Accepted for v0.9.0.

## One primary camera per host

The v0.9.x single-camera profile binds one application-selected source to
`primary`. All clients and in-flight calls on a host share its injected read
and snapshot adapters. The application binds
both adapters to the same camera and supplies only approved diagnostic
providers. There is no camera registry or per-client capture. The local
stdio profile runs the host in the capture application's process.
[Local SDK and transport](ADR-mcp-local-sdk-and-transport.md)

## Startup and readiness

The application constructs the camera, validates launch configuration and
grants, and enters the MCP host once before serving requests. In the default
borrowed mode, the application starts and stops its capture and producers.
The parent lifespan starts those sources before entering the MCP host and
exits the host before stopping the producers and camera. An application that
also runs the existing MJPEG API uses `manage_camera=False` there if the
parent owns capture. Any future mounted MCP transport must enter its
transport lifecycle from that same parent lifespan.

An explicit standalone launcher may transfer one unstarted camera to its
host. The launcher constructs the camera and injects both adapters. The
host starts it once on entry, attempts cleanup if startup fails partway,
and stops it after its reads drain. The target process must be the only
owner of that physical sensor. A second stdio subprocess aimed at the same
sensor is outside the launch profile.
[Packaged CLI entry point](../mcp/README.md#local-stdio-launch)

Dispatch becomes available after successful host entry. Camera readiness is
reported from the existing capture statistics and requires a valid frame
from the current capture run. Startup before publication and backend recovery
are visible to callers. Recovery remains the responsibility of the capture
owner. Opening or closing a client connection has no capture lifecycle
effect. A replacement borrowed host can use the application's still-running
camera. A standalone stdio relaunch creates a fresh application process.
[Camera status contract](../mcp/README.md#camera-status)

## Shutdown and work tracking

One host lives on one event loop. Closing it refuses new work, cancels
outstanding snapshot waits, and drains their tasks before releasing owned
capture or returning borrowed-source cleanup to the parent. Concurrent
close calls share the same cleanup. Cancelling one caller releases its
response while the host tracks the underlying task until it exits.
Cancelling a close caller defers that cancellation until cleanup completes.

Diagnostics are cached and nonblocking. Snapshot reads use the adapter's
cooperative wait with a one-second deadline and have no executor job to drain.
The stdio transport stops SDK dispatch on disconnect and bounds a blocked
response write to one second. It exits before the host drains its waits and
releases owned capture. [Local transport](ADR-mcp-local-sdk-and-transport.md#hosting-and-lifecycle)

## Verification

On a workstation, run the targeted lifecycle suite from the repository root.

```bash
uv run --frozen --extra preview --group dev python -m pytest tests/unit/test_mcp_host.py --no-cov
```

The Jetson run verified one Argus session per capture run and a clean
standalone shutdown. The embedded fixture overlapped two requests on one
stdio connection, simulated a pending wait on live capture for cancellation
and disconnect, and verified continued publication after the borrowed MCP
host closed. The application then stopped capture and verified fresh
publication after restart. [Physical ownership evidence](../mcp/jetson-validation.md#runtime-protocol-and-camera-behavior)
