# Local MCP camera observer

Run the [Python example](../examples/mcp_local_observer.py) on a Jetson with a
working CSI camera to check its status and capabilities and save an optional
JPEG. The client starts the installed `imx-camera mcp` command, which owns one
CPU camera for that connection.

## Set up on Jetson

Use the JetPack system Python so capture can use the JetPack-provided
OpenCV with GStreamer, the Argus/GStreamer plugins, and PyGObject. Install
`uv` and use a checkout containing this guide. The toolkit supports Python
3.10–3.12 and the `mcp` extra pins SDK 2.3.0 with matching types.

From the repository root on the Jetson, create the environment and check the
CLI. If your existing environment already uses JetPack's system interpreter
with system packages enabled, run only the last two commands.

```bash
uv venv --python /usr/bin/python3 --system-site-packages .venv
uv sync --frozen --no-dev --extra mcp
.venv/bin/imx-camera mcp --help
```

If you install from a built wheel, use the same Jetson environment and its
`mcp` extra, then run the example from the corresponding checkout.
[Wheel installation](mcp/README.md#local-stdio-launch)

## Run the example on Jetson

Stop any application using the selected CSI sensor. Use the Argus sensor ID
for your connection, which is `0` on the tested setup. First run the default
observer profile.

```bash
.venv/bin/python examples/mcp_local_observer.py --sensor-id 0
```

The JSON report should show protocol `2026-07-28`, tools `camera_status`
and `camera_capabilities`, and a camera whose state is `ready` and backend is
`cpu`. Capabilities should show `frame_format` as `BGR_CPU` and
`snapshot.configured` as false. The standalone CLI can report the sensor model
and control details as unknown.

Once that command exits, enable snapshots explicitly by choosing a new file
in a directory where you want to store a camera image.

```bash
.venv/bin/python examples/mcp_local_observer.py --sensor-id 0 \
  --snapshot-output snapshot.jpg
file snapshot.jpg
```

The JSON report should now list `camera_snapshot` too. Its `snapshot` object
contains the JPEG dimensions, byte count, frame sequence and age in
milliseconds. The saved file should be a 1280×720 JPEG matching the reported
dimensions. Open it in a local image viewer to verify the captured image.
Use a new filename for another run. The example creates the file exclusively
and leaves existing files intact. Each invocation closes its child
connection and releases the camera when finished.

## Configure another local client

Set the client's executable to the absolute path of the installed
`.venv/bin/imx-camera`. Set its arguments to `mcp`, `--sensor-id`, and `0`
for the default observer profile. Append `--enable-snapshot` to enable the
JPEG source and grant `mcp:snapshot:read`.

Run the trusted client on the Jetson under an account that can access the
camera. It launches the command with private POSIX stdin and stdout pipes
and a separate stderr stream. Stdio carries MCP JSON-RPC and requires no HTTP
listener or network token. Closing the child's stdin requests graceful
shutdown. To reconnect, launch a new child after the old one exits. Keep
requests for the same sensor on the same connection while it is open.
[Local access policy](adr/ADR-MCP-003-threat-model-and-access-policy.md#local-profile)

| Tool | Result |
| --- | --- |
| `camera_status` | Capture lifecycle, backend, recent FPS and drop counters. |
| `camera_capabilities` | Known sensor information, active frame format, declared control support and snapshot limits. |
| `camera_snapshot` | Metadata plus one `image/jpeg` content block with base64 data, when enabled and granted. |

Omit the camera ID or use `camera_id: "primary"`. The standalone JPEG source
uses fixed output dimensions of 1280×720 and quality 65. Width and height
requests are upper bounds. A requested quality is an exact encoder setting.
The example uses `max_age_ms: 2000` and `wait_ms: 500`. Age is elapsed time
since acquisition on the server's monotonic clock. The wait gives the server
time to find a qualifying frame. The JPEG ceiling is 512 KiB before base64
and the serialized response ceiling is 768 KiB.
[Tool inputs, limits and output formats](mcp/README.md#camera-snapshot)

## Troubleshoot

| Symptom | What to check |
| --- | --- |
| The client disconnects during startup. | Read the child's stderr. Verify JetPack packages and capture with the [deployment guide](GPU_CAMERA_YOLO_GUIDE.md), check the Argus sensor ID, and close any other process using the sensor. |
| The CLI reports a missing or incompatible SDK, or installation rejects the interpreter. | Run the example with the same interpreter that installed the toolkit and its `mcp` extra. The server requires `mcp` and `mcp-types` 2.3.0. On the tested Jetson, use JetPack's Python 3.10.12 with access to system packages. |
| Discovery omits `camera_snapshot`, or a direct call to it returns a JSON-RPC lookup error. | Restart with `--enable-snapshot` or use the example's `--snapshot-output` option. |
| An enabled tool returns `permission_denied`. | Check the host's grant for that tool. The default observer scope is `mcp:camera:status`; snapshot access requires `mcp:snapshot:read`. The standalone snapshot flag supplies both the registration and the extra grant. |
| `snapshot_unavailable` | Read `details.reason`. `no_source` indicates a missing or disabled preview source, `no_frame` indicates an empty JPEG slot, and `constraints_unsupported` indicates a request the configured source cannot satisfy. `timing_unavailable` includes the built-in GPU JPEG preview branch, which lacks the required acquisition timing for MCP. |
| `snapshot_stale` or `request_timeout` | With a zero wait, an old JPEG yields `snapshot_stale`. Check frame publication and retry after a fresh frame. A positive wait can end in `request_timeout` if no qualifying image arrives within the deadline. |
| `payload_too_large` | The current JPEG exceeded a byte budget. Use a source configured for a smaller image or lower JPEG quality if available. The standalone command uses a fixed source. |

## Workstation checks and compatibility

On a POSIX workstation, verify CLI and example options and run the synthetic
stdio checks from the repository root.

```bash
uv sync --frozen --extra preview --extra mcp --group dev
.venv/bin/imx-camera mcp --help
.venv/bin/python examples/mcp_local_observer.py --help
.venv/bin/python -m pytest tests/integration/test_mcp_stdio.py \
  tests/integration/test_mcp_camera_tools.py
```

The installed CLI exchanged and decoded a physical CPU JPEG on Jetson Orin
Nano Super with IMX219-77, JetPack 6.2.3 / L4T 36.5.2, and CPython 3.10.12.
Python MCP 2.3.0 and Go MCP SDK 1.8.0 both passed on that setup. Workstation
results cover synthetic capture and separate client interoperability checks.
[Physical Jetson results](mcp/jetson-validation.md) ·
[Workstation and client compatibility](adr/ADR-MCP-004-local-sdk-and-transport.md#workstation-interoperability-evidence)
