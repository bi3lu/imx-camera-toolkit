# Documentation

The documentation is organized by task. Start with the deployment guide when
bringing up a camera or inference model on a Jetson, and use the architecture
guide when choosing between CPU, GPU, MJPEG, WebRTC, and HLS paths.

| Document | Audience | Purpose |
| --- | --- | --- |
| [GPU Camera and YOLO deployment guide](GPU_CAMERA_YOLO_GUIDE.md) | Application and deployment engineers | Prepare JetPack 6.2.3, validate an IMX sensor, export YOLO, build CUDA interop, run the GPU-first example, and deploy it in field mode. |
| [CPU, GPU, and browser mode guide](GPU_PATH_GUIDE.md) | Developers and architects | Select the correct memory path and understand TensorRT cache, browser transport, and benchmark contracts. |
| [CPU/GPU release checklist](RELEASE_CHECKLIST.md) | Maintainers | Run host, Jetson, packaging, security, and release gates. |
| [MCP Python SDK runtime ADR](adr/ADR-MCP-001-python-sdk-runtime.md) | Maintainers | Reproduce the SDK comparison and validate the proposed runtime on Jetson. |
| [MCP tool contracts](mcp/README.md) | Adapter and transport developers | Implement the camera tools against shared schemas and fixtures. |
| [Local MCP installation and launch](mcp/README.md#local-stdio-launch) | Client integrators | Install the optional runtime and configure the local stdio command. |
| [MCP camera ownership ADR](adr/ADR-MCP-002-camera-ownership.md) | Application and transport developers | Bind one primary camera and coordinate host readiness, cancellation, and shutdown. |
| [MCP threat model and access policy](adr/ADR-MCP-003-threat-model-and-access-policy.md) | Maintainers and integrators | Review local and remote trust boundaries, tool scopes, and security acceptance tests. |
| [Local MCP SDK and transport ADR](adr/ADR-MCP-004-local-sdk-and-transport.md) | Maintainers and client integrators | Reproduce the versioned stdio profile, protocol errors, snapshot exchange and shutdown behavior. |
| [Bounded MCP JPEG snapshots](adr/ADR-MCP-006-bounded-jpeg-snapshots.md) | Maintainers and integrators | Review fixed preview sources, acquisition timing, and snapshot verification. |

Component-specific details remain next to their implementations:

- [camera capture](../imx_camera_toolkit/_internal/camera/README.md);
- [GPU inference](../imx_camera_toolkit/_internal/inference/README.md);
- [latest-frame consumers](../imx_camera_toolkit/_internal/consumers/README.md);
- [production WebRTC/HLS preview](../imx_camera_toolkit/_internal/production_preview/README.md);
- [camera control](../imx_camera_toolkit/_internal/camera_control/README.md);
- [HTTP API](../imx_camera_toolkit/_internal/api/README.md).

The root [README](../README.md) is the project overview and public API
reference. It should link to detailed guides instead of duplicating their
step-by-step deployment procedures.
