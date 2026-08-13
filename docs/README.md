# Documentation

The documentation is organized by task. Start with the deployment guide when
bringing up a camera or inference model on a Jetson, and use the architecture
guide when choosing between CPU, GPU, MJPEG, WebRTC, and HLS paths.

| Document | Audience | Purpose |
| --- | --- | --- |
| [GPU Camera and YOLO deployment guide](GPU_CAMERA_YOLO_GUIDE.md) | Application and deployment engineers | Prepare JetPack 6.2.2, validate an IMX sensor, export YOLO, build CUDA interop, run the GPU-first example, and deploy it in field mode. |
| [CPU, GPU, and browser mode guide](GPU_PATH_GUIDE.md) | Developers and architects | Select the correct memory path and understand TensorRT cache, browser transport, and benchmark contracts. |
| [CPU/GPU release checklist](RELEASE_CHECKLIST.md) | Maintainers | Run host, Jetson, packaging, security, and release gates. |

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
