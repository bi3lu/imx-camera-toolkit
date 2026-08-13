# Documentation guidance

Documentation is for library users, Jetson operators, and contributors. Keep
it executable, specific, and honest about what has been validated.

## Organization

- `README.md` at the repository root is the overview and public API entry.
- `docs/README.md` is the task-oriented documentation index.
- `GPU_PATH_GUIDE.md` explains architecture and mode selection.
- `GPU_CAMERA_YOLO_GUIDE.md` is the end-to-end Jetson deployment procedure.
- `RELEASE_CHECKLIST.md` defines host, hardware, and release evidence.
- Component READMEs next to `_internal` code explain implementation contracts.

Link to the canonical detailed section instead of copying long procedures.
Use relative links for repository files and verify every local target.

## Content rules

- Write in English and use the exact public import path shown by the code.
- Mark hardware rows `tested` only after running the relevant physical Jetson
  gate on the stated sensor, resolution, FPS, and JetPack stack.
- Separate workstation model export from Jetson engine build/run commands.
- State memory domains precisely: `copy=False` on `Camera` is shared CPU BGR,
  not NVMM; `GpuCamera` is borrowed NV12/NVMM.
- State ownership and shutdown order in examples.
- Do not recommend installing PyPI CUDA/TensorRT/OpenCV over JetPack packages.
- Never include real tokens, private keys, TURN credentials, device addresses,
  or cached TensorRT engines.
- Security deployment examples use loopback behind a trusted TLS proxy or
  direct TLS with an exact Host allowlist; never recommend wildcard field mode.

Before claiming a command works, compare it with `pyproject.toml`, CLI parser
options, test environment variables, and current public exports.
