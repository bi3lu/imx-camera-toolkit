"""Build the optional Jetson CUDA interoperability extension in place."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .errors import InferenceDependencyError


def build_cuda_interop(
    *,
    build_dir: str | Path | None = None,
    cuda_architecture: str = "87",
) -> Path:
    """Configure and build the pybind11 extension for the active Jetson."""
    try:
        pybind11 = importlib.import_module("pybind11")

    except ImportError as error:
        raise InferenceDependencyError(
            "pybind11 is unavailable; install imx-camera-toolkit[tensorrt]"
        ) from error

    source_root = Path(__file__).resolve().parents[3]
    native_source = source_root / "native"
    output_directory = Path(__file__).resolve().parent
    cuda_compiler = Path("/usr/local/cuda/bin/nvcc")
    pygobject_header = Path("/usr/include/pygobject-3.0/pygobject.h")

    if not native_source.joinpath("CMakeLists.txt").is_file():
        raise InferenceDependencyError(
            f"native CUDA interop sources are unavailable at {native_source}"
        )

    if not cuda_compiler.is_file():
        raise InferenceDependencyError(
            "CUDA compiler is unavailable at /usr/local/cuda/bin/nvcc"
        )
    if not pygobject_header.is_file():
        raise InferenceDependencyError(
            "PyGObject C headers are unavailable; install python-gi-dev"
        )

    temporary: tempfile.TemporaryDirectory[str] | None = None

    if build_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="imx-camera-cuda-")
        resolved_build_dir = Path(temporary.name)

    else:
        resolved_build_dir = Path(build_dir)

    try:
        subprocess.run(
            [
                "cmake",
                "-S",
                str(native_source),
                "-B",
                str(resolved_build_dir),
                "-G",
                "Ninja",
                f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
                f"-DCMAKE_CUDA_COMPILER={cuda_compiler}",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architecture}",
                f"-DIMX_CAMERA_INTEROP_OUTPUT_DIRECTORY={output_directory}",
            ],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(resolved_build_dir)],
            check=True,
        )

    except FileNotFoundError as error:
        raise InferenceDependencyError(
            "cmake and Ninja are required to build CUDA interop"
        ) from error

    except subprocess.CalledProcessError as error:
        raise InferenceDependencyError(
            f"CUDA interop build failed with exit code {error.returncode}"
        ) from error

    finally:
        if temporary is not None:
            temporary.cleanup()

    candidates = sorted(output_directory.glob("_cuda_interop*.so"))

    if not candidates:
        raise InferenceDependencyError(
            "CUDA interop build completed without producing a Python module"
        )

    return candidates[-1]


def main(arguments: Sequence[str] | None = None) -> int:
    """Build the native module from the project entry point."""
    parser = argparse.ArgumentParser(
        description="Build Jetson NvBufSurface/CUDA interoperability",
    )
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--cuda-architecture", default="87")
    options = parser.parse_args(arguments)
    module_path = build_cuda_interop(
        build_dir=options.build_dir,
        cuda_architecture=options.cuda_architecture,
    )
    print(module_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
