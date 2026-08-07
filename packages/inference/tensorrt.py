"""Reference TensorRT runner for borrowed NV12/NVMM camera frames."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

from packages.camera.models import FrameFormat, GpuFrame, MemoryType

from .cache import EngineCache, EngineCacheMetadata, sha256_file
from .contracts import (
    FrameSpec,
    InferenceResult,
    ShapeProfile,
    TensorOutput,
)
from .errors import (
    EngineBuildError,
    InferenceConfigurationError,
    InferenceDependencyError,
    InferenceError,
)
from .interop import CudaBuffer, CudaStream, InteropRuntime, NativeCudaInterop

OverlayFactory = Callable[[tuple[TensorOutput, ...]], Iterable[object]]


class TensorRTRunner:
    """Build and execute one ONNX image model on a shared CUDA stream.

    The runner is deliberately unaware of YOLO, NMS, segmentation, or label
    formats. It preprocesses an NV12/NVMM camera surface into one NCHW float32
    input tensor, executes TensorRT, and returns every output tensor by name.
    A consuming application may attach model-specific overlays through
    ``overlay_factory``.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        cache_dir: str | Path,
        input_name: str = "images",
        shape_profile: ShapeProfile,
        inference_shape: tuple[int, ...] | None = None,
        precision: str = "fp16",
        workspace_size: int = 1 << 30,
        channel_order: str = "RGB",
        scale: float = 1.0 / 255.0,
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        standard_deviation: tuple[float, float, float] = (1.0, 1.0, 1.0),
        overlay_factory: OverlayFactory | None = None,
        interop: InteropRuntime | None = None,
        tensorrt_module: ModuleType | None = None,
        numpy_module: ModuleType | None = None,
    ) -> None:
        """Store build policy without importing optional dependencies."""
        self._onnx_path = Path(onnx_path)
        self._cache = EngineCache(cache_dir, self._onnx_path.stem)
        self._input_name = input_name
        self._shape_profile = shape_profile
        self._inference_shape = inference_shape or shape_profile.optimum
        self._precision = precision.lower()
        self._workspace_size = workspace_size
        self._channel_order = channel_order.upper()
        self._scale = scale
        self._mean = mean
        self._standard_deviation = standard_deviation
        self._overlay_factory = overlay_factory
        self._interop = interop
        self._trt = tensorrt_module
        self._numpy = numpy_module

        self._frame_spec: FrameSpec | None = None
        self._runtime: Any | None = None
        self._engine: Any | None = None
        self._context: Any | None = None
        self._stream: CudaStream | None = None
        self._buffers: dict[str, CudaBuffer] = {}
        self._tensor_shapes: dict[str, tuple[int, ...]] = {}
        self._tensor_dtypes: dict[str, Any] = {}
        self._output_names: tuple[str, ...] = ()
        self._cache_hit = False
        self._closed = False

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate shape and preprocessing choices before touching CUDA."""
        if not self._input_name:
            raise InferenceConfigurationError("input_name must be non-empty")

        if self._precision not in {"fp16", "fp32"}:
            raise InferenceConfigurationError("precision must be fp16 or fp32")

        if (
            isinstance(self._workspace_size, bool)
            or not isinstance(self._workspace_size, int)
            or self._workspace_size <= 0
        ):
            raise InferenceConfigurationError("workspace_size must be positive")

        if len(self._shape_profile.minimum) != 4:
            raise InferenceConfigurationError("TensorRT image profile must be rank 4")

        for dimensions in (
            self._shape_profile.minimum,
            self._shape_profile.optimum,
            self._shape_profile.maximum,
        ):
            if dimensions[0] != 1 or dimensions[1] != 3:
                raise InferenceConfigurationError(
                    "TensorRT image profile must use NCHW shape (1, 3, H, W)"
                )

        if not self._shape_profile.contains(self._inference_shape):
            raise InferenceConfigurationError(
                "inference_shape must be within the dynamic shape profile"
            )

        if self._inference_shape[0:2] != (1, 3):
            raise InferenceConfigurationError(
                "inference_shape must use NCHW shape (1, 3, H, W)"
            )

        if self._channel_order not in {"RGB", "BGR"}:
            raise InferenceConfigurationError("channel_order must be RGB or BGR")

        for name, values in (
            ("mean", self._mean),
            ("standard_deviation", self._standard_deviation),
        ):
            if len(values) != 3 or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
                raise InferenceConfigurationError(
                    f"{name} must contain three numeric values"
                )

        if any(value == 0 for value in self._standard_deviation):
            raise InferenceConfigurationError(
                "standard_deviation values must be non-zero"
            )

        if not isinstance(self._scale, (int, float)) or isinstance(self._scale, bool):
            raise InferenceConfigurationError("scale must be numeric")

    @property
    def prepared(self) -> bool:
        """Whether engine, execution context, stream, and buffers are ready."""
        return self._context is not None and not self._closed

    @property
    def cache_hit(self) -> bool:
        """Whether ``prepare`` accepted an existing compatible engine."""
        return self._cache_hit

    @property
    def engine_cache_path(self) -> Path:
        """Resolved local engine path for diagnostics."""
        return self._cache.engine_path

    def prepare(self, frame_spec: FrameSpec) -> None:
        """Load or build a compatible engine and allocate its CUDA bindings."""
        if not isinstance(frame_spec, FrameSpec):
            raise TypeError("frame_spec must be a FrameSpec")
        if (
            frame_spec.format is not FrameFormat.NV12_NVMM
            or frame_spec.memory_type is not MemoryType.NVMM
        ):
            raise InferenceConfigurationError(
                "TensorRTRunner requires NV12_NVMM frames in NVMM memory"
            )
        if not self._onnx_path.is_file():
            raise InferenceConfigurationError(
                f"ONNX model does not exist: {self._onnx_path}"
            )

        self.close()
        self._closed = False
        trt = self._load_tensorrt()
        numpy = self._load_numpy()
        interop = self._interop or NativeCudaInterop()
        self._interop = interop

        metadata = EngineCacheMetadata(
            onnx_sha256=sha256_file(self._onnx_path),
            tensorrt_version=str(trt.__version__),
            compute_capability=interop.compute_capability(),
            precision=self._precision,
            input_name=self._input_name,
            shape_profile=self._shape_profile,
        )
        engine_bytes = self._cache.load(metadata)
        self._cache_hit = engine_bytes is not None
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        if engine_bytes is None:
            engine_bytes = self._build_engine(trt)
            self._cache.store(engine_bytes, metadata)

        engine = runtime.deserialize_cuda_engine(engine_bytes)

        if engine is None and self._cache_hit:
            self._cache_hit = False
            engine_bytes = self._build_engine(trt)
            self._cache.store(engine_bytes, metadata)
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        if engine is None:
            raise EngineBuildError("TensorRT could not deserialize engine bytes")

        context = engine.create_execution_context()

        if context is None:
            raise InferenceError("TensorRT could not create an execution context")

        if not context.set_input_shape(self._input_name, self._inference_shape):
            raise InferenceConfigurationError(
                "inference_shape is incompatible with the TensorRT engine"
            )

        missing_shapes = context.infer_shapes()

        if missing_shapes:
            formatted = ", ".join(str(name) for name in missing_shapes)
            raise InferenceConfigurationError(
                f"TensorRT requires additional input shapes: {formatted}"
            )

        stream = interop.create_stream()
        buffers: dict[str, CudaBuffer] = {}
        tensor_shapes: dict[str, tuple[int, ...]] = {}
        tensor_dtypes: dict[str, Any] = {}
        output_names: list[str] = []

        for index in range(engine.num_io_tensors):
            name = str(engine.get_tensor_name(index))
            shape = tuple(int(value) for value in context.get_tensor_shape(name))

            if not shape or any(value < 0 for value in shape):
                raise InferenceConfigurationError(
                    f"TensorRT did not resolve a concrete shape for {name!r}"
                )

            dtype = numpy.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            size = dtype.itemsize

            for dimension in shape:
                size *= dimension

            buffer = interop.allocate(int(size))

            if not context.set_tensor_address(name, buffer.pointer):
                raise InferenceError(f"could not bind TensorRT tensor {name!r}")

            mode = engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT and name != self._input_name:
                raise InferenceConfigurationError(
                    "reference TensorRTRunner supports one image input"
                )

            if mode == trt.TensorIOMode.OUTPUT:
                output_names.append(name)

            buffers[name] = buffer
            tensor_shapes[name] = shape
            tensor_dtypes[name] = dtype

        if not output_names:
            raise InferenceConfigurationError(
                "TensorRT engine must expose at least one output tensor"
            )

        input_dtype = tensor_dtypes.get(self._input_name)

        if input_dtype != numpy.dtype("float32"):
            raise InferenceConfigurationError(
                "NV12 preprocessor currently requires a float32 model input"
            )

        if tensor_shapes.get(self._input_name) != self._inference_shape:
            raise InferenceConfigurationError(
                "TensorRT input shape differs from configured inference_shape"
            )

        self._frame_spec = frame_spec
        self._runtime = runtime
        self._engine = engine
        self._context = context
        self._stream = stream
        self._buffers = buffers
        self._tensor_shapes = tensor_shapes
        self._tensor_dtypes = tensor_dtypes
        self._output_names = tuple(output_names)

    def _load_tensorrt(self) -> ModuleType:
        """Import JetPack's TensorRT binding lazily."""
        if self._trt is None:
            try:
                self._trt = importlib.import_module("tensorrt")

            except ImportError as error:
                raise InferenceDependencyError(
                    "TensorRT Python bindings from JetPack are unavailable"
                ) from error

        return self._trt

    def _load_numpy(self) -> ModuleType:
        """Import NumPy for output tensors, never for camera input pixels."""
        if self._numpy is None:
            try:
                self._numpy = importlib.import_module("numpy")

            except ImportError as error:
                raise InferenceDependencyError(
                    "TensorRTRunner requires NumPy for output tensor metadata"
                ) from error

        return self._numpy

    def _build_engine(self, trt: ModuleType) -> bytes:
        """Parse ONNX and compile one explicit-batch dynamic TensorRT engine."""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flag)
        parser = trt.OnnxParser(network, logger)
        model_bytes = self._onnx_path.read_bytes()

        if not parser.parse(model_bytes):
            errors = "; ".join(
                str(parser.get_error(index)) for index in range(parser.num_errors)
            )
            raise EngineBuildError(f"TensorRT ONNX parser failed: {errors}")

        network_inputs = {
            str(network.get_input(index).name): network.get_input(index)
            for index in range(network.num_inputs)
        }

        if self._input_name not in network_inputs:
            available = ", ".join(sorted(network_inputs)) or "none"
            raise InferenceConfigurationError(
                f"ONNX input {self._input_name!r} not found; available: {available}"
            )

        if len(network_inputs) != 1:
            raise InferenceConfigurationError(
                "reference TensorRTRunner supports one image input"
            )

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self._workspace_size)

        if self._precision == "fp16":
            if not builder.platform_has_fast_fp16:
                raise EngineBuildError("active GPU does not report fast FP16 support")

            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()

        profile.set_shape(
            self._input_name,
            self._shape_profile.minimum,
            self._shape_profile.optimum,
            self._shape_profile.maximum,
        )

        if not profile:
            raise InferenceConfigurationError(
                "TensorRT rejected the dynamic input shape profile"
            )

        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)

        if serialized is None:
            raise EngineBuildError("TensorRT failed to build a serialized engine")

        return bytes(serialized)

    def infer(self, frame: GpuFrame) -> InferenceResult:
        """Preprocess NVMM, execute TensorRT, and return named output tensors."""
        if not isinstance(frame, GpuFrame):
            raise TypeError("frame must be a GpuFrame")

        if not self.prepared:
            raise InferenceError("TensorRTRunner is not prepared")

        if self._frame_spec != FrameSpec.from_gpu_frame(frame):
            raise InferenceConfigurationError(
                "GpuFrame does not match the FrameSpec used by prepare"
            )

        interop = self._interop
        context = self._context
        stream = self._stream
        numpy = self._numpy

        if interop is None or context is None or stream is None or numpy is None:
            raise InferenceError("TensorRTRunner resources are incomplete")

        started_ns = time.monotonic_ns()
        surface = interop.import_frame(frame)
        input_buffer = self._buffers[self._input_name]
        try:
            interop.preprocess_nv12(
                surface,
                input_buffer,
                width=self._inference_shape[3],
                height=self._inference_shape[2],
                channel_order=self._channel_order,
                scale=float(self._scale),
                mean=(
                    float(self._mean[0]),
                    float(self._mean[1]),
                    float(self._mean[2]),
                ),
                standard_deviation=(
                    float(self._standard_deviation[0]),
                    float(self._standard_deviation[1]),
                    float(self._standard_deviation[2]),
                ),
                stream=stream,
            )

            if not context.execute_async_v3(stream.handle):
                raise InferenceError("TensorRT execute_async_v3 failed")

            outputs: list[TensorOutput] = []

            for name in self._output_names:
                raw = self._buffers[name].copy_to_host(stream)
                dtype = self._tensor_dtypes[name]
                shape = self._tensor_shapes[name]
                array = numpy.frombuffer(raw, dtype=dtype).reshape(shape).copy()
                outputs.append(
                    TensorOutput(
                        name=name,
                        shape=shape,
                        dtype=str(dtype),
                        data=array,
                    )
                )

        except Exception:
            stream.synchronize()
            raise

        inference_time_ns = time.monotonic_ns() - started_ns
        output_tuple = tuple(outputs)
        overlays = (
            tuple(self._overlay_factory(output_tuple))
            if self._overlay_factory is not None
            else ()
        )

        return InferenceResult(
            frame_sequence=frame.sequence,
            capture_timestamp_ns=frame.capture_timestamp_ns,
            inference_time_ns=inference_time_ns,
            outputs=output_tuple,
            overlays=overlays,
            metadata={
                "backend": "tensorrt",
                "precision": self._precision,
                "input_name": self._input_name,
                "input_shape": self._inference_shape,
                "engine_cache_hit": self._cache_hit,
                "cuda_stream": stream.handle,
                "preprocessing": {
                    "source": "NV12_NVMM",
                    "layout": "NCHW",
                    "channel_order": self._channel_order,
                    "scale": float(self._scale),
                },
            },
        )

    def close(self) -> None:
        """Synchronize the shared stream and release runner-owned resources."""
        if self._stream is not None:
            self._stream.synchronize()

        self._output_names = ()
        self._tensor_dtypes.clear()
        self._tensor_shapes.clear()
        self._buffers.clear()
        self._stream = None
        self._context = None
        self._engine = None
        self._runtime = None
        self._frame_spec = None
        self._closed = True

    def __enter__(self) -> TensorRTRunner:
        """Return this runner; callers still invoke ``prepare`` explicitly."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release CUDA and TensorRT resources."""
        self.close()
