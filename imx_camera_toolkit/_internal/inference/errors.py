"""Inference and GPU interoperability errors."""


class InferenceError(RuntimeError):
    """Base error raised by inference integrations."""


class InferenceConfigurationError(InferenceError, ValueError):
    """Invalid model, profile, precision, or preprocessing configuration."""


class InferenceDependencyError(InferenceError):
    """An optional TensorRT, CUDA, or native dependency is unavailable."""


class EngineBuildError(InferenceError):
    """TensorRT could not parse or compile the configured ONNX model."""


class CudaInteropError(InferenceError):
    """An NVMM frame could not be imported or processed by CUDA."""
