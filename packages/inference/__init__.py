"""Model-agnostic inference contracts and optional TensorRT integration."""

from .contracts import (
    FrameSpec,
    InferenceResult,
    InferenceRunner,
    ShapeProfile,
    TensorOutput,
)
from .errors import (
    CudaInteropError,
    EngineBuildError,
    InferenceConfigurationError,
    InferenceDependencyError,
    InferenceError,
)
from .model_security import ModelManifest, verify_signed_model
from .tensorrt import TensorRTRunner

__all__ = [
    "CudaInteropError",
    "EngineBuildError",
    "FrameSpec",
    "InferenceConfigurationError",
    "InferenceDependencyError",
    "InferenceError",
    "InferenceResult",
    "InferenceRunner",
    "ModelManifest",
    "ShapeProfile",
    "TensorOutput",
    "TensorRTRunner",
    "verify_signed_model",
]
