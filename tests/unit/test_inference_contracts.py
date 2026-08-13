"""Tests for model-agnostic inference contracts and engine caching."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from imx_camera_toolkit import FrameFormat, MemoryType
from imx_camera_toolkit._internal.inference.cache import (
    EngineCache,
    EngineCacheMetadata,
    sha256_file,
)
from imx_camera_toolkit._internal.inference.interop import NativeCudaInterop
from imx_camera_toolkit.inference import (
    FrameSpec,
    InferenceConfigurationError,
    InferenceResult,
    InferenceRunner,
    ResizeTransform,
    ShapeProfile,
    TensorOutput,
    TensorRTRunner,
    verify_signed_model,
)
from imx_camera_toolkit.testing import mock_gpu_frame


def test_dynamic_shape_profile_accepts_shapes_inside_all_bounds() -> None:
    """Profiles must support non-640 dynamic dimensions."""
    profile = ShapeProfile(
        minimum=(1, 3, 320, 320),
        optimum=(1, 3, 640, 640),
        maximum=(1, 3, 1280, 1280),
    )

    assert profile.contains((1, 3, 720, 1280))
    assert not profile.contains((1, 3, 1920, 1080))
    assert profile.as_dict()["opt"] == [1, 3, 640, 640]


def test_letterbox_transform_preserves_720p_geometry_for_square_model() -> None:
    """A 16:9 source must receive vertical padding rather than distortion."""
    transform = ResizeTransform.calculate((720, 1280), (640, 640), "letterbox")

    assert transform.scale == (0.5, 0.5)
    assert transform.pad_x == 0
    assert transform.pad_y == 140
    assert transform.as_dict() == {
        "resize_mode": "letterbox",
        "scale": [0.5, 0.5],
        "pad_x": 0,
        "pad_y": 140,
        "source_shape": [720, 1280],
        "model_shape": [640, 640],
    }


@pytest.mark.parametrize(
    ("minimum", "optimum", "maximum"),
    [
        ((1, 3, 640), (1, 3, 640, 640), (1, 3, 640, 640)),
        ((1, 3, 640, 640), (1, 3, 320, 320), (1, 3, 1280, 1280)),
        ((1, 3, 0, 320), (1, 3, 640, 640), (1, 3, 1280, 1280)),
    ],
)
def test_dynamic_shape_profile_rejects_invalid_bounds(
    minimum: tuple[int, ...],
    optimum: tuple[int, ...],
    maximum: tuple[int, ...],
) -> None:
    """Ranks, positivity, and min/opt/max ordering are mandatory."""
    with pytest.raises(ValueError):
        ShapeProfile(minimum=minimum, optimum=optimum, maximum=maximum)


def test_inference_result_retains_metadata_timing_and_optional_overlays() -> None:
    """Consumers receive portable metadata without a YOLO result type."""
    metadata = {"backend": "mock"}
    output = TensorOutput(
        name="boxes",
        shape=(1, 1, 4),
        dtype="float32",
        data=object(),
    )
    result = InferenceResult(
        frame_sequence=7,
        frame_timestamp_ns=100,
        capture_timestamp_ns=123,
        inference_time_ns=456,
        outputs=(output,),
        metadata=metadata,
        overlays=({"kind": "box"},),
    )
    metadata["backend"] = "changed"

    assert result.metadata["backend"] == "mock"
    assert result.frame_timestamp_ns == 100
    assert result.outputs == (output,)
    assert result.overlays == ({"kind": "box"},)
    with pytest.raises(TypeError):
        result.metadata["new"] = True  # type: ignore[index]


class _Runner:
    """Minimal structural implementation of the public protocol."""

    def prepare(self, frame_spec: FrameSpec) -> None:
        """Accept a frame spec."""

    def infer(self, frame: object) -> InferenceResult:
        """Return a protocol-shaped result."""
        raise NotImplementedError

    def close(self) -> None:
        """Release no resources."""


def test_inference_runner_is_a_public_structural_protocol() -> None:
    """Consumers need no private capture or TensorRT classes."""
    assert isinstance(_Runner(), InferenceRunner)


def test_tensorrt_runner_implements_protocol_without_loading_optional_modules(
    tmp_path: Path,
) -> None:
    """Constructing a runner remains safe on hosts without CUDA imports."""
    runner = TensorRTRunner(
        tmp_path / "model.onnx",
        cache_dir=tmp_path / "engines",
        shape_profile=ShapeProfile(
            minimum=(1, 3, 320, 320),
            optimum=(1, 3, 640, 640),
            maximum=(1, 3, 1280, 1280),
        ),
    )

    assert isinstance(runner, InferenceRunner)
    assert runner.prepared is False


def test_tensorrt_runner_validates_letterbox_padding_without_cuda(
    tmp_path: Path,
) -> None:
    """Resize policy errors must fail before optional Jetson imports."""
    profile = ShapeProfile(
        minimum=(1, 3, 640, 640),
        optimum=(1, 3, 640, 640),
        maximum=(1, 3, 640, 640),
    )
    runner = TensorRTRunner(
        tmp_path / "model.onnx",
        cache_dir=tmp_path / "engines",
        shape_profile=profile,
        resize_mode="letterbox",
        padding_value=(114, 114, 114),
    )

    assert runner.resize_transform is None
    with pytest.raises(InferenceConfigurationError, match="padding_value"):
        TensorRTRunner(
            tmp_path / "model.onnx",
            cache_dir=tmp_path / "engines",
            shape_profile=profile,
            resize_mode="letterbox",
            padding_value=(256, 114, 114),
        )


def test_native_interop_passes_checked_gst_buffer_object_to_cpp() -> None:
    """Python must not map pixels or convert the Gst.Buffer to an array."""
    calls: list[tuple[object, object, int, int]] = []
    native = ModuleType("fake_cuda_interop")
    context = object()
    native.CudaPrimaryContext = lambda ordinal: context  # type: ignore[attr-defined]
    native.CudaContextGuard = lambda owner: nullcontext()  # type: ignore[attr-defined]
    native.NvmmSurface = lambda owner, payload, width, height: calls.append(  # type: ignore[attr-defined]
        (owner, payload, width, height)
    )
    payload_type = type("Buffer", (), {"__module__": "gi.overrides.Gst"})
    gst = ModuleType("gi.repository.Gst")
    gst.Buffer = payload_type  # type: ignore[attr-defined]
    payload = payload_type()
    frame = mock_gpu_frame(payload, width=1920, height=1080)

    surface = NativeCudaInterop(native, gstreamer_module=gst).import_frame(frame)

    assert surface is None
    assert calls == [(context, payload, 1920, 1080)]


def test_native_interop_retains_and_activates_primary_cuda_context() -> None:
    """All native resources must share the retained device primary context."""
    native = ModuleType("fake_cuda_interop")
    context = object()
    calls: list[tuple[str, object]] = []

    def activate(owner: object) -> AbstractContextManager[None]:
        calls.append(("activate", owner))
        return nullcontext()

    native.CudaPrimaryContext = lambda ordinal: context  # type: ignore[attr-defined]
    native.CudaContextGuard = activate  # type: ignore[attr-defined]
    native.CudaStream = lambda owner: calls.append(("stream", owner))  # type: ignore[attr-defined]
    native.DeviceBuffer = lambda owner, size: calls.append(  # type: ignore[attr-defined]
        (f"buffer:{size}", owner)
    )
    interop = NativeCudaInterop(native)

    with interop.activate():
        interop.create_stream()
        interop.allocate(16)

    assert calls == [
        ("activate", context),
        ("stream", context),
        ("buffer:16", context),
    ]


class _TrackingInterop:
    """Host fake that exposes whether a calling thread entered activation."""

    def __init__(self) -> None:
        self.active_depth = 0
        self.activations = 0

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Track nested runner context scopes."""
        self.active_depth += 1
        self.activations += 1

        try:
            yield

        finally:
            self.active_depth -= 1


def test_tensorrt_prepare_and_close_run_inside_interop_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TensorRT creation and destruction must use a scoped CUDA context."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    interop = _TrackingInterop()
    runner = TensorRTRunner(
        model_path,
        cache_dir=tmp_path / "engines",
        input_name="input",
        shape_profile=ShapeProfile(
            minimum=(1, 3, 1, 1),
            optimum=(1, 3, 1, 1),
            maximum=(1, 3, 1, 1),
        ),
        inference_shape=(1, 3, 1, 1),
        interop=interop,  # type: ignore[arg-type]
        tensorrt_module=ModuleType("fake_tensorrt"),
        numpy_module=ModuleType("fake_numpy"),
    )
    operations: list[str] = []

    def prepare_activated(frame_spec: FrameSpec) -> None:
        assert interop.active_depth > 0
        operations.append("prepare")

    def close_activated() -> None:
        assert interop.active_depth > 0
        operations.append("close")

    monkeypatch.setattr(runner, "_prepare_activated", prepare_activated)
    runner.prepare(FrameSpec(1, 1, FrameFormat.NV12_NVMM, MemoryType.NVMM))
    monkeypatch.setattr(runner, "_close_cuda_resources", close_activated)
    runner.close()

    assert operations == ["prepare", "close"]
    assert interop.activations == 2
    assert interop.active_depth == 0


def test_tensorrt_runner_auto_selects_the_only_onnx_input(tmp_path: Path) -> None:
    """A model-neutral runner must not assume that its input is named images."""
    runner = TensorRTRunner(
        tmp_path / "model.onnx",
        cache_dir=tmp_path / "engines",
        input_name=None,
        shape_profile=ShapeProfile(
            minimum=(1, 3, 320, 320),
            optimum=(1, 3, 640, 640),
            maximum=(1, 3, 1280, 1280),
        ),
    )

    assert runner._select_input_name(("input",)) == "input"
    with pytest.raises(InferenceConfigurationError, match="exactly one"):
        runner._select_input_name(("image", "scale"))


def test_native_source_uses_egl_cuda_without_host_input_upload() -> None:
    """The reference implementation must not hide a BGR/host staging path."""
    source_path = Path(__file__).parents[2] / "native" / "src" / "cuda_interop.cu"
    source = source_path.read_text("utf-8")

    assert "pyg_boxed_get" in source
    assert "NvBufSurfaceMapEglImage" in source
    assert "cuGraphicsEGLRegisterImage" in source
    assert "nv12_to_nchw_kernel" in source
    assert "cudaMemcpyHostToDevice" not in source
    assert "numpy" not in source.lower()


def test_engine_cache_rejects_every_platform_specific_mismatch(tmp_path: Path) -> None:
    """A plan is reusable only for an exact model/GPU/TensorRT profile."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model-a")
    profile = ShapeProfile(
        minimum=(1, 3, 320, 320),
        optimum=(1, 3, 640, 640),
        maximum=(1, 3, 1280, 1280),
    )
    expected = EngineCacheMetadata(
        onnx_sha256=sha256_file(model_path),
        tensorrt_version="10.3.0",
        compute_capability=(8, 7),
        precision="fp16",
        input_name="images",
        shape_profile=profile,
    )
    cache = EngineCache(tmp_path, "model")
    cache.store(b"serialized-engine", expected)

    assert cache.load(expected) == b"serialized-engine"
    assert cache.load(replace(expected, tensorrt_version="10.4.0")) is None
    assert cache.load(replace(expected, compute_capability=(8, 6))) is None
    assert cache.load(replace(expected, precision="fp32")) is None
    assert cache.load(replace(expected, onnx_sha256="0" * 64)) is None
    assert (
        cache.load(
            replace(
                expected,
                shape_profile=ShapeProfile(
                    minimum=(1, 3, 640, 640),
                    optimum=(1, 3, 960, 960),
                    maximum=(1, 3, 1280, 1280),
                ),
            )
        )
        is None
    )


def test_engine_cache_discards_tampered_engine_bytes(tmp_path: Path) -> None:
    """A digest mismatch must remove both rebuildable cache artifacts."""
    profile = ShapeProfile(
        minimum=(1, 3, 320, 320),
        optimum=(1, 3, 640, 640),
        maximum=(1, 3, 1280, 1280),
    )
    expected = EngineCacheMetadata(
        onnx_sha256="1" * 64,
        tensorrt_version="10.3.0",
        compute_capability=(8, 7),
        precision="fp16",
        input_name="images",
        shape_profile=profile,
    )
    cache = EngineCache(tmp_path, "model")
    cache.store(b"trusted-engine", expected)
    cache.engine_path.write_bytes(b"replaced-engine")

    assert cache.load(expected) is None
    assert not cache.engine_path.exists()
    assert not cache.metadata_path.exists()


def test_ed25519_manifest_authenticates_model_and_tensor_contract(
    tmp_path: Path,
) -> None:
    """Only an exact signed manifest and matching ONNX digest are accepted."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"trusted-model")
    manifest_path = tmp_path / "model.manifest.json"
    signature_path = tmp_path / "model.manifest.sig"
    public_key_path = tmp_path / "model-public.pem"
    manifest_bytes = (
        json.dumps(
            {
                "schema_version": 1,
                "model_sha256": sha256_file(model_path),
                "model_version": "2026.08.1",
                "inputs": ["images"],
                "outputs": ["boxes", "scores"],
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    private_key = Ed25519PrivateKey.generate()
    manifest_path.write_bytes(manifest_bytes)
    signature_path.write_bytes(base64.b64encode(private_key.sign(manifest_bytes)))
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    for path in (model_path, manifest_path, signature_path, public_key_path):
        path.chmod(0o644)

    manifest = verify_signed_model(model_path, public_key_path)
    assert manifest.model_version == "2026.08.1"
    assert manifest.inputs == ("images",)
    assert manifest.outputs == ("boxes", "scores")

    model_path.write_bytes(b"tampered-model")
    with pytest.raises(InferenceConfigurationError, match="digest"):
        verify_signed_model(model_path, public_key_path)


def test_tensorrt_runner_requires_a_key_for_signed_model_mode(tmp_path: Path) -> None:
    """Required authenticity cannot silently degrade to digest-only caching."""
    with pytest.raises(InferenceConfigurationError, match="public_key_path"):
        TensorRTRunner(
            tmp_path / "model.onnx",
            cache_dir=tmp_path / "engines",
            shape_profile=ShapeProfile(
                minimum=(1, 3, 320, 320),
                optimum=(1, 3, 640, 640),
                maximum=(1, 3, 1280, 1280),
            ),
            require_signed_model=True,
        )


def test_frame_spec_preserves_gpu_capture_identity() -> None:
    """Preparation explicitly distinguishes NVMM from BGR/CPU frames."""
    spec = FrameSpec(
        width=1920,
        height=1080,
        format=FrameFormat.NV12_NVMM,
        memory_type=MemoryType.NVMM,
    )

    assert spec.width == 1920
    assert spec.format is FrameFormat.NV12_NVMM
