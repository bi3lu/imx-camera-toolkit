"""Validated local cache for platform-specific TensorRT engines."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .contracts import ShapeProfile
from .errors import InferenceConfigurationError


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without retaining the ONNX model."""
    digest = hashlib.sha256()

    with path.open("rb") as model_file:
        while chunk := model_file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EngineCacheMetadata:
    """Compatibility identity required before deserializing an engine."""

    onnx_sha256: str
    tensorrt_version: str
    compute_capability: tuple[int, int]
    precision: str
    input_name: str
    shape_profile: ShapeProfile
    engine_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate all fields used to accept or reject cached bytes."""
        if len(self.onnx_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.onnx_sha256
        ):
            raise InferenceConfigurationError(
                "onnx_sha256 must be a lowercase SHA-256 digest"
            )

        if not self.tensorrt_version:
            raise InferenceConfigurationError("tensorrt_version must be non-empty")

        if len(self.compute_capability) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.compute_capability
        ):
            raise InferenceConfigurationError(
                "compute_capability must contain non-negative major/minor integers"
            )

        if self.precision not in {"fp16", "fp32"}:
            raise InferenceConfigurationError("precision must be fp16 or fp32")

        if not self.input_name:
            raise InferenceConfigurationError("input_name must be non-empty")

        if self.engine_sha256 is not None and (
            len(self.engine_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.engine_sha256
            )
        ):
            raise InferenceConfigurationError(
                "engine_sha256 must be a lowercase SHA-256 digest or None"
            )

    def as_dict(self) -> dict[str, object]:
        """Return deterministic JSON-compatible metadata."""
        return {
            "schema_version": 2,
            "onnx_sha256": self.onnx_sha256,
            "tensorrt_version": self.tensorrt_version,
            "compute_capability": list(self.compute_capability),
            "precision": self.precision,
            "input_name": self.input_name,
            "shape_profile": self.shape_profile.as_dict(),
            "engine_sha256": self.engine_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EngineCacheMetadata:
        """Parse cache metadata while rejecting unknown schema versions."""
        if value.get("schema_version") != 2:
            raise InferenceConfigurationError("unsupported engine metadata schema")

        capability = value.get("compute_capability")
        profile = value.get("shape_profile")

        if not isinstance(capability, list) or not isinstance(profile, dict):
            raise InferenceConfigurationError("invalid engine metadata structure")

        for bound in ("min", "opt", "max"):
            if not isinstance(profile.get(bound), list):
                raise InferenceConfigurationError(
                    f"shape_profile.{bound} must be a list"
                )

        try:
            return cls(
                onnx_sha256=str(value["onnx_sha256"]),
                tensorrt_version=str(value["tensorrt_version"]),
                compute_capability=tuple(int(item) for item in capability),  # type: ignore[arg-type]
                precision=str(value["precision"]),
                input_name=str(value["input_name"]),
                shape_profile=ShapeProfile(
                    minimum=tuple(int(item) for item in profile["min"]),
                    optimum=tuple(int(item) for item in profile["opt"]),
                    maximum=tuple(int(item) for item in profile["max"]),
                ),
                engine_sha256=(
                    None
                    if value.get("engine_sha256") is None
                    else str(value["engine_sha256"])
                ),
            )

        except (KeyError, TypeError, ValueError) as error:
            raise InferenceConfigurationError(
                "invalid engine cache metadata"
            ) from error


class EngineCache:
    """Store one engine beside exact compatibility metadata."""

    def __init__(self, directory: str | Path, model_name: str) -> None:
        """Resolve deterministic cache paths without creating files yet."""
        if not isinstance(model_name, str) or not model_name.strip():
            raise InferenceConfigurationError("model_name must be non-empty")

        safe_name = "".join(
            character if character.isalnum() or character in "-." else "_"
            for character in model_name
        ).strip(".")

        if not safe_name:
            raise InferenceConfigurationError("model_name has no safe characters")

        self.directory = Path(directory)
        self.engine_path = self.directory / f"{safe_name}.engine"
        self.metadata_path = self.directory / f"{safe_name}.engine.json"

    def load(self, expected: EngineCacheMetadata) -> bytes | None:
        """Return engine bytes only when every metadata field matches."""
        try:
            self._validate_directory()

        except (FileNotFoundError, OSError, InferenceConfigurationError):
            return None

        try:
            self._validate_cache_file(self.metadata_path)
            self._validate_cache_file(self.engine_path)
            raw_metadata = json.loads(self.metadata_path.read_text("utf-8"))
            if not isinstance(raw_metadata, dict):
                raise InferenceConfigurationError("engine metadata must be a mapping")
            actual = EngineCacheMetadata.from_dict(raw_metadata)
            engine = self.engine_path.read_bytes()

        except FileNotFoundError:
            if self.engine_path.exists() or self.metadata_path.exists():
                self.discard()
            return None

        except (OSError, json.JSONDecodeError, InferenceConfigurationError):
            self.discard()
            return None

        compatible = replace(actual, engine_sha256=None) == replace(
            expected,
            engine_sha256=None,
        )

        digest_matches = actual.engine_sha256 is not None and secrets.compare_digest(
            hashlib.sha256(engine).hexdigest(),
            actual.engine_sha256,
        )

        if not compatible or not engine or not digest_matches:
            self.discard()
            return None

        return engine

    def store(self, engine: bytes, metadata: EngineCacheMetadata) -> None:
        """Atomically replace engine and metadata files in the local cache."""
        if not isinstance(engine, bytes) or not engine:
            raise InferenceConfigurationError("engine cache payload must be bytes")

        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._validate_directory()
        stored_metadata = replace(
            metadata,
            engine_sha256=hashlib.sha256(engine).hexdigest(),
        )
        metadata_bytes = (
            json.dumps(stored_metadata.as_dict(), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        self._atomic_write(self.engine_path, engine)
        self._atomic_write(self.metadata_path, metadata_bytes)

    def discard(self) -> None:
        """Remove untrusted rebuildable cache artifacts."""
        self.engine_path.unlink(missing_ok=True)
        self.metadata_path.unlink(missing_ok=True)

    def _validate_directory(self) -> None:
        """Require cache ownership and deployment-safe directory permissions."""
        if self.directory.is_symlink():
            raise InferenceConfigurationError("engine cache directory is a symlink")

        details = self.directory.stat()

        if not stat.S_ISDIR(details.st_mode):
            raise InferenceConfigurationError("engine cache path is not a directory")

        if details.st_uid not in {0, os.geteuid()}:
            raise InferenceConfigurationError("engine cache owner is invalid")

        if stat.S_IMODE(details.st_mode) not in {0o700, 0o750}:
            raise InferenceConfigurationError(
                "engine cache directory permissions must be 0700 or 0750"
            )

    @staticmethod
    def _validate_cache_file(path: Path) -> None:
        """Require regular owner-controlled cache files before deserialization."""
        if path.is_symlink():
            raise InferenceConfigurationError("engine cache file is a symlink")

        details = path.stat()

        if not stat.S_ISREG(details.st_mode):
            raise InferenceConfigurationError("engine cache entry is not a file")

        if details.st_uid not in {0, os.geteuid()}:
            raise InferenceConfigurationError("engine cache file owner is invalid")

        if stat.S_IMODE(details.st_mode) not in {0o600, 0o640}:
            raise InferenceConfigurationError(
                "engine cache file permissions must be 0600 or 0640"
            )

    def _atomic_write(self, destination: Path, payload: bytes) -> None:
        """Write and fsync a temporary file before an atomic replacement."""
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            dir=self.directory,
        )
        temporary_path = Path(temporary_name)

        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)

        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
