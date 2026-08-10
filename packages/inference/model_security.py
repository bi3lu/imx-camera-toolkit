"""Ed25519 authenticity checks for deployment-owned ONNX models."""

from __future__ import annotations

import base64
import binascii
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import sha256_file
from .errors import InferenceConfigurationError, InferenceDependencyError


def _validate_trusted_file(path: Path) -> None:
    """Reject replaceable trust inputs before reading their contents."""
    if path.is_symlink():
        raise InferenceConfigurationError(f"trusted file must not be a symlink: {path}")

    try:
        details = path.stat()

    except OSError as error:
        raise InferenceConfigurationError(
            f"trusted file is unavailable: {path}"
        ) from error

    if not stat.S_ISREG(details.st_mode):
        raise InferenceConfigurationError(f"trusted path is not a file: {path}")

    if details.st_uid not in {0, os.geteuid()}:
        raise InferenceConfigurationError(f"trusted file owner is invalid: {path}")

    if stat.S_IMODE(details.st_mode) & 0o022:
        raise InferenceConfigurationError(
            f"trusted file must not be group/world writable: {path}"
        )


def _names(value: object, field_name: str) -> tuple[str, ...]:
    """Parse a bounded list of unique tensor names."""
    if not isinstance(value, list) or not value or len(value) > 1024:
        raise InferenceConfigurationError(
            f"model manifest {field_name} must be a non-empty list"
        )

    if any(not isinstance(name, str) or not name or len(name) > 256 for name in value):
        raise InferenceConfigurationError(
            f"model manifest {field_name} contains an invalid tensor name"
        )

    names = tuple(value)

    if len(names) != len(set(names)):
        raise InferenceConfigurationError(
            f"model manifest {field_name} must contain unique names"
        )
    return names


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Signed model identity and its expected public tensor contract."""

    model_sha256: str
    model_version: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ModelManifest:
        """Parse a strict schema-versioned manifest."""
        expected_keys = {
            "schema_version",
            "model_sha256",
            "model_version",
            "inputs",
            "outputs",
        }
        if set(value) != expected_keys or value.get("schema_version") != 1:
            raise InferenceConfigurationError(
                "model manifest must use the exact schema_version 1 fields"
            )

        digest = value.get("model_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise InferenceConfigurationError(
                "model manifest model_sha256 must be a lowercase SHA-256 digest"
            )

        version = value.get("model_version")
        if not isinstance(version, str) or not version.strip() or len(version) > 128:
            raise InferenceConfigurationError(
                "model manifest model_version must be 1-128 characters"
            )

        return cls(
            model_sha256=digest,
            model_version=version,
            inputs=_names(value.get("inputs"), "inputs"),
            outputs=_names(value.get("outputs"), "outputs"),
        )


def verify_signed_model(
    onnx_path: str | Path,
    public_key_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    signature_path: str | Path | None = None,
) -> ModelManifest:
    """Verify an exact manifest signature and its ONNX SHA-256 digest."""
    model = Path(onnx_path)
    manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else model.with_suffix(".manifest.json")
    )
    signature = (
        Path(signature_path)
        if signature_path is not None
        else model.with_suffix(".manifest.sig")
    )
    public_key = Path(public_key_path)

    for path in (model, manifest, signature, public_key):
        _validate_trusted_file(path)

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as error:
        raise InferenceDependencyError(
            "signed model verification requires the cryptography package"
        ) from error

    try:
        manifest_bytes = manifest.read_bytes()
        document = json.loads(manifest_bytes)
        signature_bytes = base64.b64decode(
            signature.read_bytes().strip(),
            validate=True,
        )
        loaded_key = serialization.load_pem_public_key(public_key.read_bytes())
    except (
        OSError,
        ValueError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        binascii.Error,
    ) as error:
        raise InferenceConfigurationError(
            "could not parse signed model files"
        ) from error

    if not isinstance(document, dict):
        raise InferenceConfigurationError("model manifest must be a JSON mapping")
    parsed = ModelManifest.from_dict(document)
    if not isinstance(loaded_key, Ed25519PublicKey):
        raise InferenceConfigurationError("model public key must be Ed25519")

    try:
        loaded_key.verify(signature_bytes, manifest_bytes)
    except InvalidSignature as error:
        raise InferenceConfigurationError(
            "model manifest signature is invalid"
        ) from error

    if sha256_file(model) != parsed.model_sha256:
        raise InferenceConfigurationError("ONNX digest does not match signed manifest")
    return parsed


__all__ = ["ModelManifest", "verify_signed_model"]
