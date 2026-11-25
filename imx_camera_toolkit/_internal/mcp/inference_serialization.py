"""Project application-approved inference data into a bounded JSON payload."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sized
from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias, TypedDict

from ..inference.contracts import InferenceResult
from .contracts import MAX_SAFE_INTEGER
from .errors import MCPError

JSONValue: TypeAlias = (
    "None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]"
)
InferenceMapper: TypeAlias = Callable[[InferenceResult], dict[str, JSONValue]]

MAX_PAYLOAD_BYTES = 16 * 1024
MAX_JSON_DEPTH = 8
MAX_COLLECTION_ITEMS = 128
MAX_JSON_VALUES = 1024


class InferencePayload(TypedDict):
    """Application-defined fields for the latest-result envelope."""

    source_id: str
    model_id: str
    payload_schema: str
    payload: dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class InferencePayloadSerializer:
    """Bind one cached source and model to an explicitly versioned payload.

    The application supplies a short, bounded callback that projects only
    approved data available on the host. It returns an owned JSON object and
    must finish independently of request cancellation or host shutdown.
    """

    source_id: str
    model_id: str
    payload_schema: str
    mapper: InferenceMapper

    def __post_init__(self) -> None:
        """Reject missing or invalid descriptors before the host starts."""
        for value, limit in (
            (self.source_id, 128),
            (self.model_id, 128),
            (self.payload_schema, 256),
        ):
            if (
                type(value) is not str
                or not 1 <= len(value) <= limit
                or any(not "!" <= char <= "~" for char in value)
            ):
                raise ValueError("invalid inference payload descriptor")

        if not callable(self.mapper):
            raise ValueError("an inference payload mapper is required")

    def serialize(self, result: InferenceResult) -> InferencePayload:
        """Run the callback and validate its detached projection.

        Raises:
            MCPError: If the mapper fails or produces invalid or oversized JSON.
        """
        try:
            mapped = self.mapper(result)

        except Exception:
            mapped = None

        if type(mapped) is not dict:
            raise MCPError("internal_error")

        budget = _JSONBudget()
        payload = budget.copy_object(mapped, depth=1)

        encoded = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        )
        if len(encoded) > MAX_PAYLOAD_BYTES:
            raise MCPError("payload_too_large")

        return InferencePayload(
            source_id=self.source_id,
            model_id=self.model_id,
            payload_schema=self.payload_schema,
            payload=payload,
        )


class _JSONBudget:
    """Bound validation work and detach containers before measuring wire bytes."""

    def __init__(self) -> None:
        """Start with an empty traversal and the full payload budget."""
        self._values_left = MAX_JSON_VALUES
        self._characters_left = MAX_PAYLOAD_BYTES
        self._ancestors: set[int] = set()

    def copy_object(
        self, value: Mapping[str, object], depth: int
    ) -> dict[str, JSONValue]:
        """Validate and copy a plain object with string keys."""
        self._enter(value, depth)

        try:
            return {
                self._string(key): self._copy(item, depth + 1)
                for key, item in value.items()
            }

        finally:
            self._ancestors.remove(id(value))

    def _copy(self, value: object, depth: int) -> JSONValue:
        """Accept only plain JSON primitives and containers."""
        if type(value) is dict:
            return self.copy_object(value, depth)

        if type(value) is list:
            self._enter(value, depth)

            try:
                return [self._copy(item, depth + 1) for item in value]

            finally:
                self._ancestors.remove(id(value))

        self._charge_value()

        if value is None or type(value) is bool:
            return value

        if type(value) is int and -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER:
            return value

        if type(value) is float and isfinite(value):
            return value

        if type(value) is str:
            return self._string(value)

        raise MCPError("internal_error")

    def _enter(self, value: Sized, depth: int) -> None:
        """Track cycles and charge one bounded container to the traversal."""
        if id(value) in self._ancestors:
            raise MCPError("internal_error")

        self._charge_value()

        if depth > MAX_JSON_DEPTH or len(value) > MAX_COLLECTION_ITEMS:
            raise MCPError("payload_too_large")

        self._ancestors.add(id(value))

    def _charge_value(self) -> None:
        """Count containers and leaves against the aggregate limit."""
        self._values_left -= 1

        if self._values_left < 0:
            raise MCPError("payload_too_large")

    def _string(self, value: object) -> str:
        """Bound aggregate string work and reject invalid Unicode."""
        if type(value) is not str:
            raise MCPError("internal_error")

        self._characters_left -= len(value)

        if self._characters_left < 0:
            raise MCPError("payload_too_large")

        try:
            value.encode("utf-8")

        except UnicodeEncodeError:
            raise MCPError("internal_error") from None

        return value
