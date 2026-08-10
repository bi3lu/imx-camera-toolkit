"""Unit tests for fail-closed HTTP deployment security."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException
from fastapi.security import SecurityScopes
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import Message, Receive, Scope, Send

from packages.api.api import create_app
from packages.api.security import (
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityConfig,
    SecurityHeadersMiddleware,
    build_authorizer,
    token_sha256,
)
from packages.testing.mock_camera import MockCamera


def _security_config(**kwargs: object) -> SecurityConfig:
    """Build a field policy with separate stream and admin tokens."""
    return SecurityConfig(
        field_mode=True,
        token_grants=(
            (token_sha256("stream-token"), frozenset({"stream:read"})),
            (token_sha256("admin-token"), frozenset({"admin"})),
        ),
        allowed_hosts=("camera.example",),
        require_https=True,
        **kwargs,  # type: ignore[arg-type]
    )


def _endpoint(application: Any, path: str) -> Callable[..., Any]:
    """Resolve one route without Starlette's host thread portal."""
    for route in application.routes:
        if getattr(route, "path", None) == path:
            return cast(Callable[..., Any], route.endpoint)
    raise LookupError(path)


def _scope(
    *,
    path: str = "/protected",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Scope:
    """Build the minimal HTTP ASGI scope required by pure middleware tests."""
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers or [],
        "client": ("192.0.2.20", 1234),
        "server": ("camera.example", 443),
    }


def test_field_mode_enforces_scopes_and_hides_diagnostics_and_docs() -> None:
    """Minimal health stays public while protected surfaces require scope."""
    application = create_app(
        MockCamera(),  # type: ignore[arg-type]
        manage_camera=False,
        security_config=_security_config(),
    )

    paths = {getattr(route, "path", None) for route in application.routes}
    assert _endpoint(application, "/healthz")() == {"status": "ok"}
    assert "/debug/health" in paths
    assert "/docs" not in paths
    assert "/redoc" not in paths
    assert "/openapi.json" not in paths

    middleware = {item.cls for item in application.user_middleware}
    assert TrustedHostMiddleware in middleware
    assert HTTPSRedirectMiddleware in middleware

    authorize = build_authorizer(_security_config())
    with pytest.raises(HTTPException) as missing:
        asyncio.run(authorize(SecurityScopes(["admin"]), None))
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as insufficient:
        asyncio.run(authorize(SecurityScopes(["admin"]), "stream-token"))
    assert insufficient.value.status_code == 403

    asyncio.run(authorize(SecurityScopes(["stream:read"]), "stream-token"))
    asyncio.run(authorize(SecurityScopes(["camera:control"]), "admin-token"))


def test_field_mode_limits_request_bodies_and_sets_security_headers() -> None:
    """Pure ASGI middleware must bound payloads without buffering streams."""
    app_called = False

    async def inner(_: Scope, __: Receive, send: Send) -> None:
        nonlocal app_called
        app_called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    messages: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    limited = RequestSizeLimitMiddleware(inner, max_bytes=16)
    scope = _scope(headers=[(b"content-length", b"17")])
    asyncio.run(limited(scope, receive, send))
    assert app_called is False
    assert messages[0]["status"] == 413

    messages.clear()
    secured = SecurityHeadersMiddleware(inner, hsts=True)
    asyncio.run(secured(_scope(), receive, send))
    response_headers = dict(messages[0]["headers"])
    assert response_headers[b"x-content-type-options"] == b"nosniff"
    assert b"strict-transport-security" in response_headers


def test_field_mode_applies_per_identity_rate_limits() -> None:
    """A token and source address cannot exhaust the API without throttling."""
    async def inner(_: Scope, __: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    limiter = RateLimitMiddleware(inner, rate=0.01, burst=2)
    scope = _scope(headers=[(b"authorization", b"Bearer admin-token")])

    async def request() -> list[Message]:
        messages: list[Message] = []

        async def send(message: Message) -> None:
            messages.append(message)

        await limiter(scope, receive, send)
        return messages

    assert asyncio.run(request())[0]["status"] == 200
    assert asyncio.run(request())[0]["status"] == 200
    limited = asyncio.run(request())
    assert limited[0]["status"] == 429
    assert dict(limited[0]["headers"])[b"retry-after"] == b"1"


def test_token_file_requires_hashed_tokens_and_restrictive_permissions(
    tmp_path: Path,
) -> None:
    """Field secrets must never be loaded from broadly readable files."""
    token_file = tmp_path / "tokens.json"
    token_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tokens": [
                    {
                        "sha256": token_sha256("device-token"),
                        "scopes": ["stream:read"],
                    }
                ],
            }
        ),
        "utf-8",
    )

    token_file.chmod(0o644)
    with pytest.raises(PermissionError, match="0600 or 0640"):
        SecurityConfig.from_token_file(token_file)

    token_file.chmod(0o600)
    config = SecurityConfig.from_token_file(token_file)
    assert config.authentication_required is True
    assert "device-token" not in token_file.read_text("utf-8")


def test_field_mode_rejects_missing_token_grants() -> None:
    """A production typo must stop startup instead of disabling auth."""
    with pytest.raises(ValueError, match="requires at least one bearer token"):
        SecurityConfig(field_mode=True)


def test_field_mode_rejects_missing_api_configuration(tmp_path: Path) -> None:
    """An explicit production config path must never fall back silently."""
    with pytest.raises(FileNotFoundError):
        create_app(
            MockCamera(),  # type: ignore[arg-type]
            manage_camera=False,
            config_path=tmp_path / "missing.yml",
            security_config=_security_config(),
        )
