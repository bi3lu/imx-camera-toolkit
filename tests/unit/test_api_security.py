"""Unit tests for fail-closed HTTP deployment security."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import SecurityScopes
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send

from imx_camera_toolkit._internal.api.api import create_app
from imx_camera_toolkit._internal.api.security import (
    BROWSER_SESSION_COOKIE,
    BrowserSessionOAuth2PasswordBearer,
    BrowserSessionStore,
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    SecurityConfig,
    SecurityHeadersMiddleware,
    build_authorizer,
    token_sha256,
)
from imx_camera_toolkit._internal.testing.mock_camera import MockCamera


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
    client: tuple[str, int] = ("192.0.2.20", 1234),
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
        "client": client,
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


def test_authorizer_registers_bearer_dependency_in_fastapi_openapi() -> None:
    """Postponed annotations must not turn the bearer token into a query input."""
    application = FastAPI()
    authorize = build_authorizer(_security_config())

    @application.get(
        "/protected",
        dependencies=[Security(authorize, scopes=["stream:read"])],
    )
    def protected() -> dict[str, bool]:
        return {"ok": True}

    operation = application.openapi()["paths"]["/protected"]["get"]

    assert operation.get("parameters", []) == []
    assert operation["security"] == [{"OAuth2PasswordBearer": ["stream:read"]}]


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


def test_request_size_limit_rejects_invalid_and_streamed_lengths() -> None:
    """Body limits must handle malformed headers and chunked overflow."""

    async def rejecting_inner(_: Scope, receive: Receive, __: Send) -> None:
        assert (await receive())["type"] == "http.request"
        assert (await receive())["type"] == "http.disconnect"
        raise RuntimeError("request disconnected")

    async def empty_receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def run_invalid_length() -> list[Message]:
        messages: list[Message] = []

        async def send(message: Message) -> None:
            messages.append(message)

        middleware = RequestSizeLimitMiddleware(rejecting_inner, max_bytes=16)
        await middleware(
            _scope(headers=[(b"content-length", b"invalid")]),
            empty_receive,
            send,
        )
        return messages

    chunks = iter(
        (
            {"type": "http.request", "body": b"1234567890", "more_body": True},
            {"type": "http.request", "body": b"abcdefghij", "more_body": False},
        )
    )

    async def chunked_receive() -> Message:
        return cast(Message, next(chunks))

    async def run_streamed_body() -> list[Message]:
        messages: list[Message] = []

        async def send(message: Message) -> None:
            messages.append(message)

        middleware = RequestSizeLimitMiddleware(rejecting_inner, max_bytes=16)
        await middleware(_scope(), chunked_receive, send)
        return messages

    invalid = asyncio.run(run_invalid_length())
    streamed = asyncio.run(run_streamed_body())

    assert invalid[0]["status"] == 400
    assert streamed[0]["status"] == 413


def test_security_helpers_cover_missing_credentials_and_session_capacity() -> None:
    """Missing credentials fail closed and bounded stores evict old sessions."""
    extractor = BrowserSessionOAuth2PasswordBearer(
        tokenUrl="/auth/session",
        auto_error=True,
    )
    request = Request({"type": "http", "headers": []})

    with pytest.raises(HTTPException) as missing:
        asyncio.run(extractor(request))

    assert missing.value.status_code == 401

    store = BrowserSessionStore(ttl_seconds=10, max_entries=1)
    first_id = store.create(frozenset({"stream:read"}))
    second_id = store.create(frozenset({"stream:read"}))

    assert store.resolve(first_id) is None
    assert store.resolve(second_id) is not None
    assert store.resolve("") is None
    store.revoke("")
    store.clear()
    assert store.resolve(second_id) is None

    optional_extractor = BrowserSessionOAuth2PasswordBearer(
        tokenUrl="/auth/session",
        auto_error=False,
    )
    bearer_request = Request(
        {
            "type": "http",
            "headers": [(b"authorization", b"Bearer stream-token")],
        }
    )
    empty_request = Request({"type": "http", "headers": []})
    assert asyncio.run(optional_extractor(bearer_request)) == "stream-token"
    assert asyncio.run(optional_extractor(empty_request)) is None

    expiring_now = [0.0]
    expiring_store = BrowserSessionStore(
        ttl_seconds=1,
        max_entries=2,
        clock=lambda: expiring_now[0],
    )
    stale_id = expiring_store.create(frozenset({"stream:read"}))
    expiring_now[0] = 1.0
    expiring_store.create(frozenset({"stream:read"}))
    assert expiring_store.resolve(stale_id) is None


def test_security_helpers_reject_invalid_inputs_and_use_anonymous_buckets(
    tmp_path: Path,
) -> None:
    """Invalid security inputs fail closed without manufacturing identities."""
    with pytest.raises(ValueError, match="non-empty"):
        token_sha256("")

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        SecurityConfig(
            token_grants=(("invalid", frozenset({"stream:read"})),),
        )

    digest = token_sha256("duplicate")

    with pytest.raises(ValueError, match="duplicate"):
        SecurityConfig(
            token_grants=(
                (digest, frozenset({"stream:read"})),
                (digest, frozenset({"admin"})),
            ),
        )

    with pytest.raises(ValueError, match="positive"):
        BrowserSessionStore(ttl_seconds=0)

    with pytest.raises(ValueError, match="must be a boolean"):
        SecurityConfig(field_mode="yes")  # type: ignore[arg-type]

    authorize = build_authorizer(SecurityConfig())
    asyncio.run(authorize(SecurityScopes(["admin"]), None))

    async def inner(_: Scope, __: Receive, ___: Send) -> None:
        return

    anonymous = RateLimitMiddleware(inner, rate=1.0, burst=1)
    configured = RateLimitMiddleware(
        inner,
        rate=1.0,
        burst=1,
        security_config=_security_config(),
    )
    invalid_token_scope = _scope(headers=[(b"authorization", b"Bearer invalid-token")])

    assert anonymous._credential_identity(_scope(), "192.0.2.20") == (
        "anonymous:192.0.2.20"
    )

    assert configured._credential_identity(invalid_token_scope, "192.0.2.20") == (
        "anonymous:192.0.2.20"
    )

    token_file = tmp_path / "invalid-tokens.json"
    token_file.write_text("not-json", "utf-8")
    token_file.chmod(0o600)

    with pytest.raises(ValueError, match="could not load"):
        SecurityConfig.from_token_file(token_file)

    token_file.write_text("{}", "utf-8")
    with pytest.raises(ValueError, match="schema_version 1"):
        SecurityConfig.from_token_file(token_file)

    async def failing_inner(_: Scope, __: Receive, ___: Send) -> None:
        raise RuntimeError("application failure")

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_: Message) -> None:
        return

    middleware = RequestSizeLimitMiddleware(failing_inner, max_bytes=16)

    with pytest.raises(RuntimeError, match="application failure"):
        asyncio.run(middleware(_scope(), receive, send))


def test_token_file_schema_and_middleware_bypass_paths(tmp_path: Path) -> None:
    """Malformed grants fail closed while non-request traffic bypasses limits."""
    token_file = tmp_path / "tokens.json"
    invalid_documents = (
        ({"schema_version": 1, "tokens": {}}, "tokens must be a list"),
        ({"schema_version": 1, "tokens": ["token"]}, "must be a mapping"),
        (
            {"schema_version": 1, "tokens": [{"sha256": "value"}]},
            "accept only sha256 and scopes",
        ),
        (
            {
                "schema_version": 1,
                "tokens": [{"sha256": token_sha256("token"), "scopes": "admin"}],
            },
            "list of strings",
        ),
    )

    for document, message in invalid_documents:
        token_file.write_text(json.dumps(document), "utf-8")
        token_file.chmod(0o600)

        with pytest.raises(ValueError, match=message):
            SecurityConfig.from_token_file(token_file)

    called = 0

    async def inner(_: Scope, __: Receive, send: Send) -> None:
        nonlocal called
        called += 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_: Message) -> None:
        return

    limiter = RateLimitMiddleware(inner, rate=1.0, burst=1)
    asyncio.run(limiter(_scope(path="/healthz"), receive, send))

    non_http = _scope()
    non_http["type"] = "websocket"
    size_limiter = RequestSizeLimitMiddleware(inner, max_bytes=16)
    asyncio.run(size_limiter(non_http, receive, send))

    assert called == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"allowed_hosts": ()}, "allowed_hosts"),
        ({"rate_limit_per_second": float("inf")}, "rate_limit_per_second"),
        ({"browser_session_ttl_seconds": 0}, "positive integer"),
        (
            {"token_grants": ((token_sha256("token"), frozenset({"unknown:scope"})),)},
            "unknown token scope",
        ),
        (
            {"token_grants": ((token_sha256("token"), frozenset()),)},
            "at least one scope",
        ),
    ],
)
def test_security_config_rejects_invalid_policy_values(
    kwargs: dict[str, object], message: str
) -> None:
    """Security policy validation must fail closed for malformed limits and grants."""
    with pytest.raises(ValueError, match=message):
        SecurityConfig(**kwargs)  # type: ignore[arg-type]


def test_field_mode_applies_per_identity_rate_limits() -> None:
    """A token and source address cannot exhaust the API without throttling."""

    async def inner(_: Scope, __: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    limiter = RateLimitMiddleware(
        inner,
        rate=0.01,
        burst=2,
        security_config=_security_config(),
    )
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


def test_browser_sessions_expire_and_can_be_revoked() -> None:
    """Only digests remain server-side and TTL/revocation invalidate sessions."""
    now = [100.0]
    store = BrowserSessionStore(
        ttl_seconds=10,
        max_entries=4,
        clock=lambda: now[0],
    )
    session_id = store.create(frozenset({"stream:read"}))

    assert session_id not in repr(store._sessions)
    assert store.resolve(session_id) is not None

    store.revoke(session_id)
    assert store.resolve(session_id) is None

    expiring_id = store.create(frozenset({"stream:read"}))
    now[0] += 10
    assert store.resolve(expiring_id) is None


def test_authorizer_accepts_issued_session_but_not_bearer_value_in_cookie() -> None:
    """Cookie credentials must resolve through server-side session state only."""
    security = _security_config()
    sessions = BrowserSessionStore()
    authorize = build_authorizer(security, sessions)
    extractor = BrowserSessionOAuth2PasswordBearer(
        tokenUrl="/auth/session",
        auto_error=False,
    )
    session_id = sessions.create(frozenset({"stream:read"}))

    async def extract(cookie_value: str) -> str | None:
        request = Request(
            {
                "type": "http",
                "headers": [
                    (
                        b"cookie",
                        f"{BROWSER_SESSION_COOKIE}={cookie_value}".encode("ascii"),
                    )
                ],
            }
        )
        return await extractor(request)

    issued = asyncio.run(extract(session_id))
    assert issued is not None
    asyncio.run(authorize(SecurityScopes(["stream:read"]), issued))

    raw_bearer_cookie = asyncio.run(extract("stream-token"))
    assert raw_bearer_cookie is not None

    with pytest.raises(HTTPException) as invalid:
        asyncio.run(authorize(SecurityScopes(["stream:read"]), raw_bearer_cookie))

    assert invalid.value.status_code == 401


def test_rate_limit_uses_independent_verified_browser_session_buckets() -> None:
    """One browser session must not consume another session's credential bucket."""

    async def inner(_: Scope, __: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    security = _security_config()
    sessions = BrowserSessionStore()
    first_id = sessions.create(frozenset({"stream:read"}))
    second_id = sessions.create(frozenset({"stream:read"}))
    limiter = RateLimitMiddleware(
        inner,
        rate=0.01,
        burst=2,
        security_config=security,
        browser_sessions=sessions,
    )

    async def request(scope: Scope) -> int:
        messages: list[Message] = []

        async def send(message: Message) -> None:
            messages.append(message)

        await limiter(scope, receive, send)
        return cast(int, messages[0]["status"])

    first_scope = _scope(
        headers=[(b"cookie", f"{BROWSER_SESSION_COOKIE}={first_id}".encode("ascii"))],
        client=("192.0.2.20", 1234),
    )
    second_scope = _scope(
        headers=[(b"cookie", f"{BROWSER_SESSION_COOKIE}={second_id}".encode("ascii"))],
        client=("192.0.2.21", 1234),
    )

    assert asyncio.run(request(first_scope)) == 200
    assert asyncio.run(request(first_scope)) == 200
    assert asyncio.run(request(first_scope)) == 429
    assert asyncio.run(request(second_scope)) == 200


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
