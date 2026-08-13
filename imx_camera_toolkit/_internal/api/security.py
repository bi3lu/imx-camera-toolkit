"""Shared deployment security for camera HTTP applications."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
import threading
import time
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

SCOPE_DESCRIPTIONS: dict[str, str] = {
    "stream:read": "Read camera preview streams and signaling data.",
    "camera:read": "Read camera controls and processing configuration.",
    "camera:control": "Change camera controls and processing configuration.",
    "profiles:write": "Create, apply, and delete camera control profiles.",
    "admin": "Read detailed device and client diagnostics.",
}
ALL_SCOPES = frozenset(SCOPE_DESCRIPTIONS)
BROWSER_SESSION_COOKIE = "imx_camera_session"


class BrowserSessionOAuth2PasswordBearer(OAuth2PasswordBearer):
    """Read a Bearer header first and a same-site browser session second."""

    async def __call__(self, request: Request) -> str | None:
        """Return the regular OAuth2 token or the HttpOnly session cookie."""
        token = await super().__call__(request)
        return token or request.cookies.get(BROWSER_SESSION_COOKIE)


def token_sha256(token: str) -> str:
    """Return the lowercase digest stored in a device token file."""
    if not isinstance(token, str) or not token:
        raise ValueError("token must be a non-empty string")

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _validate_digest(value: object) -> str:
    """Validate one persisted bearer-token digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("token sha256 must be a lowercase SHA-256 digest")

    return value


def _validate_secret_file(path: Path) -> None:
    """Require a regular, tightly permissioned file owned by root or this user."""
    if path.is_symlink():
        raise PermissionError(f"security file must not be a symlink: {path}")

    details = path.stat()

    if not stat.S_ISREG(details.st_mode):
        raise PermissionError(f"security file must be regular: {path}")

    if details.st_uid not in {0, os.geteuid()}:
        raise PermissionError(f"security file has an untrusted owner: {path}")

    permissions = stat.S_IMODE(details.st_mode)

    if permissions & 0o137:
        raise PermissionError(f"security file permissions must be 0600 or 0640: {path}")


@dataclass(frozen=True, slots=True)
class SecurityConfig:
    """HTTP authentication, transport, and abuse-prevention policy."""

    field_mode: bool = False
    token_grants: tuple[tuple[str, frozenset[str]], ...] = ()
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "[::1]")
    require_https: bool = False
    rate_limit_per_second: float = 20.0
    rate_limit_burst: int = 40
    max_request_body_bytes: int = 128 * 1024

    def __post_init__(self) -> None:
        """Reject incomplete field deployments and invalid limits."""
        for name in ("field_mode", "require_https"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

        if self.field_mode and not self.token_grants:
            raise ValueError("field mode requires at least one bearer token")

        if not self.allowed_hosts or any(
            not isinstance(host, str) or not host.strip() for host in self.allowed_hosts
        ):
            raise ValueError("allowed_hosts must contain non-empty host names")

        if (
            isinstance(self.rate_limit_per_second, bool)
            or not isinstance(self.rate_limit_per_second, (int, float))
            or not math.isfinite(self.rate_limit_per_second)
            or not 0 < self.rate_limit_per_second <= 100_000
        ):
            raise ValueError(
                "rate_limit_per_second must be finite and between 0 and 100000"
            )

        for name in ("rate_limit_burst", "max_request_body_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        seen: set[str] = set()

        for digest, scopes in self.token_grants:
            _validate_digest(digest)

            if digest in seen:
                raise ValueError("token file contains duplicate SHA-256 digests")

            seen.add(digest)
            unknown = scopes - ALL_SCOPES

            if unknown:
                formatted = ", ".join(sorted(unknown))
                raise ValueError(f"unknown token scope(s): {formatted}")

            if not scopes:
                raise ValueError("every token must grant at least one scope")

    @property
    def authentication_required(self) -> bool:
        """Whether protected routes must receive a bearer token."""
        return self.field_mode or bool(self.token_grants)

    @property
    def docs_enabled(self) -> bool:
        """Whether interactive API schemas may be published."""
        return not self.field_mode

    @classmethod
    def from_token_file(
        cls,
        token_file: str | Path,
        *,
        field_mode: bool = True,
        allowed_hosts: Sequence[str] = ("localhost", "127.0.0.1", "[::1]"),
        require_https: bool = False,
        rate_limit_per_second: float = 20.0,
        rate_limit_burst: int = 40,
        max_request_body_bytes: int = 128 * 1024,
    ) -> SecurityConfig:
        """Load hashed per-device tokens from a protected JSON file."""
        path = Path(token_file)
        _validate_secret_file(path)

        try:
            document = json.loads(path.read_text("utf-8"))

        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not load token file {path}: {error}") from error

        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise ValueError("token file must use schema_version 1")

        raw_tokens = document.get("tokens")

        if not isinstance(raw_tokens, list):
            raise ValueError("token file tokens must be a list")

        grants: list[tuple[str, frozenset[str]]] = []

        for raw_token in raw_tokens:
            if not isinstance(raw_token, dict):
                raise ValueError("each token grant must be a mapping")

            if set(raw_token) != {"sha256", "scopes"}:
                raise ValueError("token grants accept only sha256 and scopes")

            raw_scopes = raw_token["scopes"]

            if not isinstance(raw_scopes, list) or not all(
                isinstance(scope, str) for scope in raw_scopes
            ):
                raise ValueError("token scopes must be a list of strings")

            grants.append(
                (_validate_digest(raw_token["sha256"]), frozenset(raw_scopes))
            )

        return cls(
            field_mode=field_mode,
            token_grants=tuple(grants),
            allowed_hosts=tuple(allowed_hosts),
            require_https=require_https,
            rate_limit_per_second=rate_limit_per_second,
            rate_limit_burst=rate_limit_burst,
            max_request_body_bytes=max_request_body_bytes,
        )


class _TokenBucket:
    """Small thread-safe token bucket keyed by request identity."""

    def __init__(self, rate: float, burst: int) -> None:
        """Initialize a limiter with a refill rate and maximum burst."""
        self._rate = float(rate)
        self._burst = float(burst)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Consume one token if the named bucket is not empty."""
        now = time.monotonic()

        with self._lock:
            tokens, updated = self._buckets.get(key, (self._burst, now))
            tokens = min(self._burst, tokens + (now - updated) * self._rate)
            allowed = tokens >= 1.0
            self._buckets[key] = (tokens - 1.0 if allowed else tokens, now)

            if len(self._buckets) > 4096:
                cutoff = now - max(self._burst / self._rate * 2.0, 60.0)
                self._buckets = {
                    name: value
                    for name, value in self._buckets.items()
                    if value[1] >= cutoff
                }

            return allowed


async def _plain_response(
    send: Send,
    status: int,
    body: bytes,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    """Emit one compact ASGI response from a middleware."""
    resolved_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    if headers:
        resolved_headers.extend(headers)

    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": resolved_headers,
        }
    )
    await send({"type": "http.response.body", "body": body})


class RequestSizeLimitMiddleware:
    """Reject request bodies that exceed the configured byte ceiling."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        """Initialize the middleware with an ASGI app and byte limit."""
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject oversized HTTP requests before forwarding them."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", ()):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await _plain_response(
                            send,
                            413,
                            b'{"detail":"request body too large"}',
                        )
                        return

                except ValueError:
                    await _plain_response(
                        send,
                        400,
                        b'{"detail":"invalid content length"}',
                    )
                    return

        consumed = 0
        exceeded = False

        async def limited_receive() -> Message:
            """Count streamed request bytes before forwarding a message."""
            nonlocal consumed, exceeded
            message = await receive()

            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))

                if consumed > self.max_bytes:
                    exceeded = True
                    return {"type": "http.disconnect"}

            return message

        try:
            await self.app(scope, limited_receive, send)

        except Exception:
            if exceeded:
                await _plain_response(send, 413, b'{"detail":"request body too large"}')
                return

            raise


class RateLimitMiddleware:
    """Apply independent per-IP and per-token request buckets."""

    def __init__(self, app: ASGIApp, rate: float, burst: int) -> None:
        """Initialize independent request buckets for the ASGI app."""
        self.app = app
        self._limiter = _TokenBucket(rate, burst)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Rate-limit HTTP requests by client address and credential."""
        if scope["type"] != "http" or scope.get("path") == "/healthz":
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        address = str(client[0]) if client else "unknown"
        headers = {name.lower(): value for name, value in scope.get("headers", ())}
        authorization = headers.get(b"authorization", b"")
        token_identity = hashlib.sha256(authorization).hexdigest()

        if not self._limiter.allow(f"ip:{address}") or not self._limiter.allow(
            f"token:{token_identity}"
        ):
            await _plain_response(
                send,
                429,
                b'{"detail":"rate limit exceeded"}',
                [(b"retry-after", b"1")],
            )
            return

        await self.app(scope, receive, send)


class SecurityHeadersMiddleware:
    """Attach browser hardening headers without buffering streaming bodies."""

    def __init__(self, app: ASGIApp, hsts: bool = False) -> None:
        """Initialize response hardening and optional HSTS."""
        self.app = app
        self._hsts = hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Attach security headers to each HTTP response."""

        async def send_with_headers(message: Message) -> None:
            """Add hardening headers before forwarding a response message."""
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", ()))
                headers.extend(
                    (
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (
                            b"permissions-policy",
                            b"camera=(), microphone=(), geolocation=()",
                        ),
                    )
                )

                if self._hsts:
                    headers.append((b"strict-transport-security", b"max-age=31536000"))
                message["headers"] = headers

            await send(message)

        await self.app(scope, receive, send_with_headers)


Authorizer = Callable[..., Coroutine[Any, Any, None]]


def build_authorizer(config: SecurityConfig) -> Authorizer:
    """Build a FastAPI OAuth2-scope dependency for one application."""
    oauth2 = BrowserSessionOAuth2PasswordBearer(
        tokenUrl="/auth/token",
        scopes=SCOPE_DESCRIPTIONS,
        scheme_name="OAuth2PasswordBearer",
        auto_error=False,
    )

    async def authorize(
        security_scopes: SecurityScopes,
        token: str | None = Depends(oauth2),
    ) -> None:
        """Authorize a bearer token against the requested OAuth2 scopes."""
        if not config.authentication_required:
            return

        authenticate = f'Bearer scope="{security_scopes.scope_str}"'

        if token is None:
            raise HTTPException(
                status_code=401,
                detail="bearer token required",
                headers={"WWW-Authenticate": authenticate},
            )

        presented = token_sha256(token)
        granted: frozenset[str] | None = None

        for digest, scopes in config.token_grants:
            if secrets.compare_digest(presented, digest):
                granted = scopes

        if granted is None:
            raise HTTPException(
                status_code=401,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": authenticate},
            )

        required = set(security_scopes.scopes)
        if "admin" not in granted and not required.issubset(granted):
            raise HTTPException(
                status_code=403,
                detail="token has insufficient scope",
                headers={"WWW-Authenticate": authenticate},
            )

    return authorize


def apply_security_middleware(application: FastAPI, config: SecurityConfig) -> None:
    """Apply field-mode middleware in a streaming-safe order."""
    application.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=config.max_request_body_bytes,
    )
    if not config.field_mode:
        return

    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(config.allowed_hosts),
    )
    if config.require_https:
        application.add_middleware(HTTPSRedirectMiddleware)
    application.add_middleware(
        RateLimitMiddleware,
        rate=config.rate_limit_per_second,
        burst=config.rate_limit_burst,
    )
    application.add_middleware(
        SecurityHeadersMiddleware,
        hsts=config.require_https,
    )


__all__ = [
    "ALL_SCOPES",
    "SCOPE_DESCRIPTIONS",
    "SecurityConfig",
    "apply_security_middleware",
    "build_authorizer",
    "token_sha256",
]
