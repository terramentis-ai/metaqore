"""Custom FastAPI middleware for MetaQore."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from metaqore.config import MetaQoreConfig


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request-scoped identifiers and latency tracking."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request_id = str(uuid4())
        start = time.perf_counter()
        request.state.request_id = request_id
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000
        request.state.latency_ms = latency_ms
        response.headers.setdefault("X-Request-ID", request_id)
        response.headers.setdefault("X-Response-Time", f"{latency_ms:.2f}ms")
        return response


class GovernanceEnforcementMiddleware(BaseHTTPMiddleware):
    """Expose the active governance mode to downstream handlers."""

    def __init__(self, app: ASGIApp, config: MetaQoreConfig) -> None:
        super().__init__(app)
        self._config = config

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        request.state.governance_mode = self._config.governance_mode
        response = await call_next(request)
        response.headers.setdefault(
            "X-MetaQore-Governance-Mode", self._config.governance_mode.value
        )
        return response


class PrivilegedClientMiddleware(BaseHTTPMiddleware):
    """Flag requests coming from privileged clients (e.g., TerraQore Studio)."""

    HEADER_NAME = "X-MetaQore-Privileged"

    def __init__(self, app: ASGIApp, *, privileged_token: Optional[str] = None) -> None:
        super().__init__(app)
        self._privileged_token = privileged_token

    def _is_privileged(self, header_value: Optional[str]) -> bool:
        if not header_value:
            return False
        if not self._privileged_token:
            return header_value.lower() in {"1", "true", "yes", "terraqore"}
        return header_value == self._privileged_token

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        header_value = request.headers.get(self.HEADER_NAME)
        request.state.is_privileged = self._is_privileged(header_value)
        response = await call_next(request)
        if getattr(request.state, "is_privileged", False):
            response.headers.setdefault(self.HEADER_NAME, "true")
        return response


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Simple API-key or JWT guard for all API routes."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_key: Optional[str],
        jwt_secret_key: Optional[str],
        jwt_algorithm: str = "HS256",
        header_name: str = "Authorization",
    ) -> None:
        super().__init__(app)
        self._api_key = api_key
        self._jwt_secret_key = jwt_secret_key
        self._jwt_algorithm = jwt_algorithm
        self._header_name = header_name

    def _extract_token(self, request: Request) -> Optional[str]:
        header_value = request.headers.get(self._header_name)
        if header_value and header_value.lower().startswith("bearer "):
            return header_value.split(" ", 1)[1]
        return header_value or request.headers.get("X-API-Key")

    def _verify_jwt(self, token: str) -> bool:
        """Verify JWT token if JWT is enabled."""
        if not self._jwt_secret_key:
            return False
        try:
            from jose import jwt

            jwt.decode(token, self._jwt_secret_key, algorithms=[self._jwt_algorithm])
            return True
        except Exception:
            return False

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if not self._api_key and not self._jwt_secret_key:
            return await call_next(request)
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        token = self._extract_token(request)
        if not token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authorization required"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check API key first
        if self._api_key and token == self._api_key:
            return await call_next(request)

        # Check JWT
        if self._verify_jwt(token):
            return await call_next(request)

        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid token"},
            headers={"WWW-Authenticate": "Bearer"},
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-backed token bucket rate limiter for distributed rate limiting."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        enabled: bool,
        per_minute: int,
        burst: int,
        redis_url: str,
    ) -> None:
        super().__init__(app)
        self._enabled = enabled
        self._per_minute = per_minute
        self._burst = burst
        self._redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self._lock = asyncio.Lock()

    async def _get_redis(self) -> redis.Redis:
        """Lazy initialize Redis client."""
        if self._redis is None:
            import redis.asyncio as redis

            self._redis = redis.from_url(self._redis_url)
        return self._redis

    def _key(self, request: Request) -> str:
        auth = request.headers.get("Authorization") or request.headers.get("X-API-Key")
        return auth or (request.client.host if request.client else "anonymous")

    async def _consume_token(self, key: str) -> bool:
        """Consume a token from Redis-based bucket."""
        redis_client = await self._get_redis()
        now = time.time()
        window_start = now - 60.0  # 1 minute window

        # Use Redis sorted set to track timestamps
        bucket_key = f"ratelimit:{key}"

        # Remove old entries
        await redis_client.zremrangebyscore(bucket_key, 0, window_start)

        # Check current count
        count = await redis_client.zcard(bucket_key)
        if count >= self._burst:
            return False

        # Add new timestamp
        await redis_client.zadd(bucket_key, {str(now): now})

        # Set expiration to clean up
        await redis_client.expire(bucket_key, 60)

        return True

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if not self._enabled or not request.url.path.startswith("/api/"):
            return await call_next(request)

        key = self._key(request)

        async with self._lock:
            allowed = await self._consume_token(key)

        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
            )

        return await call_next(request)


def register_middlewares(
    app: FastAPI, config: MetaQoreConfig, *, privileged_token: Optional[str] = None
) -> None:
    """Register MetaQore middleware stack on the supplied FastAPI app."""

    app.add_middleware(GZipMiddleware, minimum_size=512)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        APIKeyAuthMiddleware,
        api_key=config.api_key,
        jwt_secret_key=config.jwt_secret_key,
        jwt_algorithm=config.jwt_algorithm,
        header_name=config.api_key_header,
    )
    app.add_middleware(
        RateLimitMiddleware,
        enabled=config.enable_rate_limit,
        per_minute=config.rate_limit_per_minute,
        burst=config.rate_limit_burst,
        redis_url=config.redis_url,
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(GovernanceEnforcementMiddleware, config=config)
    app.add_middleware(PrivilegedClientMiddleware, privileged_token=privileged_token)


__all__ = [
    "RequestContextMiddleware",
    "GovernanceEnforcementMiddleware",
    "PrivilegedClientMiddleware",
    "register_middlewares",
]
