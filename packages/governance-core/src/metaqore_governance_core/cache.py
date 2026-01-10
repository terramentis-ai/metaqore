"""Cache abstraction for MetaQore services."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory cache only")


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all values from cache."""
        pass


class InMemoryCache(CacheBackend):
    """Simple in-memory cache backend."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache (TTL ignored for in-memory)."""
        self._cache[key] = value

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear all values from cache."""
        self._cache.clear()


class RedisCache(CacheBackend):
    """Redis cache backend."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available")
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value = await self.redis.get(key)
            if value is not None:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache with optional TTL."""
        try:
            json_value = json.dumps(value)
            if ttl:
                await self.redis.setex(key, ttl, json_value)
            else:
                await self.redis.set(key, json_value)
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    async def delete(self, key: str) -> None:
        """Delete value from Redis cache."""
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")

    async def clear(self) -> None:
        """Clear all values from Redis cache."""
        try:
            await self.redis.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")


class Cache:
    """Cache abstraction with backend selection."""

    def __init__(self, backend: CacheBackend):
        self.backend = backend

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return await self.backend.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        await self.backend.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        await self.backend.delete(key)

    async def clear(self) -> None:
        """Clear all values from cache."""
        await self.backend.clear()

    @classmethod
    def create_in_memory_cache(cls) -> "Cache":
        """Create cache with in-memory backend."""
        return cls(InMemoryCache())

    @classmethod
    def create_redis_cache(
        cls,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ) -> "Cache":
        """Create cache with Redis backend."""
        return cls(RedisCache(host, port, db, password))