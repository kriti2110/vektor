from __future__ import annotations

import hashlib
import json
from typing import Any


def cache_key(*parts: Any) -> str:
    """Stable cache key from arbitrary jsonable parts."""
    payload = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return "vektor:" + hashlib.sha256(payload).hexdigest()[:32]


class QueryCache:
    """Thin async Redis wrapper. Falls back to no-op if Redis is unavailable.

    The fallback is deliberate — local dev / CI without Redis still works,
    just without caching. Production wires this to a real Redis cluster.
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 3600) -> None:
        self.redis_url = redis_url
        self.ttl = ttl_seconds
        self._redis = None
        self._available = False

    async def connect(self) -> None:
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self._redis.ping()
            self._available = True
        except Exception:
            self._available = False
            self._redis = None

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
        self._redis = None
        self._available = False

    async def get(self, key: str) -> Any | None:
        if not self._available:
            return None
        try:
            raw = await self._redis.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if not self._available:
            return
        try:
            await self._redis.setex(key, ttl or self.ttl, json.dumps(value, default=str))
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available
