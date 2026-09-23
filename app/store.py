"""Kho key-value dùng chung cho quota, phiên hội thoại và cache câu trả lời.

`RedisStore` cho production. `MemoryStore` cho dev và test: dữ liệu mất khi restart và không
chia sẻ giữa các worker.
"""

import time
from typing import Protocol


class KVStore(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def incr(self, key: str, ttl: int | None = None) -> int: ...
    async def decr(self, key: str) -> int: ...
    async def ping(self) -> bool: ...
    async def aclose(self) -> None: ...


class MemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float | None]] = {}

    def _alive(self, key: str) -> tuple[str, float | None] | None:
        item = self._data.get(key)
        if item and item[1] is not None and item[1] <= time.monotonic():
            del self._data[key]
            return None
        return item

    async def get(self, key: str) -> str | None:
        item = self._alive(key)
        return item[0] if item else None

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        self._data[key] = (value, time.monotonic() + ttl if ttl else None)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def incr(self, key: str, ttl: int | None = None) -> int:
        item = self._alive(key)
        if item is None:
            self._data[key] = ("1", time.monotonic() + ttl if ttl else None)
            return 1
        value = int(item[0]) + 1
        self._data[key] = (str(value), item[1])
        return value

    async def decr(self, key: str) -> int:
        item = self._alive(key)
        value = int(item[0]) - 1 if item else -1
        self._data[key] = (str(value), item[1] if item else None)
        return value

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class RedisStore:
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis

        self._r = redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> str | None:
        return await self._r.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        await self._r.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        await self._r.delete(key)

    async def incr(self, key: str, ttl: int | None = None) -> int:
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            if ttl:
                # NX: chỉ đặt TTL ở lần tạo key, không gia hạn ở mỗi lần tăng.
                pipe.expire(key, ttl, nx=True)
            value, *_ = await pipe.execute()
        return int(value)

    async def decr(self, key: str) -> int:
        return int(await self._r.decr(key))

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._r.aclose()


def make_store(redis_url: str) -> KVStore:
    return RedisStore(redis_url) if redis_url else MemoryStore()
