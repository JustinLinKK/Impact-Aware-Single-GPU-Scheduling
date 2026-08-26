"""Run/RAM/Redis CUDA-doc caches with TTL, stale reads, and single-flight."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable
import json
import random
import secrets
import threading
import time

from ..redis_cache import RedisLRUCache
from .models import CUDA_DOCS_CACHE_PREFIX, CudaDocsContext, CudaDocsSettings


@dataclass(frozen=True, slots=True)
class CacheLookup:
    hit: bool
    tier: str = "none"
    freshness: str = "unavailable"
    context: CudaDocsContext | None = None
    negative_reason: str | None = None


@dataclass(slots=True)
class _Entry:
    context: CudaDocsContext | None
    fresh_until: float
    stale_until: float
    negative_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict() if self.context else None,
            "fresh_until": self.fresh_until,
            "stale_until": self.stale_until,
            "negative_reason": self.negative_reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "_Entry":
        context = value.get("context")
        return cls(
            context=(
                CudaDocsContext.from_dict(context)
                if isinstance(context, dict)
                else None
            ),
            fresh_until=float(value.get("fresh_until") or 0.0),
            stale_until=float(value.get("stale_until") or 0.0),
            negative_reason=(
                str(value.get("negative_reason"))
                if value.get("negative_reason")
                else None
            ),
        )


class RunMemo:
    def __init__(self):
        self._values: dict[str, CudaDocsContext] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> CudaDocsContext | None:
        with self._lock:
            return self._values.get(key)

    def set(self, key: str, context: CudaDocsContext) -> None:
        if not context.evidence_chunks:
            return
        with self._lock:
            self._values[key] = context

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


class TTLRUCache:
    def __init__(self, max_entries: int, *, clock: Callable[[], float] = time.time):
        self.max_entries = max(0, int(max_entries))
        self.clock = clock
        self._values: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[_Entry | None, str]:
        now = self.clock()
        with self._lock:
            entry = self._values.get(key)
            if entry is None:
                return None, "unavailable"
            if now > entry.stale_until:
                self._values.pop(key, None)
                return None, "unavailable"
            self._values.move_to_end(key)
            return entry, "fresh" if now <= entry.fresh_until else "stale"

    def set(self, key: str, entry: _Entry) -> None:
        if self.max_entries <= 0:
            return
        with self._lock:
            self._values[key] = entry
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


class CudaDocsCache:
    def __init__(
        self,
        settings: CudaDocsSettings | Any,
        *,
        redis_cache: RedisLRUCache | None = None,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
    ):
        self.settings = CudaDocsSettings.from_any(settings)
        self.clock = clock
        self.rng = rng or random.Random()
        self.memo = RunMemo()
        self.ram = TTLRUCache(self.settings.ram_cache_max_entries, clock=clock)
        self.redis_cache = redis_cache
        self._flights: dict[str, threading.Event] = {}
        self._flight_lock = threading.Lock()

    def get(self, key: str) -> CacheLookup:
        memo = self.memo.get(key)
        if memo is not None:
            return CacheLookup(True, "run", memo.freshness, memo)
        entry, freshness = self.ram.get(key)
        if entry is not None:
            return self._lookup_from_entry(entry, tier="ram", freshness=freshness)
        entry = self._redis_get(key)
        if entry is None:
            return CacheLookup(False)
        freshness = "fresh" if self.clock() <= entry.fresh_until else "stale"
        if self.clock() > entry.stale_until:
            return CacheLookup(False)
        self.ram.set(key, self._ram_entry(entry))
        return self._lookup_from_entry(entry, tier="redis", freshness=freshness)

    def set_context(self, key: str, context: CudaDocsContext) -> None:
        now = self.clock()
        fresh = now + self._jitter(self.settings.positive_ttl_seconds)
        stale = now + self._jitter(self.settings.stale_ttl_seconds)
        stale = max(fresh, stale)
        entry = _Entry(context=context, fresh_until=fresh, stale_until=stale)
        self.ram.set(key, self._ram_entry(entry))
        self.memo.set(key, context)
        self._redis_set(key, entry)

    def set_negative(
        self, key: str, reason: str, *, ttl_seconds: int | None = None
    ) -> None:
        now = self.clock()
        ttl = self._jitter(ttl_seconds or self.settings.negative_ttl_seconds)
        entry = _Entry(
            context=None,
            fresh_until=now + ttl,
            stale_until=now + ttl,
            negative_reason=str(reason),
        )
        self.ram.set(key, self._ram_entry(entry))
        self._redis_set(key, entry)

    def begin_singleflight(self, key: str) -> tuple[bool, threading.Event]:
        with self._flight_lock:
            event = self._flights.get(key)
            if event is not None:
                return False, event
            event = threading.Event()
            self._flights[key] = event
            return True, event

    def finish_singleflight(self, key: str) -> None:
        with self._flight_lock:
            event = self._flights.pop(key, None)
        if event is not None:
            event.set()

    def wait_singleflight(self, event: threading.Event) -> bool:
        return event.wait(timeout=self.settings.singleflight_wait_seconds)

    def acquire_distributed_lock(self, key: str) -> str | None:
        if self.redis_cache is None or self.redis_cache.client is None:
            return "local-only"
        token = secrets.token_hex(16)
        acquired = self.redis_cache.acquire_lock(
            key + ":lock",
            token,
            deadline_ms=int(self.settings.total_enrichment_deadline_seconds * 1000),
        )
        if acquired is None:
            # Redis is an optimization. A connection error must not suppress a
            # policy-eligible hosted lookup.
            return "local-only"
        return token if acquired else None

    def release_distributed_lock(self, key: str, token: str | None) -> None:
        if self.redis_cache is None or token in {None, "local-only"}:
            return
        self.redis_cache.release_lock(key + ":lock", token)

    def _lookup_from_entry(
        self, entry: _Entry, *, tier: str, freshness: str
    ) -> CacheLookup:
        if entry.negative_reason:
            return CacheLookup(
                True,
                tier,
                freshness,
                negative_reason=entry.negative_reason,
            )
        return CacheLookup(True, tier, freshness, entry.context)

    def _jitter(self, value: int | float) -> float:
        fraction = self.settings.ttl_jitter_fraction
        return max(1.0, float(value) * self.rng.uniform(1.0 - fraction, 1.0 + fraction))

    def _ram_entry(self, entry: _Entry) -> _Entry:
        """Bound L1 residence separately from the shared positive TTL."""

        expires = self.clock() + self.settings.ram_cache_ttl_seconds
        return _Entry(
            context=entry.context,
            fresh_until=min(entry.fresh_until, expires),
            stale_until=min(entry.stale_until, expires),
            negative_reason=entry.negative_reason,
        )

    def _redis_get(self, key: str) -> _Entry | None:
        client = self.redis_cache.client if self.redis_cache is not None else None
        if client is None:
            return None
        try:
            raw = client.get(key)
            if raw is None:
                client.zrem(CUDA_DOCS_CACHE_PREFIX + ":__index__", key)
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            client.zadd(CUDA_DOCS_CACHE_PREFIX + ":__index__", {key: self.clock()})
            return _Entry.from_dict(json.loads(str(raw)))
        except Exception:
            return None

    def _redis_set(self, key: str, entry: _Entry) -> None:
        client = self.redis_cache.client if self.redis_cache is not None else None
        if client is None or self.settings.redis_namespace_capacity <= 0:
            return
        ttl = max(1, int(entry.stale_until - self.clock()))
        index_key = CUDA_DOCS_CACHE_PREFIX + ":__index__"
        try:
            client.set(
                key,
                json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":")),
                ex=ttl,
            )
            client.zadd(index_key, {key: self.clock()})
            count = int(client.zcard(index_key) or 0)
            overflow = count - self.settings.redis_namespace_capacity
            if overflow > 0:
                victims = list(client.zrange(index_key, 0, overflow - 1) or [])
                if victims:
                    client.delete(*victims)
                    client.zrem(index_key, *victims)
        except Exception:
            return
