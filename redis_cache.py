"""Optional Redis-backed LRU cache helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import hashlib
import json
import logging
import os
import time


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RedisCacheSettings:
    enabled: bool = False
    url: str = "redis://127.0.0.1:6379/0"
    url_env: str = "LOCALML_SCHEDULER_REDIS_URL"
    key_prefix: str = "localml_scheduler"
    ttl_seconds: int | None = 300
    max_entries: int | None = 4096
    socket_timeout_seconds: float = 0.2
    cache_graph_queries: bool = True
    cache_vector_queries: bool = True

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.url = str(self.url or "redis://127.0.0.1:6379/0").strip()
        self.url_env = str(self.url_env or "LOCALML_SCHEDULER_REDIS_URL").strip()
        self.key_prefix = str(self.key_prefix or "localml_scheduler").strip().strip(":") or "localml_scheduler"
        if self.ttl_seconds is not None:
            self.ttl_seconds = max(1, int(self.ttl_seconds))
        if self.max_entries is not None:
            self.max_entries = max(0, int(self.max_entries))
        self.socket_timeout_seconds = max(0.0, float(self.socket_timeout_seconds))
        self.cache_graph_queries = bool(self.cache_graph_queries)
        self.cache_vector_queries = bool(self.cache_vector_queries)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RedisCacheSettings":
        return cls(**(payload or {}))

    def resolved_url(self) -> str:
        if self.url_env:
            env_value = os.getenv(self.url_env)
            if env_value:
                return env_value
        return self.url

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "url": self.url,
            "url_env": self.url_env,
            "key_prefix": self.key_prefix,
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
            "socket_timeout_seconds": self.socket_timeout_seconds,
            "cache_graph_queries": self.cache_graph_queries,
            "cache_vector_queries": self.cache_vector_queries,
        }


class RedisLRUCache:
    """Tiny application-level LRU cache layered on Redis string keys.

    Redis itself can enforce maxmemory policies, but this wrapper keeps an
    explicit per-namespace recency index so vector and graph caches can be
    evicted independently in the same Redis database.
    """

    def __init__(self, settings: RedisCacheSettings, *, redis_client: Any | None = None):
        self.settings = settings
        self._client = redis_client
        self._connect_failed = False
        self.enabled = bool(settings.enabled)

    @classmethod
    def from_settings(cls, owner_settings: Any, *, redis_client: Any | None = None) -> "RedisLRUCache | None":
        settings = getattr(owner_settings, "redis_cache", None)
        if settings is None:
            return None
        if isinstance(settings, dict):
            settings = RedisCacheSettings.from_dict(settings)
        if not getattr(settings, "enabled", False):
            return None
        return cls(settings, redis_client=redis_client)

    @property
    def client(self) -> Any | None:
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client
        if self._connect_failed:
            return None
        try:
            import redis

            self._client = redis.Redis.from_url(
                self.settings.resolved_url(),
                socket_timeout=self.settings.socket_timeout_seconds,
                socket_connect_timeout=self.settings.socket_timeout_seconds,
            )
            self._client.ping()
        except Exception as exc:  # pragma: no cover - depends on optional service
            LOGGER.debug("Redis cache unavailable; continuing without it: %s", exc)
            self._connect_failed = True
            return None
        return self._client

    def _index_key(self, namespace: str) -> str:
        return f"{self.settings.key_prefix}:lru:{namespace}:__index__"

    def _payload_key(self, namespace: str, payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        return f"{self.settings.key_prefix}:lru:{namespace}:{digest}"

    def _decode(self, raw: Any) -> Any:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))

    def _evict_over_capacity(self, client: Any, namespace: str) -> None:
        max_entries = self.settings.max_entries
        if max_entries is None:
            return
        index_key = self._index_key(namespace)
        try:
            count = int(client.zcard(index_key) or 0)
        except Exception:
            return
        overflow = count - max_entries
        if overflow <= 0:
            return
        try:
            victims = list(client.zrange(index_key, 0, overflow - 1) or [])
            if not victims:
                return
            client.delete(*victims)
            client.zrem(index_key, *victims)
        except Exception:
            return

    def get(self, namespace: str, payload: dict[str, Any]) -> Any | None:
        max_entries = self.settings.max_entries
        if max_entries is not None and max_entries <= 0:
            return None
        client = self.client
        if client is None:
            return None
        key = self._payload_key(namespace, payload)
        try:
            raw = client.get(key)
            if raw is None:
                client.zrem(self._index_key(namespace), key)
                return None
            client.zadd(self._index_key(namespace), {key: time.time()})
            return self._decode(raw)
        except Exception:
            return None

    def set(self, namespace: str, payload: dict[str, Any], value: Any) -> None:
        client = self.client
        if client is None:
            return
        max_entries = self.settings.max_entries
        if max_entries is not None and max_entries <= 0:
            return
        key = self._payload_key(namespace, payload)
        try:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            if self.settings.ttl_seconds is None:
                client.set(key, encoded)
            else:
                client.set(key, encoded, ex=self.settings.ttl_seconds)
            client.zadd(self._index_key(namespace), {key: time.time()})
            self._evict_over_capacity(client, namespace)
        except Exception:
            return

    def invalidate_namespace(self, namespace: str) -> None:
        client = self.client
        if client is None:
            return
        index_key = self._index_key(namespace)
        try:
            keys = list(client.zrange(index_key, 0, -1) or [])
            if keys:
                client.delete(*keys)
            client.delete(index_key)
        except Exception:
            return


def graph_cache_enabled(settings: Any) -> bool:
    redis_settings = getattr(settings, "redis_cache", None)
    if isinstance(redis_settings, dict):
        redis_settings = RedisCacheSettings.from_dict(redis_settings)
    return bool(getattr(redis_settings, "enabled", False) and getattr(redis_settings, "cache_graph_queries", True))


def vector_cache_enabled(settings: Any) -> bool:
    redis_settings = getattr(settings, "redis_cache", None)
    if isinstance(redis_settings, dict):
        redis_settings = RedisCacheSettings.from_dict(redis_settings)
    return bool(getattr(redis_settings, "enabled", False) and getattr(redis_settings, "cache_vector_queries", True))


def invalidate_graph_cache(settings: Any) -> None:
    if not graph_cache_enabled(settings):
        return
    cache = RedisLRUCache.from_settings(settings)
    if cache is not None:
        cache.invalidate_namespace("graph:knowledge")
