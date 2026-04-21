"""In-memory immutable baseline model cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable
import threading

import torch

from ..schemas import CacheStats, import_string, utc_now


LoaderFn = Callable[[str], bytes | bytearray | Any]
UpdateHook = Callable[[str, "CachedModelEntry", dict[str, Any] | None], None]


def default_torch_loader(path: str) -> bytes:
    """Load a torch checkpoint onto CPU and return serialized bytes."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    buffer = BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _materialize_payload_bytes(path: str, loader_fn: LoaderFn | None = None, loader_target: str | None = None) -> bytes:
    loader = loader_fn
    if loader is None and loader_target:
        imported = import_string(loader_target)
        loader = imported
    loader = loader or default_torch_loader
    raw = loader(path)
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    buffer = BytesIO()
    torch.save(raw, buffer)
    return buffer.getvalue()


@dataclass(slots=True)
class CachedModelEntry:
    model_id: str
    baseline_model_path: str
    payload: bytes
    size_bytes: int
    pinned: bool = False
    hits: int = 0
    misses: int = 0
    last_loaded_at: str = field(default_factory=utc_now)
    last_accessed_at: str = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_stats_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "baseline_model_path": self.baseline_model_path,
            "size_bytes": self.size_bytes,
            "pinned": self.pinned,
            "hits": self.hits,
            "misses": self.misses,
            "last_loaded_at": self.last_loaded_at,
            "last_accessed_at": self.last_accessed_at,
            "metadata": dict(self.metadata),
        }


class BaselineModelCache:
    """LRU RAM cache for CPU-side immutable baseline model payloads."""

    def __init__(self, memory_budget_bytes: int, *, on_update: UpdateHook | None = None):
        self.memory_budget_bytes = memory_budget_bytes
        self.on_update = on_update
        self._entries: OrderedDict[str, CachedModelEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._used_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _notify(self, event_name: str, entry: CachedModelEntry, payload: dict[str, Any] | None = None) -> None:
        if self.on_update is not None:
            self.on_update(event_name, entry, payload)

    def _touch(self, entry: CachedModelEntry) -> None:
        entry.last_accessed_at = utc_now()
        self._entries.move_to_end(entry.model_id)

    def _evict_as_needed(self, incoming_size: int) -> None:
        while self._used_bytes + incoming_size > self.memory_budget_bytes and self._entries:
            eviction_candidate_id, eviction_candidate = next(iter(self._entries.items()))
            if eviction_candidate.pinned:
                movable = [entry for entry in self._entries.values() if not entry.pinned]
                if not movable:
                    break
                eviction_candidate = movable[0]
                eviction_candidate_id = eviction_candidate.model_id
            self._entries.pop(eviction_candidate_id)
            self._used_bytes -= eviction_candidate.size_bytes
            self._evictions += 1
            self._notify("cache_evicted", eviction_candidate, {"used_bytes": self._used_bytes})

    def _store_entry(
        self,
        model_id: str,
        baseline_model_path: str,
        payload: bytes,
        *,
        pinned: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> CachedModelEntry | None:
        size_bytes = len(payload)
        if size_bytes > self.memory_budget_bytes:
            return None
        self._evict_as_needed(size_bytes)
        if self._used_bytes + size_bytes > self.memory_budget_bytes:
            return None
        entry = CachedModelEntry(
            model_id=model_id,
            baseline_model_path=baseline_model_path,
            payload=payload,
            size_bytes=size_bytes,
            pinned=pinned,
            misses=1,
            metadata=metadata or {},
        )
        self._entries[model_id] = entry
        self._used_bytes += size_bytes
        self._notify("cache_loaded", entry, {"used_bytes": self._used_bytes})
        return entry

    def preload(
        self,
        model_id: str,
        baseline_model_path: str,
        *,
        loader_fn: LoaderFn | None = None,
        loader_target: str | None = None,
        pin: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        with self._lock:
            existing = self._entries.get(model_id)
            if existing is not None:
                if pin:
                    existing.pinned = True
                self._touch(existing)
                self._notify("cache_touched", existing, {"used_bytes": self._used_bytes})
                return True
            payload = _materialize_payload_bytes(baseline_model_path, loader_fn=loader_fn, loader_target=loader_target)
            entry = self._store_entry(
                model_id,
                baseline_model_path,
                payload,
                pinned=pin,
                metadata=metadata,
            )
            return entry is not None

    def get(
        self,
        model_id: str,
        baseline_model_path: str,
        *,
        loader_fn: LoaderFn | None = None,
        loader_target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bytes:
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is not None:
                entry.hits += 1
                self._hits += 1
                self._touch(entry)
                self._notify("cache_hit", entry, {"used_bytes": self._used_bytes})
                return bytes(entry.payload)

            self._misses += 1
            payload = _materialize_payload_bytes(baseline_model_path, loader_fn=loader_fn, loader_target=loader_target)
            entry = self._store_entry(
                model_id,
                baseline_model_path,
                payload,
                pinned=False,
                metadata=metadata,
            )
            if entry is not None:
                entry.misses = 1
                self._notify("cache_miss", entry, {"used_bytes": self._used_bytes})
                return bytes(entry.payload)

            temp_entry = CachedModelEntry(
                model_id=model_id,
                baseline_model_path=baseline_model_path,
                payload=b"",
                size_bytes=len(payload),
                pinned=False,
                hits=0,
                misses=1,
                metadata=metadata or {},
            )
            self._notify("cache_miss_uncached", temp_entry, {"reason": "payload larger than cache budget"})
            return payload

    def evict(self, model_id: str) -> bool:
        with self._lock:
            entry = self._entries.pop(model_id, None)
            if entry is None:
                return False
            self._used_bytes -= entry.size_bytes
            self._evictions += 1
            self._notify("cache_evicted", entry, {"used_bytes": self._used_bytes})
            return True

    def pin(self, model_id: str) -> bool:
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                return False
            entry.pinned = True
            self._touch(entry)
            self._notify("cache_pinned", entry, None)
            return True

    def unpin(self, model_id: str) -> bool:
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                return False
            entry.pinned = False
            self._touch(entry)
            self._notify("cache_unpinned", entry, None)
            return True

    def snapshot_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [entry.to_stats_dict() for entry in self._entries.values()]

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                entries=len(self._entries),
                pinned_entries=sum(1 for entry in self._entries.values() if entry.pinned),
                used_bytes=self._used_bytes,
                memory_budget_bytes=self.memory_budget_bytes,
            )
