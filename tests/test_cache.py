from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from localml_scheduler.client import SchedulerClient
from localml_scheduler.model_cache.baseline_cache import BaselineModelCache, _materialize_payload_bytes
from localml_scheduler.model_cache.cache_server import CacheClient, CacheServer
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.redis_cache import RedisCacheSettings, RedisLRUCache


class _FakeRedis:
    def __init__(self) -> None:
        self.values = {}
        self.sorted_sets = {}

    def ping(self):
        return True

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        del ex
        self.values[key] = value
        return True

    def zadd(self, key, mapping):
        self.sorted_sets.setdefault(key, {}).update(mapping)

    def zcard(self, key):
        return len(self.sorted_sets.get(key, {}))

    def zrange(self, key, start, end):
        items = sorted(self.sorted_sets.get(key, {}).items(), key=lambda item: item[1])
        if end == -1:
            selected = items[start:]
        else:
            selected = items[start : end + 1]
        return [item[0] for item in selected]

    def zrem(self, key, *members):
        bucket = self.sorted_sets.setdefault(key, {})
        for member in members:
            bucket.pop(member, None)

    def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)
            self.sorted_sets.pop(key, None)


class BaselineCacheTest(unittest.TestCase):
    def _make_checkpoint(self, path: Path, value: int) -> None:
        torch.save({"value": value}, path)

    def test_hit_miss_and_eviction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            a = tmp_path / "a.pt"
            b = tmp_path / "b.pt"
            self._make_checkpoint(a, 1)
            self._make_checkpoint(b, 2)

            budget = len(_materialize_payload_bytes(str(a))) + 16
            cache = BaselineModelCache(memory_budget_bytes=budget)
            payload_a_first = cache.get("a", str(a))
            payload_a_second = cache.get("a", str(a))
            self.assertEqual(payload_a_first, payload_a_second)
            self.assertEqual(cache.stats().hits, 1)
            self.assertEqual(cache.stats().misses, 1)

            cache.preload("b", str(b))
            stats = cache.stats()
            self.assertGreaterEqual(stats.evictions, 1)
            self.assertEqual(stats.entries, 1)

    def test_pinned_entry_survives_eviction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            a = tmp_path / "a.pt"
            b = tmp_path / "b.pt"
            self._make_checkpoint(a, 1)
            self._make_checkpoint(b, 2)

            budget = len(_materialize_payload_bytes(str(a))) + 16
            cache = BaselineModelCache(memory_budget_bytes=budget)
            cache.preload("a", str(a), pin=True)
            cache.preload("b", str(b))
            entries = {entry["model_id"] for entry in cache.snapshot_entries()}
            self.assertIn("a", entries)

    def test_entry_capacity_evicts_least_recently_used_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            a = tmp_path / "a.pt"
            b = tmp_path / "b.pt"
            c = tmp_path / "c.pt"
            self._make_checkpoint(a, 1)
            self._make_checkpoint(b, 2)
            self._make_checkpoint(c, 3)

            cache = BaselineModelCache(memory_budget_bytes=1024 * 1024, entry_capacity=2)
            cache.preload("a", str(a))
            cache.preload("b", str(b))
            cache.get("a", str(a))
            cache.preload("c", str(c))

            entries = [entry["model_id"] for entry in cache.snapshot_entries()]
            self.assertEqual(entries, ["a", "c"])

    def test_get_refreshes_lru_recency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            a = tmp_path / "a.pt"
            b = tmp_path / "b.pt"
            c = tmp_path / "c.pt"
            self._make_checkpoint(a, 1)
            self._make_checkpoint(b, 2)
            self._make_checkpoint(c, 3)

            cache = BaselineModelCache(memory_budget_bytes=1024 * 1024, entry_capacity=2)
            cache.preload("a", str(a))
            cache.preload("b", str(b))
            cache.get("a", str(a))
            cache.preload("c", str(c))

            entries = {entry["model_id"] for entry in cache.snapshot_entries()}
            self.assertEqual(entries, {"a", "c"})

    def test_preload_refreshes_lru_recency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            a = tmp_path / "a.pt"
            b = tmp_path / "b.pt"
            c = tmp_path / "c.pt"
            self._make_checkpoint(a, 1)
            self._make_checkpoint(b, 2)
            self._make_checkpoint(c, 3)

            cache = BaselineModelCache(memory_budget_bytes=1024 * 1024, entry_capacity=2)
            cache.preload("a", str(a))
            cache.preload("b", str(b))
            cache.preload("a", str(a))
            cache.preload("c", str(c))

            entries = {entry["model_id"] for entry in cache.snapshot_entries()}
            self.assertEqual(entries, {"a", "c"})

    def test_ram_percent_cap_can_be_tighter_than_byte_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            a = tmp_path / "a.pt"
            b = tmp_path / "b.pt"
            self._make_checkpoint(a, 1)
            self._make_checkpoint(b, 2)

            payload_size = len(_materialize_payload_bytes(str(a)))
            fake_total_memory = payload_size * 4
            with mock.patch(
                "localml_scheduler.model_cache.baseline_cache.psutil.virtual_memory",
                return_value=SimpleNamespace(total=fake_total_memory),
            ):
                cache = BaselineModelCache(
                    memory_budget_bytes=payload_size * 3,
                    max_ram_percent=0.40,
                )
                cache.preload("a", str(a))
                cache.preload("b", str(b))

            stats = cache.stats()
            self.assertLess(stats.effective_memory_budget_bytes, stats.memory_budget_bytes)
            self.assertEqual(stats.entries, 1)

    def test_preload_fails_when_only_pinned_entries_can_be_evicted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            a = tmp_path / "a.pt"
            b = tmp_path / "b.pt"
            self._make_checkpoint(a, 1)
            self._make_checkpoint(b, 2)

            cache = BaselineModelCache(memory_budget_bytes=1024 * 1024, entry_capacity=1)
            self.assertTrue(cache.preload("a", str(a), pin=True))
            self.assertFalse(cache.preload("b", str(b)))
            entries = [entry["model_id"] for entry in cache.snapshot_entries()]
            self.assertEqual(entries, ["a"])

    def test_payload_larger_than_effective_budget_is_not_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            a = tmp_path / "a.pt"
            self._make_checkpoint(a, 1)

            payload_size = len(_materialize_payload_bytes(str(a)))
            cache = BaselineModelCache(memory_budget_bytes=payload_size - 1)
            self.assertFalse(cache.preload("a", str(a)))
            self.assertEqual(cache.stats().entries, 0)

    def test_api_cache_stats_reports_extended_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                baseline_cache={"entry_capacity": 8, "max_ram_percent": 0.2, "warm_queue_top_k": 3},
            )
            api = SchedulerClient(settings)
            stats = api.cache_stats()["stats"]
            self.assertIn("entry_capacity", stats)
            self.assertIn("max_ram_percent", stats)
            self.assertIn("system_total_memory_bytes", stats)
            self.assertIn("effective_memory_budget_bytes", stats)
            self.assertEqual(stats["entry_capacity"], 8)
            self.assertEqual(stats["max_ram_percent"], 0.2)

    def test_cache_server_uses_short_socket_for_deep_runtime_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_root = Path(tmpdir) / ("nested" * 12) / ("workspace" * 12)
            settings = SchedulerSettings(runtime_root=deep_root)
            address = settings.cache_address()

            if sys.platform == "win32":
                self.assertIsInstance(address, tuple)
                self.assertEqual(address[0], settings.cache_server_host)
                self.assertEqual(address[1], settings.cache_server_port)
            else:
                self.assertIsInstance(address, str)
                self.assertLess(len(address.encode("utf-8")), 100)
                self.assertNotIn(str(deep_root), address)

            server = CacheServer(settings, BaselineModelCache(memory_budget_bytes=1024 * 1024))
            try:
                server.start()
                self.assertTrue(CacheClient(settings).ping())
            finally:
                server.stop()
            if isinstance(address, str):
                self.assertFalse(Path(address).exists())

    def test_redis_lru_cache_evicts_least_recently_used_entry(self) -> None:
        cache = RedisLRUCache(
            RedisCacheSettings(enabled=True, key_prefix="test", max_entries=2, ttl_seconds=60),
            redis_client=_FakeRedis(),
        )

        with mock.patch("localml_scheduler.redis_cache.time.time", side_effect=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            cache.set("vector:test", {"id": "a"}, {"value": 1})
            cache.set("vector:test", {"id": "b"}, {"value": 2})
            self.assertEqual(cache.get("vector:test", {"id": "a"}), {"value": 1})
            cache.set("vector:test", {"id": "c"}, {"value": 3})

            self.assertIsNone(cache.get("vector:test", {"id": "b"}))
            self.assertEqual(cache.get("vector:test", {"id": "a"}), {"value": 1})
            self.assertEqual(cache.get("vector:test", {"id": "c"}), {"value": 3})


if __name__ == "__main__":
    unittest.main()
