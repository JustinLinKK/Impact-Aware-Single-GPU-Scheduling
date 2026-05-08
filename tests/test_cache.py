from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from localml_scheduler.client import SchedulerClient
from localml_scheduler.model_cache.baseline_cache import BaselineModelCache, _materialize_payload_bytes
from localml_scheduler.config import SchedulerSettings


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


if __name__ == "__main__":
    unittest.main()
