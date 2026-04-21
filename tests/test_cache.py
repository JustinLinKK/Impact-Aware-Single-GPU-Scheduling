from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch

from localml_scheduler.cache.baseline_cache import BaselineModelCache, _materialize_payload_bytes


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


if __name__ == "__main__":
    unittest.main()
