from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
import tempfile
import unittest

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import TrainingJob
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


class _ReadBarrierStore(SQLiteStateStore):
    """Force legacy read-then-write updates to start from the same payload."""

    def __init__(self, settings: SchedulerSettings, barrier: Barrier):
        super().__init__(settings)
        self.read_barrier = barrier
        self.synchronize_reads = False

    def get_job(self, job_id: str):
        job = super().get_job(job_id)
        if self.synchronize_reads:
            self.read_barrier.wait(timeout=5)
        return job


class StateStoreConcurrencyTest(unittest.TestCase):
    def test_concurrent_metadata_updates_do_not_overwrite_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer_count = 8
            settings = SchedulerSettings(runtime_root=Path(tmpdir) / "runtime")
            store = _ReadBarrierStore(settings, Barrier(writer_count))
            job = store.submit_job(
                TrainingJob.create(
                    runner_target="localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
                    baseline_model_id="toy",
                    baseline_model_path=str(Path(tmpdir) / "baseline.pt"),
                )
            )
            store.synchronize_reads = True

            def update(writer_id: int) -> None:
                store.update_job(job.job_id, metadata_updates={f"writer_{writer_id}": writer_id})

            with ThreadPoolExecutor(max_workers=writer_count) as pool:
                list(pool.map(update, range(writer_count)))

            store.synchronize_reads = False
            final = store.get_job(job.job_id)
            self.assertIsNotNone(final)
            assert final is not None
            self.assertEqual(
                {key: final.metadata.get(key) for key in (f"writer_{index}" for index in range(writer_count))},
                {f"writer_{index}": index for index in range(writer_count)},
            )


if __name__ == "__main__":
    unittest.main()
