from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
import tempfile
import unittest

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import ProgressSnapshot
from localml_scheduler.execution.control import ControlPlane


class AtomicJsonWriteTest(unittest.TestCase):
    def test_concurrent_heartbeat_writers_do_not_share_a_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=Path(tmpdir) / "runtime")
            control = ControlPlane(settings)
            job_id = "concurrent-heartbeat"
            writer_count = 12
            barrier = Barrier(writer_count)

            def write_heartbeats(writer_id: int) -> None:
                barrier.wait()
                for step in range(20):
                    control.write_heartbeat(
                        ProgressSnapshot(
                            job_id=job_id,
                            epoch=writer_id,
                            global_step=(writer_id * 100) + step,
                            message=f"writer-{writer_id}-" + ("x" * 16_384),
                        )
                    )

            with ThreadPoolExecutor(max_workers=writer_count) as pool:
                list(pool.map(write_heartbeats, range(writer_count)))

            heartbeat = control.read_heartbeat(job_id)
            self.assertIsNotNone(heartbeat)
            self.assertEqual(heartbeat.job_id, job_id)
            self.assertEqual(
                list(settings.job_runtime_dir(job_id).glob(".heartbeat.json.*.tmp")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
