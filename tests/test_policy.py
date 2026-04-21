from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from localml_scheduler.schemas import TrainingJob
from localml_scheduler.scheduler.policies import PriorityFifoPolicy


class PriorityPolicyTest(unittest.TestCase):
    def test_fifo_with_equal_priority(self) -> None:
        job_a = TrainingJob.create("module:runner", "baseline-a", "/tmp/a.pt", priority=3)
        job_b = TrainingJob.create("module:runner", "baseline-b", "/tmp/b.pt", priority=3)
        job_a.queue_sequence = 1
        job_b.queue_sequence = 2
        policy = PriorityFifoPolicy(enable_priority_aging=False)
        self.assertEqual(policy.choose([job_b, job_a]).job_id, job_a.job_id)

    def test_aging_can_overtake_newer_job(self) -> None:
        old_job = TrainingJob.create("module:runner", "baseline-a", "/tmp/a.pt", priority=1)
        new_job = TrainingJob.create("module:runner", "baseline-b", "/tmp/b.pt", priority=3)
        old_job.queue_sequence = 1
        new_job.queue_sequence = 2
        old_job.submitted_at = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        policy = PriorityFifoPolicy(aging_interval_seconds=30, aging_priority_increment=1, enable_priority_aging=True)
        self.assertEqual(policy.choose([new_job, old_job]).job_id, old_job.job_id)


if __name__ == "__main__":
    unittest.main()
