from __future__ import annotations

import unittest

from localml_scheduler.schemas import TrainingJob
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.scheduler.queue import RunnableJobQueue


class RunnableQueueTest(unittest.TestCase):
    def test_higher_priority_first(self) -> None:
        low = TrainingJob.create("module:runner", "baseline-a", "/tmp/a.pt", priority=1)
        high = TrainingJob.create("module:runner", "baseline-b", "/tmp/b.pt", priority=9)
        low.queue_sequence = 1
        high.queue_sequence = 2
        queue = RunnableJobQueue(PriorityFifoPolicy(enable_priority_aging=False), [low, high])
        self.assertEqual(queue.peek().job_id, high.job_id)

    def test_pop_removes_first_job(self) -> None:
        first = TrainingJob.create("module:runner", "baseline-a", "/tmp/a.pt", priority=3)
        second = TrainingJob.create("module:runner", "baseline-b", "/tmp/b.pt", priority=2)
        first.queue_sequence = 1
        second.queue_sequence = 2
        queue = RunnableJobQueue(PriorityFifoPolicy(enable_priority_aging=False), [first, second])
        popped = queue.pop()
        self.assertEqual(popped.job_id, first.job_id)
        self.assertEqual(queue.peek().job_id, second.job_id)


if __name__ == "__main__":
    unittest.main()
