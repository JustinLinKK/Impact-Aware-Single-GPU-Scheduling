from __future__ import annotations

import tempfile
import unittest

from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerConfig
from localml_scheduler.domain import BatchResolution, JobRun, TrainingJob
from localml_scheduler.dto import SubmitJobRequest


class SchedulerClientSurfaceTest(unittest.TestCase):
    def test_submit_job_request_accepts_split_spec_and_run_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            client = SchedulerClient(settings)
            seed_job = TrainingJob.create(
                "module:runner",
                "baseline-a",
                "/tmp/a.pt",
                priority=4,
                runner_kwargs={"batch_size": 2},
            )
            request = SubmitJobRequest(
                spec=seed_job.to_job_spec(),
                run=JobRun(metadata={"source": "client-test"}),
            )

            stored = client.submit(request)

            self.assertEqual(stored.job_id, seed_job.job_id)
            self.assertEqual(stored.priority, 4)
            self.assertEqual(stored.metadata["source"], "client-test")
            self.assertIsNotNone(client.inspect(stored.job_id))

    def test_batch_resolution_apply_updates_runner_kwargs_and_metadata(self) -> None:
        job = TrainingJob.create(
            "module:runner",
            "baseline-a",
            "/tmp/a.pt",
            runner_kwargs={"batch_size": 2},
        )

        updated = BatchResolution.apply(job, 8)

        self.assertEqual(updated.config.runner_kwargs["batch_size"], 8)
        self.assertEqual(updated.metadata["resolved_batch_size"], 8)
        self.assertEqual(updated.metadata["placement_batch_param_name"], "batch_size")
        self.assertEqual(BatchResolution.resolved_batch_size(updated), 8)


if __name__ == "__main__":
    unittest.main()
