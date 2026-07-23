from __future__ import annotations

import tempfile
import unittest

from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerConfig
from localml_scheduler.domain import BatchProbeProfile, BatchResolution, JobRun, RuntimeProfile, TrainingJob
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

    def test_graph_recommendation_surface_uses_persisted_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerConfig(runtime_root=tmpdir)
            client = SchedulerClient(settings)
            job = client.submit(
                TrainingJob.create(
                    "module:runner",
                    "baseline-a",
                    "/tmp/a.pt",
                    max_epochs=4,
                    runner_kwargs={"batch_size": 2},
                )
            )
            client.upsert_batch_probe_profile(
                BatchProbeProfile(
                    probe_key="probe-1",
                    model_key="baseline-a",
                    device_type="test-gpu",
                    shape_signature="shape-1",
                    batch_param_name="batch_size",
                    resolved_batch_size=8,
                    observations=2,
                    last_job_id=job.job_id,
                )
            )
            client.upsert_runtime_profile(
                RuntimeProfile.create(
                    signature=job.packing.signature or job.job_id,
                    hardware_key=client.store.hardware_key(),
                    backend_name="exclusive",
                    resolved_batch_size=8,
                    strategy="epoch_1",
                    epoch_1_seconds=12.0,
                    estimated_total_runtime_seconds=48.0,
                    confidence=0.9,
                    observations=1,
                    last_job_id=job.job_id,
                )
            )

            recommendation = client.recommend_batch_size(
                model_or_signature="baseline-a",
                hardware="test-gpu",
                shape_signature="shape-1",
                current_batch_size=2,
            )
            runtime_estimate = client.get_runtime_estimate(
                model_or_signature=job.packing.signature or job.job_id,
                batch_size=8,
                backend="exclusive",
            )
            tuning_outcome = client.record_tuning_outcome(
                job_id=job.job_id,
                chosen_batch_size=8,
                chosen_epochs=4,
                recommendation_source="unit-test",
                outcome_metrics={"loss": 0.1},
            )

            self.assertTrue(recommendation["found"])
            self.assertEqual(recommendation["recommended_batch_size"], 8)
            self.assertTrue(runtime_estimate["found"])
            self.assertTrue(tuning_outcome["ok"])
            self.assertEqual(len(client.list_events(job_id=job.job_id, event_type="tuning_outcome_recorded")), 1)


if __name__ == "__main__":
    unittest.main()
