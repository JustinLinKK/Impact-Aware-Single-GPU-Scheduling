from __future__ import annotations

import unittest

from scheduler_benchmark_test.standard import STRESS_TEST_DATA_JOB_COUNT
from scheduler_benchmark_test.standard.stress_test_data import (
    A10_ARTIFACT,
    BLACKWELL_ARTIFACT,
    verify_predictions,
    write_fixture,
)


class StressTestDataFixtureTest(unittest.TestCase):
    def test_fixture_is_deterministic_and_all_models_predict_on_both_artifacts(self) -> None:
        self.assertEqual(write_fixture(check=True), [])
        for artifact in (A10_ARTIFACT, BLACKWELL_ARTIFACT):
            with self.subTest(artifact=artifact.name):
                report = verify_predictions(batch_size=2, artifact_path=artifact)
                self.assertTrue(report["accepted"])
                self.assertEqual(report["job_count"], STRESS_TEST_DATA_JOB_COUNT)
                self.assertEqual(
                    report["finite_positive_prediction_count"],
                    STRESS_TEST_DATA_JOB_COUNT,
                )
                self.assertTrue(report["cpu_only"])
                self.assertEqual(
                    report["cuda_allocation_after"],
                    report["cuda_allocation_before"],
                )


if __name__ == "__main__":
    unittest.main()
