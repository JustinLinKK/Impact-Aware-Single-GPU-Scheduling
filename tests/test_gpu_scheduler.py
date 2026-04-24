from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from localml_scheduler.adapters.mlevolve import build_mlevolve_job, build_packing_signature
from localml_scheduler.schemas import PackingSpec, SoloProfile, TrainingJob
from localml_scheduler.scheduler.gpu_scheduler import GpuPlacementPlanner, compatibility_score
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.settings import SchedulerSettings
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


class GpuSchedulerUnitTest(unittest.TestCase):
    def test_settings_file_parses_nested_gpu_scheduler(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "scheduler.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        f'runtime_root: "{tmpdir}"',
                        "gpu_scheduler:",
                        "  enabled: true",
                        '  backend_priority: ["mps", "exclusive"]',
                        "  max_packed_jobs_per_gpu: 2",
                        "  memory:",
                        "    safe_vram_budget_gib: 12.5",
                        "  telemetry:",
                        "    device_poll_ms: 250",
                        "  mps:",
                        "    default_primary_active_thread_pct: 55",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings.from_file(settings_path)
            self.assertTrue(settings.gpu_scheduler.enabled)
            self.assertEqual(settings.gpu_scheduler.memory.safe_vram_budget_gib, 12.5)
            self.assertEqual(settings.gpu_scheduler.telemetry.device_poll_ms, 250)
            self.assertEqual(settings.gpu_scheduler.mps.default_primary_active_thread_pct, 55)

    def test_packing_spec_round_trip(self) -> None:
        job = TrainingJob.create(
            "module:runner",
            "baseline-a",
            "/tmp/a.pt",
            packing=PackingSpec(eligible=True, signature="family:abcd", family="family", max_slowdown_ratio=1.2),
        )
        restored = TrainingJob.from_dict(job.to_dict())
        self.assertTrue(restored.packing.eligible)
        self.assertEqual(restored.packing.signature, "family:abcd")
        self.assertEqual(restored.packing.family, "family")
        self.assertEqual(restored.packing.max_slowdown_ratio, 1.2)

    def test_signature_generation_is_stable(self) -> None:
        left = build_packing_signature(
            runner_target="pkg.runner:train",
            baseline_model_id="baseline-a",
            task_type="classification",
            runner_kwargs={"batch_size": 16, "precision": "bf16"},
            max_steps=100,
            max_epochs=3,
            family="toy-family",
        )
        right = build_packing_signature(
            runner_target="pkg.runner:train",
            baseline_model_id="baseline-a",
            task_type="classification",
            runner_kwargs={"precision": "bf16", "batch_size": 16},
            max_steps=100,
            max_epochs=3,
            family="toy-family",
        )
        self.assertEqual(left, right)

    def test_planner_requires_solo_profiles_and_falls_back_when_mps_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir)
            store = SQLiteStateStore(settings)
            planner = GpuPlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))

            primary = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="baseline-a",
                baseline_model_path="/tmp/a.pt",
                runner_target="pkg.runner:train",
                packing_family="toy",
                packing_eligible=True,
                task_type="classification",
            )
            secondary = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="baseline-b",
                baseline_model_path="/tmp/b.pt",
                runner_target="pkg.runner:train",
                packing_family="toy",
                packing_eligible=True,
                task_type="classification",
            )
            primary.queue_sequence = 1
            secondary.queue_sequence = 2

            plan = planner.choose_plan([primary, secondary], backend_available={"exclusive": True, "mps": False})
            self.assertEqual(plan.mode, "exclusive")
            self.assertIn("unavailable", plan.reason)

            plan = planner.choose_plan([primary, secondary], backend_available={"exclusive": True, "mps": True})
            self.assertEqual(plan.mode, "exclusive")
            self.assertIn("solo profile", plan.reason)

    def test_compatibility_score_prefers_lower_utilization_partner(self) -> None:
        settings = SchedulerSettings(runtime_root=Path(tempfile.mkdtemp()))
        primary = TrainingJob.create("pkg.runner:train", "baseline-a", "/tmp/a.pt", priority=9)
        partner = TrainingJob.create("pkg.runner:train", "baseline-b", "/tmp/b.pt", priority=3)
        primary_profile = SoloProfile(signature="a", peak_vram_mb=2048, avg_gpu_utilization=0.25)
        low_util = SoloProfile(signature="b", peak_vram_mb=2048, avg_gpu_utilization=0.20)
        high_util = SoloProfile(signature="c", peak_vram_mb=2048, avg_gpu_utilization=0.78)
        low_score = compatibility_score(primary, partner, primary_profile, low_util, None, settings)
        high_score = compatibility_score(primary, partner, primary_profile, high_util, None, settings)
        self.assertGreater(low_score, high_score)


if __name__ == "__main__":
    unittest.main()
