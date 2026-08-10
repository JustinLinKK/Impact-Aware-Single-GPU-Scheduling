from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from localml_scheduler.adapters.mlevolve import build_mlevolve_job, build_packing_signature
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import (
    BatchSizeObservation,
    CombinationProfile,
    PackingSpec,
    PreloadSource,
    ResourceRequirements,
    TrainingJob,
    build_batch_size_observation_key,
    build_group_signature,
)
from localml_scheduler.execution.backends import MPSBackend
from localml_scheduler.hardware import HardwareProfile, build_hardware_key
from localml_scheduler.scheduler.placement_planner import PlacementPlanner
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.scheduler.service import SchedulerService
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def _fake_hardware_profile(name: str) -> HardwareProfile:
    return HardwareProfile(
        hardware_key=build_hardware_key(
            os_name="linux",
            gpu_name=name,
            total_vram_mb=24576,
            compute_capability="9.0",
            cuda_runtime="12.8",
            torch_version="2.8.0",
        ),
        os_name="linux",
        gpu_name=name,
        total_vram_mb=24576,
        compute_capability="9.0",
        cuda_runtime="12.8",
        torch_version="2.8.0",
    )


class GpuSchedulerUnitTest(unittest.TestCase):
    def test_settings_file_parses_only_time_aware_scheduler_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "scheduler.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        f'runtime_root: "{tmpdir}"',
                        "gpu_scheduler:",
                        '  mode: "parallel_time_aware"',
                        "  parallel_job_cap: 3",
                        '  backend_priority: ["stream", "cuda_process", "exclusive"]',
                        "  memory:",
                        "    gpu_vram_gib: 16",
                        "    predicted_budget_fraction: 0.8",
                        "  colocation:",
                        "    min_gain: 1.05",
                        "  submission_defaults:",
                        "    packing_eligible: true",
                        '    backend_allowlist: ["cuda_process"]',
                        "  telemetry:",
                        "    device_poll_ms: 250",
                    ]
                ),
                encoding="utf-8",
            )

            settings = SchedulerSettings.from_file(settings_path)

            self.assertEqual(settings.gpu_scheduler.mode, "parallel_time_aware")
            self.assertEqual(settings.gpu_scheduler.parallel_job_cap, 3)
            self.assertEqual(settings.gpu_scheduler.memory.gpu_vram_gib, 16)
            self.assertEqual(settings.gpu_scheduler.memory.predicted_budget_fraction, 0.8)
            self.assertEqual(settings.gpu_scheduler.colocation.min_gain, 1.05)
            self.assertEqual(settings.gpu_scheduler.telemetry.device_poll_ms, 250)

    def test_removed_fill_configuration_fails_loudly(self) -> None:
        for payload in (
            {"mode": "parallel_auto_pack"},
            {"max_packed_jobs_per_gpu": 2},
            {"parallel_optimizer": {}},
            {"memory": {"safe_vram_budget_gib": 12.5}},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                SchedulerSettings(gpu_scheduler=payload)

    def test_null_current_nested_scheduler_blocks_get_defaults(self) -> None:
        settings = SchedulerSettings(
            gpu_scheduler={
                "submission_defaults": None,
                "mps": None,
                "cuda_process": None,
                "stream": None,
            }
        )
        self.assertIsNotNone(settings.gpu_scheduler.submission_defaults)
        self.assertIsNotNone(settings.gpu_scheduler.mps)
        self.assertIsNotNone(settings.gpu_scheduler.cuda_process)
        self.assertIsNotNone(settings.gpu_scheduler.stream)

    def test_packing_and_preload_specs_round_trip(self) -> None:
        job = TrainingJob.create(
            "module:runner",
            "baseline-a",
            "/tmp/a.pt",
            packing=PackingSpec(
                eligible=True,
                signature="family:abcd",
                family="family",
                max_slowdown_ratio=1.2,
            ),
            preload_source=PreloadSource(
                model_id="startpoint-shared",
                model_path="/tmp/shared.ckpt",
                loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
            ),
        )

        restored = TrainingJob.from_dict(job.to_dict())

        self.assertTrue(restored.packing.eligible)
        self.assertEqual(restored.packing.signature, "family:abcd")
        self.assertEqual(restored.packing.max_slowdown_ratio, 1.2)
        self.assertEqual(restored.preload_source.model_id, "startpoint-shared")

    def test_signature_generation_is_stable(self) -> None:
        kwargs = {
            "runner_target": "pkg.runner:train",
            "baseline_model_id": "baseline-a",
            "task_type": "classification",
            "max_steps": 100,
            "max_epochs": 3,
            "family": "toy-family",
        }
        left = build_packing_signature(
            **kwargs,
            runner_kwargs={"batch_size": 16, "precision": "bf16"},
        )
        right = build_packing_signature(
            **kwargs,
            runner_kwargs={"precision": "bf16", "batch_size": 16},
        )
        self.assertEqual(left, right)

    def test_mps_availability_requires_linux_binary_and_cuda(self) -> None:
        backend = MPSBackend(
            SchedulerSettings(),
            executor=mock.Mock(),
            mps_binary="/usr/bin/nvidia-cuda-mps-control",
        )
        with mock.patch("localml_scheduler.execution.backends.sys.platform", "win32"), mock.patch(
            "localml_scheduler.execution.backends._cuda_runtime_visible", return_value=True
        ):
            self.assertFalse(backend.available())
        with mock.patch("localml_scheduler.execution.backends.sys.platform", "linux"), mock.patch(
            "localml_scheduler.execution.backends._cuda_runtime_visible", return_value=False
        ):
            self.assertFalse(backend.available())
        with mock.patch("localml_scheduler.execution.backends.sys.platform", "linux"), mock.patch(
            "localml_scheduler.execution.backends._cuda_runtime_visible", return_value=True
        ):
            self.assertTrue(backend.available())

    def test_planner_falls_back_to_one_exclusive_anchor_without_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir)
            store = SQLiteStateStore(settings)
            planner = PlacementPlanner(
                settings,
                store,
                PriorityFifoPolicy(enable_priority_aging=False),
            )
            jobs = []
            for index in range(2):
                job = build_mlevolve_job(
                    workflow_id="wf",
                    baseline_model_id=f"baseline-{index}",
                    baseline_model_path=f"/tmp/{index}.pt",
                    runner_target="pkg.runner:train",
                    packing_family="toy",
                    packing_eligible=True,
                )
                job.queue_sequence = index + 1
                jobs.append(job)

            plan = planner.choose_plan(
                jobs,
                backend_available={"exclusive": True, "cuda_process": True},
            )

            self.assertEqual(plan.mode, "exclusive")
            self.assertEqual(plan.backend_name, "exclusive")
            self.assertEqual(plan.job_ids, (jobs[0].job_id,))
            self.assertIn("runtime estimate unavailable", plan.reason)

    def test_measurements_and_time_aware_profiles_are_hardware_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir)
            store = SQLiteStateStore(settings)
            hardware_a = _fake_hardware_profile("gpu-a")
            hardware_b = _fake_hardware_profile("gpu-b")
            for hardware, peak in ((hardware_a, 2048), (hardware_b, 4096)):
                store.upsert_batch_size_observation(
                    BatchSizeObservation(
                        observation_key=build_batch_size_observation_key(
                            "model-a",
                            "shape-a",
                            hardware.hardware_key,
                            "cuda_process",
                            4,
                        ),
                        model_key="model-a",
                        shape_signature="shape-a",
                        hardware_key=hardware.hardware_key,
                        backend_name="cuda_process",
                        batch_param_name="batch_size",
                        batch_size=4,
                        peak_vram_mb=peak,
                        avg_vram_mb=peak * 0.9,
                    )
                )
            restored = store.get_batch_size_observation(
                model_key="model-a",
                shape_signature="shape-a",
                hardware_key=hardware_a.hardware_key,
                backend_name="cuda_process",
                batch_size=4,
            )
            self.assertEqual(restored.peak_vram_mb, 2048)

            group_signature = build_group_signature(["sig-a", "sig-b"])
            for hardware, score in ((hardware_a, 1.1), (hardware_b, 1.2)):
                store.upsert_combination_profile(
                    CombinationProfile.create(
                        group_signature=group_signature,
                        hardware_key=hardware.hardware_key,
                        backend_name="cuda_process",
                        scheduler_mode="parallel_time_aware",
                        batch_vector={"left": 4, "right": 4},
                        compatible=True,
                        objective_score=score,
                    )
                )
            best = store.best_combination_profile(
                group_signature=group_signature,
                hardware_key=hardware_a.hardware_key,
                backend_name="cuda_process",
                scheduler_mode="parallel_time_aware",
            )
            self.assertEqual(best.objective_score, 1.1)

    def test_warm_cache_prefers_shared_preload_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workspace"
            workdir.mkdir(parents=True)
            shared = workdir / "shared.ckpt"
            shared.write_bytes(b"shared-startpoint")
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                baseline_cache={"warm_queue_top_k": 4, "entry_capacity": 8},
            )
            store = SQLiteStateStore(settings)
            preload = PreloadSource(
                model_id="tree-startpoint",
                model_path=str(shared),
                loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
            )
            for index in range(2):
                script = workdir / f"candidate_{index}.py"
                script.write_text("print('candidate')\n", encoding="utf-8")
                store.submit_job(
                    build_mlevolve_job(
                        workflow_id="wf",
                        baseline_model_id=f"candidate-{index}",
                        baseline_model_path=str(script),
                        runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                        preload_source=preload,
                        resource_requirements=ResourceRequirements(requires_gpu=False),
                    )
                )

            service = SchedulerService(settings, store=store)
            service._warm_cache()

            entries = service.cache.snapshot_entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["model_id"], "tree-startpoint")


if __name__ == "__main__":
    unittest.main()
