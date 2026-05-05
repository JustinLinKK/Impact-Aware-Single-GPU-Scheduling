from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from localml_scheduler.adapters.mlevolve import build_mlevolve_job, build_packing_signature
from localml_scheduler.execution.backends import MPSBackend
from localml_scheduler.hardware import HardwareProfile, build_hardware_key
from localml_scheduler.schemas import (
    BatchSizeObservation,
    CombinationProfile,
    PackingSpec,
    ResourceRequirements,
    SoloProfile,
    TrainingJob,
    build_batch_size_observation_key,
    build_group_signature,
)
from localml_scheduler.scheduler.gpu_scheduler import GpuPlacementPlanner, compatibility_score
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.settings import (
    SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
    SCHEDULER_MODE_PARALLEL_DEFAULT,
    SchedulerSettings,
)
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


def _planner(settings: SchedulerSettings, store: SQLiteStateStore | None = None) -> tuple[SQLiteStateStore, GpuPlacementPlanner]:
    scheduler_store = store or SQLiteStateStore(settings)
    return scheduler_store, GpuPlacementPlanner(settings, scheduler_store, PriorityFifoPolicy(enable_priority_aging=False))


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

    def test_settings_file_parses_modes_optimizer_and_submission_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "scheduler.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        f'runtime_root: "{tmpdir}"',
                        "gpu_scheduler:",
                        '  mode: "parallel_batch_optimized"',
                        "  candidate_window_size: 12",
                        '  backend_priority: ["stream", "cuda_process", "exclusive"]',
                        "  parallel_optimizer:",
                        '    batch_search_mode: "power_of_two"',
                        "    target_vram_fraction: 0.9",
                        "  submission_defaults:",
                        "    packing_eligible: true",
                        '    backend_allowlist: ["cuda_process"]',
                        '    batch_probe_model_key: "scheduler-default"',
                        "  cuda_process:",
                        "    default_omp_num_threads: 4",
                        "  stream:",
                        "    enabled: true",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings.from_file(settings_path)
            self.assertEqual(settings.gpu_scheduler.mode, SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED)
            self.assertEqual(settings.gpu_scheduler.candidate_window_size, 12)
            self.assertEqual(settings.gpu_scheduler.parallel_optimizer.batch_search_mode, "power_of_two")
            self.assertEqual(settings.gpu_scheduler.parallel_optimizer.target_vram_fraction, 0.9)
            self.assertTrue(settings.gpu_scheduler.submission_defaults.packing_eligible)
            self.assertEqual(settings.gpu_scheduler.submission_defaults.backend_allowlist, ["cuda_process"])
            self.assertEqual(settings.gpu_scheduler.submission_defaults.batch_probe_model_key, "scheduler-default")
            self.assertTrue(settings.gpu_scheduler.stream.enabled)
            self.assertEqual(settings.gpu_scheduler.cuda_process.default_omp_num_threads, 4)

    def test_settings_file_tolerates_null_nested_scheduler_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "scheduler.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        f'runtime_root: "{tmpdir}"',
                        "gpu_scheduler:",
                        "  submission_defaults: null",
                        "  parallel_optimizer: null",
                        "  mps: null",
                        "  cuda_process: null",
                        "  stream: null",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings.from_file(settings_path)
            self.assertIsNotNone(settings.gpu_scheduler.submission_defaults)
            self.assertEqual(settings.gpu_scheduler.submission_defaults.backend_allowlist, ["mps", "cuda_process"])
            self.assertIsNotNone(settings.gpu_scheduler.parallel_optimizer)
            self.assertIsNotNone(settings.gpu_scheduler.mps)
            self.assertIsNotNone(settings.gpu_scheduler.cuda_process)
            self.assertIsNotNone(settings.gpu_scheduler.stream)

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

    def test_mps_backend_availability_requires_supported_platform_binary_and_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir)
            backend = MPSBackend(settings, executor=mock.Mock(), mps_binary="/usr/bin/nvidia-cuda-mps-control")

            with mock.patch("localml_scheduler.execution.backends.sys.platform", "win32"), mock.patch(
                "localml_scheduler.execution.backends._cuda_runtime_visible",
                return_value=True,
            ):
                self.assertFalse(backend.available())

            with mock.patch("localml_scheduler.execution.backends.sys.platform", "linux"), mock.patch(
                "localml_scheduler.execution.backends._cuda_runtime_visible",
                return_value=False,
            ):
                self.assertFalse(backend.available())

            with mock.patch("localml_scheduler.execution.backends.sys.platform", "linux"), mock.patch(
                "localml_scheduler.execution.backends._cuda_runtime_visible",
                return_value=True,
            ):
                self.assertTrue(backend.available())

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

    def test_planner_prefers_stream_for_structured_jobs_and_reroutes_raw_jobs_to_cuda_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_PARALLEL_DEFAULT,
                    "backend_priority": ["stream", "cuda_process", "exclusive"],
                },
            )
            store, planner = _planner(settings)

            structured_a = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="structured-a",
                baseline_model_path="/tmp/a.pt",
                runner_target="pkg.runner:train",
                runner_kwargs={"batch_size": 4},
                resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=512),
                packing_family="toy",
                packing_eligible=True,
                task_type="classification",
            )
            structured_b = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="structured-b",
                baseline_model_path="/tmp/b.pt",
                runner_target="pkg.runner:train",
                runner_kwargs={"batch_size": 4},
                resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=512),
                packing_family="toy",
                packing_eligible=True,
                task_type="classification",
            )
            structured_a.queue_sequence = 1
            structured_b.queue_sequence = 2

            structured_plan = planner.choose_plan(
                [structured_a, structured_b],
                backend_available={"exclusive": True, "stream": True, "cuda_process": True},
            )
            self.assertEqual(structured_plan.mode, "packed_pair")
            self.assertEqual(structured_plan.backend_name, "stream")

            raw_a = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="raw-a",
                baseline_model_path="/tmp/raw-a.py",
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                runner_kwargs={"batch_size": 4},
                resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=512),
                packing_family="mlevolve_script",
                packing_eligible=True,
                packing_backend_allowlist=["mps", "cuda_process"],
                task_type="mlevolve_script",
            )
            raw_b = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="raw-b",
                baseline_model_path="/tmp/raw-b.py",
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                runner_kwargs={"batch_size": 4},
                resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=512),
                packing_family="mlevolve_script",
                packing_eligible=True,
                packing_backend_allowlist=["mps", "cuda_process"],
                task_type="mlevolve_script",
            )
            raw_a.queue_sequence = 1
            raw_b.queue_sequence = 2

            raw_plan = planner.choose_plan(
                [raw_a, raw_b],
                backend_available={"exclusive": True, "stream": True, "cuda_process": True},
            )
            self.assertEqual(raw_plan.mode, "packed_pair")
            self.assertEqual(raw_plan.backend_name, "cuda_process")

    def test_parallel_default_prefers_three_way_cuda_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_PARALLEL_DEFAULT,
                    "backend_priority": ["cuda_process", "exclusive"],
                    "max_packed_jobs_per_gpu": 3,
                    "allow_three_way_packing": True,
                },
            )
            _, planner = _planner(settings)
            jobs: list[TrainingJob] = []
            for index in range(3):
                job = build_mlevolve_job(
                    workflow_id="wf",
                    baseline_model_id=f"baseline-{index}",
                    baseline_model_path=f"/tmp/{index}.pt",
                    runner_target="pkg.runner:train",
                    runner_kwargs={"batch_size": 4 + index},
                    resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=384),
                    packing_family="toy",
                    packing_eligible=True,
                    task_type="classification",
                )
                job.queue_sequence = index + 1
                jobs.append(job)

            plan = planner.choose_plan(jobs, backend_available={"exclusive": True, "cuda_process": True})
            self.assertEqual(plan.mode, "packed_group")
            self.assertEqual(plan.backend_name, "cuda_process")
            self.assertEqual(len(plan.job_ids), 3)

    def test_parallel_batch_optimizer_binary_finds_best_safe_batch_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
                    "backend_priority": ["cuda_process", "exclusive"],
                    "memory": {"safe_vram_budget_gib": 0.75},
                    "parallel_optimizer": {"batch_search_mode": "binary", "target_vram_fraction": 1.0},
                },
            )
            store, planner = _planner(settings)
            store._hardware_profile = _fake_hardware_profile("planner-binary")

            jobs: list[TrainingJob] = []
            peaks = {
                "job-a": {1: 100, 2: 200, 3: 300, 4: 400, 5: 500},
                "job-b": {1: 120, 2: 240, 3: 360, 4: 480, 5: 600},
            }
            for index, model_id in enumerate(["job-a", "job-b"], start=1):
                job = build_mlevolve_job(
                    workflow_id="wf",
                    baseline_model_id=model_id,
                    baseline_model_path=f"/tmp/{model_id}.pt",
                    runner_target="pkg.runner:train",
                    runner_kwargs={"batch_size": 2, "probe_max_batch_size": 5},
                    resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=256),
                    packing_family="toy",
                    packing_eligible=True,
                    task_type="classification",
                )
                job.queue_sequence = index
                jobs.append(job)
                shape_signature = planner._shape_signature(job)
                for batch_size, peak in peaks[model_id].items():
                    store.upsert_batch_size_observation(
                        BatchSizeObservation(
                            observation_key=build_batch_size_observation_key(
                                model_id,
                                shape_signature,
                                store.hardware_key(),
                                "cuda_process",
                                batch_size,
                            ),
                            model_key=model_id,
                            shape_signature=shape_signature,
                            hardware_key=store.hardware_key(),
                            backend_name="cuda_process",
                            batch_param_name="batch_size",
                            batch_size=batch_size,
                            peak_vram_mb=peak,
                            last_job_id=job.job_id,
                        )
                    )

            plan = planner.choose_plan(jobs, backend_available={"exclusive": True, "cuda_process": True})
            self.assertEqual(plan.mode, "packed_pair")
            self.assertEqual(plan.backend_name, "cuda_process")
            self.assertEqual(plan.batch_overrides[jobs[0].job_id], 4)
            self.assertEqual(plan.batch_overrides[jobs[1].job_id], 3)

    def test_parallel_batch_optimizer_power_of_two_restricts_batch_vector_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
                    "backend_priority": ["cuda_process", "exclusive"],
                    "memory": {"safe_vram_budget_gib": 0.9},
                    "parallel_optimizer": {"batch_search_mode": "power_of_two", "target_vram_fraction": 1.0},
                },
            )
            store, planner = _planner(settings)
            store._hardware_profile = _fake_hardware_profile("planner-pow2")

            jobs: list[TrainingJob] = []
            for index, model_id in enumerate(["pow2-a", "pow2-b"], start=1):
                job = build_mlevolve_job(
                    workflow_id="wf",
                    baseline_model_id=model_id,
                    baseline_model_path=f"/tmp/{model_id}.pt",
                    runner_target="pkg.runner:train",
                    runner_kwargs={"batch_size": 2},
                    resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=220),
                    packing_family="toy",
                    packing_eligible=True,
                    task_type="classification",
                )
                job.queue_sequence = index
                jobs.append(job)
                shape_signature = planner._shape_signature(job)
                for batch_size, peak in {1: 110, 2: 220, 4: 440, 8: 880}.items():
                    store.upsert_batch_size_observation(
                        BatchSizeObservation(
                            observation_key=build_batch_size_observation_key(
                                model_id,
                                shape_signature,
                                store.hardware_key(),
                                "cuda_process",
                                batch_size,
                            ),
                            model_key=model_id,
                            shape_signature=shape_signature,
                            hardware_key=store.hardware_key(),
                            backend_name="cuda_process",
                            batch_param_name="batch_size",
                            batch_size=batch_size,
                            peak_vram_mb=peak,
                            last_job_id=job.job_id,
                        )
                    )

            plan = planner.choose_plan(jobs, backend_available={"exclusive": True, "cuda_process": True})
            self.assertEqual(plan.mode, "packed_pair")
            self.assertEqual(plan.batch_overrides[jobs[0].job_id], 4)
            self.assertEqual(plan.batch_overrides[jobs[1].job_id], 4)

    def test_parallel_batch_optimizer_uses_cached_optimal_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
                    "backend_priority": ["cuda_process", "exclusive"],
                },
            )
            store, planner = _planner(settings)
            store._hardware_profile = _fake_hardware_profile("planner-cache")

            jobs: list[TrainingJob] = []
            for index in range(2):
                job = build_mlevolve_job(
                    workflow_id="wf",
                    baseline_model_id=f"cached-{index}",
                    baseline_model_path=f"/tmp/cached-{index}.pt",
                    runner_target="pkg.runner:train",
                    runner_kwargs={"batch_size": 2},
                    resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=256),
                    packing_family="toy",
                    packing_eligible=True,
                    task_type="classification",
                )
                job.queue_sequence = index + 1
                jobs.append(job)

            group_signature = build_group_signature([job.packing.signature or job.job_id for job in jobs])
            store.upsert_combination_profile(
                CombinationProfile.create(
                    group_signature=group_signature,
                    hardware_key=store.hardware_key(),
                    backend_name="cuda_process",
                    scheduler_mode=SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
                    batch_vector={jobs[0].job_id: 4, jobs[1].job_id: 8},
                    compatible=True,
                    resolved_optimal=True,
                    objective_score=0.98,
                    peak_vram_mb=900,
                    fallback_order=[jobs[1].job_id, jobs[0].job_id],
                )
            )

            plan = planner.choose_plan(jobs, backend_available={"exclusive": True, "cuda_process": True})
            self.assertEqual(plan.mode, "packed_pair")
            self.assertEqual(plan.reason, "cached optimal packed group selected")
            self.assertEqual(plan.batch_overrides[jobs[0].job_id], 4)
            self.assertEqual(plan.batch_overrides[jobs[1].job_id], 8)
            self.assertEqual(plan.fallback_order, [jobs[1].job_id, jobs[0].job_id])

    def test_batch_size_and_combination_profiles_are_hardware_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir)
            store = SQLiteStateStore(settings)
            hardware_a = _fake_hardware_profile("gpu-a")
            hardware_b = _fake_hardware_profile("gpu-b")
            store._hardware_profile = hardware_a

            observation = BatchSizeObservation(
                observation_key=build_batch_size_observation_key("model-a", "shape-a", hardware_a.hardware_key, "cuda_process", 4),
                model_key="model-a",
                shape_signature="shape-a",
                hardware_key=hardware_a.hardware_key,
                backend_name="cuda_process",
                batch_param_name="batch_size",
                batch_size=4,
                peak_vram_mb=2048,
                observations=2,
            )
            other_observation = BatchSizeObservation(
                observation_key=build_batch_size_observation_key("model-a", "shape-a", hardware_b.hardware_key, "cuda_process", 4),
                model_key="model-a",
                shape_signature="shape-a",
                hardware_key=hardware_b.hardware_key,
                backend_name="cuda_process",
                batch_param_name="batch_size",
                batch_size=4,
                peak_vram_mb=4096,
            )
            store.upsert_batch_size_observation(observation)
            store.upsert_batch_size_observation(other_observation)

            restored = store.get_batch_size_observation(
                model_key="model-a",
                shape_signature="shape-a",
                hardware_key=hardware_a.hardware_key,
                backend_name="cuda_process",
                batch_size=4,
            )
            self.assertIsNotNone(restored)
            self.assertEqual(restored.peak_vram_mb, 2048)
            self.assertEqual(len(store.list_batch_size_observations(hardware_key=hardware_a.hardware_key)), 1)

            group_signature = build_group_signature(["sig-a", "sig-b"])
            store.upsert_combination_profile(
                CombinationProfile.create(
                    group_signature=group_signature,
                    hardware_key=hardware_a.hardware_key,
                    backend_name="cuda_process",
                    scheduler_mode=SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
                    batch_vector={"left": 4, "right": 3},
                    compatible=True,
                    resolved_optimal=False,
                    objective_score=0.91,
                )
            )
            store.upsert_combination_profile(
                CombinationProfile.create(
                    group_signature=group_signature,
                    hardware_key=hardware_a.hardware_key,
                    backend_name="cuda_process",
                    scheduler_mode=SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
                    batch_vector={"left": 5, "right": 2},
                    compatible=True,
                    resolved_optimal=True,
                    objective_score=0.87,
                )
            )
            store.upsert_combination_profile(
                CombinationProfile.create(
                    group_signature=group_signature,
                    hardware_key=hardware_b.hardware_key,
                    backend_name="cuda_process",
                    scheduler_mode=SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
                    batch_vector={"left": 8, "right": 1},
                    compatible=True,
                    resolved_optimal=True,
                    objective_score=0.99,
                )
            )
            store.upsert_combination_profile(
                CombinationProfile.create(
                    group_signature=group_signature,
                    hardware_key=hardware_a.hardware_key,
                    backend_name="cuda_process",
                    scheduler_mode=SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
                    batch_vector={"left": 6, "right": 1},
                    compatible=False,
                    resolved_optimal=True,
                    objective_score=1.0,
                    last_failure_reason="oom",
                )
            )

            best = store.best_combination_profile(
                group_signature=group_signature,
                hardware_key=hardware_a.hardware_key,
                backend_name="cuda_process",
                scheduler_mode=SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
            )
            self.assertIsNotNone(best)
            self.assertEqual(best.batch_vector, {"left": 5, "right": 2})
            self.assertEqual(len(store.list_combination_profiles(hardware_key=hardware_a.hardware_key)), 3)


if __name__ == "__main__":
    unittest.main()
