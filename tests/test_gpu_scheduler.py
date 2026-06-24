from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from localml_scheduler.adapters.mlevolve import build_mlevolve_job, build_packing_signature
from localml_scheduler.execution.backends import MPSBackend, StreamMPSBackend
from localml_scheduler.hardware import HardwareProfile, build_hardware_key
from localml_scheduler.domain import (
    BatchSizeObservation,
    BatchProbeProfile,
    CombinationProfile,
    PackingSpec,
    PlacementDecision,
    PreloadSource,
    ResourceRequirements,
    SoloProfile,
    TrainingJob,
    build_batch_probe_key,
    build_batch_size_observation_key,
    build_group_signature,
)
from localml_scheduler.scheduler.compatibility import compatibility_score
from localml_scheduler.scheduler.placement_planner import PlacementPlanner
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.scheduler.planner_types import DispatchPlan
from localml_scheduler.scheduler.service import SchedulerService
from localml_scheduler.config import (
    SCHEDULER_MODE_AUTO,
    SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
    SCHEDULER_MODE_PARALLEL_AUTO_PACK,
    SCHEDULER_MODE_PARALLEL_DEFAULT,
    SchedulerSettings,
    effective_scheduler_mode,
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


def _planner(settings: SchedulerSettings, store: SQLiteStateStore | None = None) -> tuple[SQLiteStateStore, PlacementPlanner]:
    scheduler_store = store or SQLiteStateStore(settings)
    return scheduler_store, PlacementPlanner(settings, scheduler_store, PriorityFifoPolicy(enable_priority_aging=False))


class _FakeParallelSupervisor:
    def __init__(self, *, available_backends: dict[str, bool]):
        self._available_backends = dict(available_backends)
        self.dispatched_jobs: list[TrainingJob] = []

    def active_group(self):
        return None

    def available_backends(self) -> dict[str, bool]:
        return dict(self._available_backends)

    def dispatch(
        self,
        jobs: list[TrainingJob],
        *,
        mode: str,
        backend_name: str,
        batch_overrides: dict[str, int] | None = None,
        fallback_order: list[str] | None = None,
    ) -> PlacementDecision:
        self.dispatched_jobs = list(jobs)
        return PlacementDecision(
            can_run=True,
            reason="fake dispatch",
            gpu_slot=0,
            mode=mode,
            backend_name=backend_name,
            job_ids=[job.job_id for job in jobs],
            batch_overrides=dict(batch_overrides or {}),
            fallback_order=list(fallback_order or []),
        )


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
                        "    vram_budget_fraction: 0.95",
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
            self.assertEqual(settings.gpu_scheduler.memory.vram_budget_fraction, 0.95)
            self.assertEqual(settings.gpu_scheduler.telemetry.device_poll_ms, 250)
            self.assertEqual(settings.gpu_scheduler.mps.default_primary_active_thread_pct, 55)

    def test_default_scheduler_mode_is_auto_with_stream_mps_priority(self) -> None:
        settings = SchedulerSettings(runtime_root=Path(tempfile.mkdtemp()))
        self.assertEqual(settings.gpu_scheduler.mode, SCHEDULER_MODE_AUTO)
        self.assertEqual(effective_scheduler_mode(settings.gpu_scheduler.mode), SCHEDULER_MODE_PARALLEL_AUTO_PACK)
        self.assertEqual(settings.gpu_scheduler.backend_priority, ["stream_mps", "stream", "cuda_process", "mps", "exclusive"])
        self.assertEqual(settings.gpu_scheduler.concurrent_backend_allowlist, ["stream_mps", "stream"])
        self.assertEqual(settings.gpu_scheduler.submission_defaults.backend_allowlist, [])

    def test_auto_mode_boot_probe_resolves_backend_priority_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": "auto",
                    "backend_priority": ["mps", "exclusive"],
                    "stream": {"enabled": False},
                    "mps": {"enabled": False},
                    "cuda_process": {"enabled": False},
                },
            )
            store = SQLiteStateStore(settings)
            supervisor = _FakeParallelSupervisor(
                available_backends={
                    "exclusive": True,
                    "stream_mps": True,
                    "stream": True,
                    "cuda_process": True,
                    "mps": True,
                }
            )

            SchedulerService(settings, store=store, supervisor=supervisor)

            self.assertTrue(settings.gpu_scheduler.stream.enabled)
            self.assertTrue(settings.gpu_scheduler.mps.enabled)
            self.assertTrue(settings.gpu_scheduler.cuda_process.enabled)
            self.assertEqual(settings.gpu_scheduler.backend_priority, ["stream_mps", "stream", "cuda_process", "mps", "exclusive"])
            self.assertEqual(settings.gpu_scheduler.concurrent_backend_allowlist, ["stream_mps", "stream"])
            events = store.list_events(event_type="scheduler_auto_backend_probe")
            self.assertEqual(len(events), 1)
            payload = events[0]["payload"]
            self.assertEqual(payload["configured_mode"], "auto")
            self.assertEqual(payload["effective_scheduler_mode"], SCHEDULER_MODE_PARALLEL_AUTO_PACK)
            self.assertEqual(payload["backend_priority"], ["stream_mps", "stream", "cuda_process", "mps", "exclusive"])

    def test_explicit_scheduler_mode_preserves_configured_backend_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_PARALLEL_DEFAULT,
                    "backend_priority": ["cuda_process", "exclusive"],
                    "concurrent_backend_allowlist": ["cuda_process"],
                    "stream": {"enabled": False},
                    "mps": {"enabled": False},
                },
            )
            store = SQLiteStateStore(settings)
            supervisor = _FakeParallelSupervisor(
                available_backends={
                    "exclusive": True,
                    "stream_mps": True,
                    "stream": True,
                    "cuda_process": True,
                    "mps": True,
                }
            )

            SchedulerService(settings, store=store, supervisor=supervisor)

            self.assertEqual(settings.gpu_scheduler.mode, SCHEDULER_MODE_PARALLEL_DEFAULT)
            self.assertEqual(effective_scheduler_mode(settings.gpu_scheduler.mode), SCHEDULER_MODE_PARALLEL_DEFAULT)
            self.assertFalse(settings.gpu_scheduler.stream.enabled)
            self.assertFalse(settings.gpu_scheduler.mps.enabled)
            self.assertEqual(settings.gpu_scheduler.backend_priority, ["cuda_process", "exclusive"])
            self.assertEqual(settings.gpu_scheduler.concurrent_backend_allowlist, ["cuda_process"])
            self.assertEqual(store.list_events(event_type="scheduler_auto_backend_probe"), [])

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
                        "    binary_range_up: 10",
                        "    binary_range_down: 2",
                        "    power_of_two_range_up: 4",
                        "    power_of_two_range_down: 1",
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
            self.assertEqual(settings.gpu_scheduler.parallel_optimizer.binary_range_up, 10)
            self.assertEqual(settings.gpu_scheduler.parallel_optimizer.binary_range_down, 2)
            self.assertEqual(settings.gpu_scheduler.parallel_optimizer.power_of_two_range_up, 4)
            self.assertEqual(settings.gpu_scheduler.parallel_optimizer.power_of_two_range_down, 1)
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
            self.assertEqual(settings.gpu_scheduler.submission_defaults.backend_allowlist, [])
            self.assertIsNotNone(settings.gpu_scheduler.parallel_optimizer)
            self.assertIsNotNone(settings.gpu_scheduler.mps)
            self.assertIsNotNone(settings.gpu_scheduler.cuda_process)
            self.assertIsNotNone(settings.gpu_scheduler.stream)

    def test_legacy_vram_budget_keys_load_into_canonical_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "scheduler.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        f'runtime_root: "{tmpdir}"',
                        "gpu_scheduler:",
                        "  batch_probe_target_memory_fraction: 0.91",
                        "  auto_pack:",
                        "    target_vram_fraction: 0.92",
                        "  parallel_optimizer:",
                        "    target_vram_fraction: 0.93",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings.from_file(settings_path)
            self.assertEqual(settings.gpu_scheduler.memory.vram_budget_fraction, 0.91)
            emitted = settings.to_dict()["gpu_scheduler"]
            self.assertNotIn("batch_probe_target_memory_fraction", emitted)
            self.assertNotIn("target_vram_fraction", emitted["auto_pack"])
            self.assertNotIn("target_vram_fraction", emitted["parallel_optimizer"])

    def test_default_vram_budget_is_95_percent_of_detected_vram(self) -> None:
        settings = SchedulerSettings(runtime_root=Path(tempfile.mkdtemp()))
        self.assertEqual(settings.gpu_scheduler.memory.vram_budget_fraction, 0.95)
        self.assertEqual(settings.gpu_scheduler.memory.budget_mb(32768), 31129.6)

    def test_settings_file_parses_baseline_cache_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "scheduler.yaml"
            settings_path.write_text(
                "\n".join(
                    [
                        f'runtime_root: "{tmpdir}"',
                        "baseline_cache:",
                        "  warm_queue_policy: budget_only",
                        "  warm_queue_top_k: 5",
                        "  entry_capacity: 8",
                        "  max_ram_percent: 0.2",
                        "  memory_budget_bytes: 123456",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings.from_file(settings_path)
            self.assertEqual(settings.baseline_cache.warm_queue_policy, "budget_only")
            self.assertEqual(settings.baseline_cache.warm_queue_top_k, 5)
            self.assertEqual(settings.baseline_cache.entry_capacity, 8)
            self.assertEqual(settings.baseline_cache.max_ram_percent, 0.2)
            self.assertEqual(settings.baseline_cache.memory_budget_bytes, 123456)

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

    def test_preload_source_round_trip(self) -> None:
        job = TrainingJob.create(
            "module:runner",
            "baseline-a",
            "/tmp/a.pt",
            preload_source=PreloadSource(
                model_id="startpoint-shared",
                model_path="/tmp/shared.ckpt",
                loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
            ),
        )
        restored = TrainingJob.from_dict(job.to_dict())
        self.assertIsNotNone(restored.preload_source)
        self.assertEqual(restored.preload_source.model_id, "startpoint-shared")
        self.assertEqual(restored.preload_source.model_path, "/tmp/shared.ckpt")

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

    def test_stream_mps_backend_availability_requires_stream_mps_binary_and_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "stream": {"enabled": True},
                    "mps": {"enabled": True},
                },
            )
            backend = StreamMPSBackend(settings, executor=mock.Mock(), mps_binary="/usr/bin/nvidia-cuda-mps-control")

            with mock.patch("localml_scheduler.execution.backends.sys.platform", "linux"), mock.patch(
                "localml_scheduler.execution.backends._cuda_runtime_visible",
                return_value=True,
            ):
                self.assertTrue(backend.available())

            settings.gpu_scheduler.stream.enabled = False
            with mock.patch("localml_scheduler.execution.backends.sys.platform", "linux"), mock.patch(
                "localml_scheduler.execution.backends._cuda_runtime_visible",
                return_value=True,
            ):
                self.assertFalse(backend.available())

    def test_planner_requires_solo_profiles_and_falls_back_when_mps_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir, gpu_scheduler={"mode": SCHEDULER_MODE_PARALLEL_DEFAULT})
            store = SQLiteStateStore(settings)
            planner = PlacementPlanner(settings, store, PriorityFifoPolicy(enable_priority_aging=False))

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

    def test_planner_honors_raw_job_backend_allowlist_from_scheduler_config(self) -> None:
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
                packing_backend_allowlist=["stream", "cuda_process"],
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
                packing_backend_allowlist=["stream", "cuda_process"],
                task_type="mlevolve_script",
            )
            raw_a.queue_sequence = 1
            raw_b.queue_sequence = 2

            raw_plan = planner.choose_plan(
                [raw_a, raw_b],
                backend_available={"exclusive": True, "stream": True, "cuda_process": True},
            )
            self.assertEqual(raw_plan.mode, "packed_pair")
            self.assertEqual(raw_plan.backend_name, "stream")

    def test_parallel_auto_pack_uses_batch_probe_memory_without_runtime_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_PARALLEL_AUTO_PACK,
                    "backend_priority": ["cuda_process", "exclusive"],
                    "memory": {"vram_budget_fraction": 0.95},
                },
            )
            store, planner = _planner(settings)
            store._hardware_profile = _fake_hardware_profile("auto-pack-probed")

            jobs = []
            for index, model_id in enumerate(["auto-probe-a", "auto-probe-b"], start=1):
                job = build_mlevolve_job(
                    workflow_id="wf",
                    baseline_model_id=model_id,
                    baseline_model_path=f"/tmp/{model_id}.py",
                    runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                    runner_kwargs={"batch_size": 4},
                    resource_requirements=ResourceRequirements(requires_gpu=True),
                    packing_family="mlevolve_script",
                    packing_eligible=True,
                    packing_backend_allowlist=["cuda_process"],
                    task_type="mlevolve_script",
                )
                job.queue_sequence = index
                jobs.append(job)
                shape_signature = planner._shape_signature(job)
                probe_key = build_batch_probe_key(
                    model_id,
                    store.hardware_profile().gpu_name,
                    shape_signature,
                    search_mode=settings.gpu_scheduler.batch_probe_search_mode,
                )
                store.upsert_batch_probe_profile(
                    BatchProbeProfile(
                        probe_key=probe_key,
                        model_key=model_id,
                        device_type=store.hardware_profile().gpu_name,
                        shape_signature=shape_signature,
                        batch_param_name="batch_size",
                        resolved_batch_size=4,
                        peak_vram_mb=1024,
                        memory_total_mb=24576,
                        last_job_id=job.job_id,
                    )
                )

            plan = planner.choose_plan(jobs, backend_available={"exclusive": True, "cuda_process": True})

            self.assertEqual(plan.mode, "packed_pair")
            self.assertEqual(plan.backend_name, "cuda_process")
            self.assertEqual(plan.reason, "auto-pack group selected")

    def test_parallel_auto_pack_missing_memory_estimates_dispatches_calibration_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_PARALLEL_AUTO_PACK,
                    "backend_priority": ["cuda_process", "exclusive"],
                },
            )
            _store, planner = _planner(settings)
            first = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="missing-a",
                baseline_model_path="/tmp/a.py",
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                runner_kwargs={"batch_size": 4},
                resource_requirements=ResourceRequirements(requires_gpu=True),
                packing_family="mlevolve_script",
                packing_eligible=True,
                packing_backend_allowlist=["cuda_process"],
                task_type="mlevolve_script",
            )
            second = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="missing-b",
                baseline_model_path="/tmp/b.py",
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                runner_kwargs={"batch_size": 4},
                resource_requirements=ResourceRequirements(requires_gpu=True),
                packing_family="mlevolve_script",
                packing_eligible=True,
                packing_backend_allowlist=["cuda_process"],
                task_type="mlevolve_script",
            )
            first.queue_sequence = 1
            second.queue_sequence = 2

            plan = planner.choose_plan([first, second], backend_available={"exclusive": True, "cuda_process": True})

            self.assertEqual(plan.mode, "exclusive")
            self.assertIn("calibration probe", plan.reason)

    def test_parallel_auto_pack_group_size_uses_candidate_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_PARALLEL_AUTO_PACK,
                    "backend_priority": ["cuda_process", "exclusive"],
                    "candidate_window_size": 4,
                    "max_packed_jobs_per_gpu": 2,
                },
            )
            store, planner = _planner(settings)
            store._hardware_profile = _fake_hardware_profile("auto-pack-size")
            jobs: list[TrainingJob] = []
            for index in range(5):
                job = build_mlevolve_job(
                    workflow_id="wf",
                    baseline_model_id=f"auto-size-{index}",
                    baseline_model_path=f"/tmp/auto-size-{index}.pt",
                    runner_target="pkg.runner:train",
                    runner_kwargs={"batch_size": 4},
                    resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=256),
                    packing_family="toy",
                    packing_eligible=True,
                    packing_backend_allowlist=["cuda_process"],
                    task_type="classification",
                )
                job.queue_sequence = index + 1
                jobs.append(job)

            plan = planner.choose_plan(jobs, backend_available={"exclusive": True, "cuda_process": True})

            self.assertEqual(plan.mode, "packed_group")
            self.assertEqual(len(plan.job_ids), 4)
            self.assertEqual(
                planner.last_decision_trace["candidate_group_sizing"],
                {"window_size": 4, "max_group_size": 4, "include_singletons": True},
            )

    def test_raw_mlevolve_jobs_are_not_preemptible_without_checkpoint_resume_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=tmpdir)
            service = SchedulerService(settings, store=SQLiteStateStore(settings))
            raw = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="raw",
                baseline_model_path="/tmp/raw.py",
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                runner_kwargs={"batch_size": 4},
                task_type="mlevolve_script",
            )
            structured = TrainingJob.create(
                "pkg.runner:train",
                "structured",
                "/tmp/structured.pt",
                task_type="classification",
            )

            self.assertFalse(service._supports_safe_preemption(raw))
            raw.metadata["supports_checkpoint_resume"] = True
            self.assertTrue(service._supports_safe_preemption(raw))
            self.assertTrue(service._supports_safe_preemption(structured))

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

    def test_parallel_default_group_size_respects_fixed_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_PARALLEL_DEFAULT,
                    "backend_priority": ["cuda_process", "exclusive"],
                    "candidate_window_size": 4,
                    "max_packed_jobs_per_gpu": 2,
                    "allow_three_way_packing": False,
                },
            )
            store, planner = _planner(settings)
            store._hardware_profile = _fake_hardware_profile("fixed-pack-size")
            jobs: list[TrainingJob] = []
            for index in range(4):
                job = build_mlevolve_job(
                    workflow_id="wf",
                    baseline_model_id=f"fixed-size-{index}",
                    baseline_model_path=f"/tmp/fixed-size-{index}.pt",
                    runner_target="pkg.runner:train",
                    runner_kwargs={"batch_size": 4},
                    resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=256),
                    packing_family="toy",
                    packing_eligible=True,
                    task_type="classification",
                )
                job.queue_sequence = index + 1
                jobs.append(job)

            plan = planner.choose_plan(jobs, backend_available={"exclusive": True, "cuda_process": True})

            self.assertEqual(plan.mode, "packed_pair")
            self.assertEqual(len(plan.job_ids), 2)
            self.assertEqual(
                planner.last_decision_trace["candidate_group_sizing"],
                {"window_size": 4, "max_group_size": 2, "include_singletons": False},
            )

    def test_parallel_batch_optimizer_binary_finds_best_safe_batch_vector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_PARALLEL_BATCH_OPTIMIZED,
                    "backend_priority": ["cuda_process", "exclusive"],
                    "memory": {"vram_budget_fraction": 0.03125},
                    "parallel_optimizer": {
                        "batch_search_mode": "binary",
                        "binary_range_up": 3,
                        "binary_range_down": 1,
                    },
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
                    "memory": {"vram_budget_fraction": 0.0375},
                    "parallel_optimizer": {
                        "batch_search_mode": "power_of_two",
                        "power_of_two_range_up": 2,
                        "power_of_two_range_down": 0,
                    },
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

    def test_parallel_batch_optimizer_binary_thresholds_clip_candidate_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "parallel_optimizer": {
                        "batch_search_mode": "binary",
                        "binary_range_up": 4,
                        "binary_range_down": 3,
                    }
                },
            )
            _, planner = _planner(settings)
            job = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="binary-threshold",
                baseline_model_path="/tmp/binary-threshold.pt",
                runner_target="pkg.runner:train",
                runner_kwargs={"batch_size": 8, "probe_max_batch_size": 10},
                resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=256),
                packing_family="toy",
                packing_eligible=True,
                task_type="classification",
            )

            self.assertEqual(planner._candidate_batch_sizes(job), [5, 6, 7, 8, 9, 10])

    def test_parallel_batch_optimizer_power_of_two_thresholds_clip_candidate_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(
                runtime_root=tmpdir,
                gpu_scheduler={
                    "parallel_optimizer": {
                        "batch_search_mode": "power_of_two",
                        "power_of_two_range_up": 2,
                        "power_of_two_range_down": 1,
                    }
                },
            )
            _, planner = _planner(settings)
            job = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="pow2-threshold",
                baseline_model_path="/tmp/pow2-threshold.pt",
                runner_target="pkg.runner:train",
                runner_kwargs={"batch_size": 12, "probe_max_batch_size": 20},
                resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=256),
                packing_family="toy",
                packing_eligible=True,
                task_type="classification",
            )

            self.assertEqual(planner._candidate_batch_sizes(job), [4, 8, 16])

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

    def test_scheduler_warm_cache_prefers_job_preload_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workspace"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                baseline_cache={"warm_queue_top_k": 4, "entry_capacity": 8},
            )
            store = SQLiteStateStore(settings)
            shared_startpoint = workdir / "shared-start.ckpt"
            shared_startpoint.write_bytes(b"shared-startpoint")
            script_a = workdir / "candidate_a.py"
            script_b = workdir / "candidate_b.py"
            script_a.write_text("print('a')\n", encoding="utf-8")
            script_b.write_text("print('b')\n", encoding="utf-8")

            preload_source = PreloadSource(
                model_id="tree-startpoint",
                model_path=str(shared_startpoint),
                loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
            )
            for index, script_path in enumerate((script_a, script_b), start=1):
                job = build_mlevolve_job(
                    workflow_id="wf",
                    baseline_model_id=f"mlevolve-script-{index}",
                    baseline_model_path=str(script_path),
                    runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                    runner_kwargs={
                        "script_path": str(script_path),
                        "working_dir": str(workdir),
                        "result_path": str(workdir / f"result_{index}.json"),
                        "timeout": 30,
                    },
                    task_type="mlevolve_script",
                    loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
                    preload_source=preload_source,
                    resource_requirements=ResourceRequirements(requires_gpu=False),
                    priority=10 - index,
                )
                store.submit_job(job)

            service = SchedulerService(settings, store=store)
            service._warm_cache()
            entries = service.cache.snapshot_entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["model_id"], "tree-startpoint")
            self.assertEqual(entries[0]["baseline_model_path"], str(shared_startpoint))

    def test_scheduler_warm_cache_budget_only_fills_by_memory_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workspace"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                baseline_cache={
                    "warm_queue_policy": "budget_only",
                    "warm_queue_top_k": 1,
                    "entry_capacity": 8,
                    "memory_budget_bytes": 10,
                },
            )
            store = SQLiteStateStore(settings)
            shared_paths = []
            for index, size in enumerate((4, 4, 4), start=1):
                shared_startpoint = workdir / f"shared_{index}.ckpt"
                shared_startpoint.write_bytes(b"x" * size)
                shared_paths.append(shared_startpoint)
                script_path = workdir / f"candidate_{index}.py"
                script_path.write_text(f"print('{index}')\n", encoding="utf-8")
                preload_source = PreloadSource(
                    model_id=f"tree-startpoint-{index}",
                    model_path=str(shared_startpoint),
                    loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
                )
                job = build_mlevolve_job(
                    workflow_id="wf",
                    baseline_model_id=f"mlevolve-script-{index}",
                    baseline_model_path=str(script_path),
                    runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                    runner_kwargs={
                        "script_path": str(script_path),
                        "working_dir": str(workdir),
                        "result_path": str(workdir / f"result_{index}.json"),
                        "timeout": 30,
                    },
                    task_type="mlevolve_script",
                    loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
                    preload_source=preload_source,
                    resource_requirements=ResourceRequirements(requires_gpu=False),
                    priority=10 - index,
                )
                store.submit_job(job)

            service = SchedulerService(settings, store=store)
            service._warm_cache()
            entries = service.cache.snapshot_entries()
            self.assertEqual(len(entries), 2)
            self.assertEqual([entry["model_id"] for entry in entries], ["tree-startpoint-1", "tree-startpoint-2"])

    def test_dispatch_preload_uses_resolved_preload_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workspace"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(runtime_root=runtime_root, baseline_cache={"entry_capacity": 8})
            store = SQLiteStateStore(settings)
            script_path = workdir / "candidate.py"
            script_path.write_text("print('script')\n", encoding="utf-8")
            shared_startpoint = workdir / "shared-start.ckpt"
            shared_startpoint.write_bytes(b"shared-startpoint")
            job = build_mlevolve_job(
                workflow_id="wf",
                baseline_model_id="mlevolve-script",
                baseline_model_path=str(script_path),
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(workdir),
                    "result_path": str(workdir / "result.json"),
                    "timeout": 30,
                },
                task_type="mlevolve_script",
                loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
                preload_source=PreloadSource(
                    model_id="tree-startpoint",
                    model_path=str(shared_startpoint),
                    loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
                ),
                resource_requirements=ResourceRequirements(requires_gpu=False),
            )
            service = SchedulerService(settings, store=store)
            service._preload_job_baseline(job)
            entries = service.cache.snapshot_entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["model_id"], "tree-startpoint")
            self.assertEqual(entries[0]["baseline_model_path"], str(shared_startpoint))

    def test_parallel_dispatch_prefetches_shared_preload_target_once_for_packed_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workspace"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                gpu_scheduler={"backend_priority": ["cuda_process", "exclusive"]},
                baseline_cache={"entry_capacity": 8, "warm_queue_top_k": 0},
            )
            store = SQLiteStateStore(settings)
            shared_startpoint = workdir / "shared-start.ckpt"
            shared_startpoint.write_bytes(b"shared-startpoint")
            preload_source = PreloadSource(
                model_id="tree-startpoint",
                model_path=str(shared_startpoint),
                loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
            )
            jobs: list[TrainingJob] = []
            for index in range(2):
                script_path = workdir / f"candidate_{index}.py"
                script_path.write_text(f"print('{index}')\n", encoding="utf-8")
                job = build_mlevolve_job(
                    workflow_id="wf",
                    baseline_model_id=f"mlevolve-script-{index}",
                    baseline_model_path=str(script_path),
                    runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                    runner_kwargs={
                        "script_path": str(script_path),
                        "working_dir": str(workdir),
                        "result_path": str(workdir / f"result_{index}.json"),
                        "timeout": 30,
                    },
                    task_type="mlevolve_script",
                    loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
                    preload_source=preload_source,
                    resource_requirements=ResourceRequirements(requires_gpu=False, estimated_vram_mb=256),
                    packing_family="mlevolve_script",
                    packing_eligible=True,
                )
                store.submit_job(job)
                jobs.append(job)

            fake_supervisor = _FakeParallelSupervisor(available_backends={"exclusive": True, "cuda_process": True})
            service = SchedulerService(settings, store=store, supervisor=fake_supervisor)
            service.planner = mock.Mock()
            service.planner.choose_plan.return_value = DispatchPlan(
                mode="packed_pair",
                backend_name="cuda_process",
                job_ids=tuple(job.job_id for job in jobs),
                reason="test packed prefetch",
                batch_overrides={},
                fallback_order=[jobs[1].job_id],
            )

            service._dispatch_if_idle()

            entries = service.cache.snapshot_entries()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["model_id"], "tree-startpoint")
            self.assertEqual(fake_supervisor.dispatched_jobs[0].job_id, jobs[0].job_id)
            self.assertEqual(fake_supervisor.dispatched_jobs[1].job_id, jobs[1].job_id)
            cache_touch_events = store.list_events(event_type="cache_touched")
            self.assertEqual(len(cache_touch_events), 1)
            cache_loaded_events = store.list_events(event_type="cache_loaded")
            self.assertEqual(len(cache_loaded_events), 1)


if __name__ == "__main__":
    unittest.main()
