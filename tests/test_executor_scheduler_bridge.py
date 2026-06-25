from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

from engine.executor import Interpreter
from localml_scheduler.adapters.mlevolve import build_model_family_profile_key
from localml_scheduler.domain import BatchProbeProfile, JobStatus, TrainingJob
from localml_scheduler.config import SchedulerSettings


class _FakeStore:
    def list_events(self, *, job_id: str | None = None) -> list[dict[str, object]]:
        return []


class _FakeSchedulerClient:
    def __init__(self, settings: SchedulerSettings, *, active_service: bool = True, create_probe_profiles: bool = True):
        self.settings = settings
        self.store = _FakeStore()
        self.submitted_jobs: list[TrainingJob] = []
        self._jobs: dict[str, TrainingJob] = {}
        self.batch_probe_profiles: dict[str, BatchProbeProfile] = {}
        self.active_service = active_service
        self.create_probe_profiles = create_probe_profiles
        self.create_service_calls = 0
        self.submit_many_calls = 0

    def submit(self, job: TrainingJob) -> TrainingJob:
        self.submitted_jobs.append(job)
        if job.task_type == "mlevolve_model_family_probe":
            if self.create_probe_profiles and job.batch_probe.profile_key:
                self.batch_probe_profiles[job.batch_probe.profile_key] = BatchProbeProfile(
                    probe_key=job.batch_probe.profile_key,
                    model_key=job.batch_probe.model_key or job.metadata.get("model_family") or "family",
                    device_type="test-gpu",
                    shape_signature=job.batch_probe.shape_signature_override or "family-shape",
                    batch_param_name=job.batch_probe.batch_param_name,
                    resolved_batch_size=8,
                    peak_vram_mb=1024,
                    memory_total_mb=4096,
                    target_budget_mb=3891,
                )
                job.metadata["resolved_batch_size"] = 8
            job.mark_status(JobStatus.COMPLETED if self.create_probe_profiles else JobStatus.FAILED)
            self._jobs[job.job_id] = job
            return job

        result_path = Path(job.config.runner_kwargs["result_path"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "term_out": ["scheduler bridge ok"],
                    "exec_time": 0.01,
                    "exc_type": None,
                    "exc_info": {},
                    "exc_stack": [],
                }
            ),
            encoding="utf-8",
        )
        job.mark_status(JobStatus.COMPLETED)
        self._jobs[job.job_id] = job
        return job

    def submit_many(self, jobs: list[TrainingJob]) -> list[TrainingJob]:
        self.submit_many_calls += 1
        submitted = []
        for job in jobs:
            submitted.append(self.submit(job))
        return submitted

    def plan_job_packet(self, *, candidates, limit=8):
        return {
            "packet_id": "packet-test",
            "jobs": [{"candidate": candidate, "optimization_context": {"confidence": 0.0}} for candidate in candidates],
            "evidence_refs": [],
            "confidence": 0.0,
        }

    def record_tuning_outcome(self, **kwargs):
        self.last_tuning_outcome = kwargs
        return {"ok": True}

    def get_batch_probe_profile(self, profile_key: str) -> BatchProbeProfile | None:
        return self.batch_probe_profiles.get(profile_key)

    def inspect(self, job_id: str) -> TrainingJob | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.mark_status(JobStatus.CANCELLED, reason="cancelled")

    def scheduler_service_active(self, *, max_staleness_seconds: float | None = None) -> bool:
        return self.active_service

    def create_service(self):
        self.create_service_calls += 1

        api = self

        class _FakeService:
            def start(self, *, background: bool = False):
                api.active_service = True
                return self

            def stop(self):
                api.active_service = False

        return _FakeService()


class InterpreterSchedulerBridgeTest(unittest.TestCase):
    def test_scheduler_submission_uses_scheduler_defaults_and_preload_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                gpu_scheduler={
                    "submission_defaults": {
                        "requires_gpu": True,
                        "estimated_vram_mb": 4096,
                        "estimated_ram_mb": 2048,
                        "packing_eligible": True,
                        "packing_family": "scheduler-owned-family",
                        "packing_max_slowdown_ratio": 1.15,
                        "backend_allowlist": ["cuda_process"],
                        "batch_probe_enabled": True,
                        "batch_probe_model_key": "scheduler-model-key",
                        "batch_probe_probe_timeout_seconds": 11,
                        "batch_probe_poll_interval_seconds": 0.25,
                        "batch_probe_max_multiplier": 4,
                        "batch_probe_search_mode": "power_of_two",
                    }
                },
            )
            interpreter = Interpreter(working_dir=workdir, timeout=10, max_parallel_run=1)
            fake_api = _FakeSchedulerClient(settings)
            bridge_cfg = SimpleNamespace(
                wait_timeout_seconds=5,
                wait_poll_interval_seconds=0.01,
                preload_source_model_id="shared-startpoint",
                preload_source_model_path=str(workdir / "shared-start.ckpt"),
                preload_source_loader_target="localml_scheduler.adapters.mlevolve_runner:load_raw_file",
            )
            (workdir / "shared-start.ckpt").write_bytes(b"checkpoint")
            interpreter.attach_scheduler(fake_api, bridge_cfg)

            result = interpreter._run_scheduler_job(
                code="batch_size = 3\nprint('hello from bridge')\n",
                id="node-1",
                working_dir=str(workdir),
            )

            self.assertIsNone(result.exc_type)
            self.assertEqual(result.term_out, ["scheduler bridge ok"])
            self.assertEqual(len(fake_api.submitted_jobs), 1)

            submitted = fake_api.submitted_jobs[0]
            self.assertTrue(submitted.resource_requirements.requires_gpu)
            self.assertEqual(submitted.resource_requirements.estimated_vram_mb, 4096)
            self.assertEqual(submitted.resource_requirements.estimated_ram_mb, 2048)
            self.assertTrue(submitted.packing.eligible)
            self.assertEqual(submitted.packing.family, "scheduler-owned-family")
            self.assertEqual(submitted.packing.max_slowdown_ratio, 1.15)
            self.assertEqual(submitted.packing.backend_allowlist, ["cuda_process"])
            self.assertTrue(submitted.batch_probe.enabled)
            self.assertEqual(submitted.batch_probe.model_key, "scheduler-model-key")
            self.assertEqual(submitted.batch_probe.search_mode, "power_of_two")
            self.assertEqual(submitted.config.runner_kwargs["probe_timeout_seconds"], 11)
            self.assertEqual(submitted.config.runner_kwargs["probe_poll_interval_seconds"], 0.25)
            self.assertEqual(submitted.config.runner_kwargs["probe_max_batch_size"], 8)
            self.assertTrue(submitted.baseline_model_id.startswith("mlevolve-script-"))
            self.assertIsNotNone(submitted.preload_source)
            self.assertEqual(submitted.preload_source.model_id, "shared-startpoint")
            self.assertEqual(submitted.preload_source.model_path, str(workdir / "shared-start.ckpt"))
            self.assertEqual(
                submitted.preload_source.loader_target,
                "localml_scheduler.adapters.mlevolve_runner:load_raw_file",
            )

    def test_scheduler_submission_reuses_model_family_probe_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(runtime_root=runtime_root)
            interpreter = Interpreter(working_dir=workdir, timeout=10, max_parallel_run=1)
            fake_api = _FakeSchedulerClient(settings)
            profile_key = build_model_family_profile_key(task_id="unit-task", model_family="safe-family")
            fake_api.batch_probe_profiles[profile_key] = BatchProbeProfile(
                probe_key=profile_key,
                model_key="safe-family",
                device_type="test-gpu",
                shape_signature="shape-safe-family",
                batch_param_name="batch_size",
                resolved_batch_size=8,
                peak_vram_mb=1024,
                memory_total_mb=4096,
                target_budget_mb=3891,
            )
            cfg = SimpleNamespace(
                start_cpu_id=0,
                cpu_number=1,
                exp_id="unit-task",
                exp_name="unit-exp",
                experiment=SimpleNamespace(mode="hardware_aware"),
                agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=1)),
            )
            interpreter.cfg = cfg
            interpreter.attach_scheduler(fake_api, SimpleNamespace(wait_timeout_seconds=5, wait_poll_interval_seconds=0.01))

            result = interpreter._run_scheduler_job(
                code="MODEL_FAMILY = 'safe-family'\nbatch_size = 6\nprint('family reuse')\n",
                id="node-family",
                working_dir=str(workdir),
            )

            self.assertIsNone(result.exc_type)
            submitted = fake_api.submitted_jobs[0]
            self.assertEqual(submitted.batch_probe.model_key, "safe-family")
            self.assertEqual(submitted.batch_probe.profile_key, profile_key)
            self.assertTrue(submitted.batch_probe.reuse_only)
            self.assertEqual(submitted.config.runner_kwargs["batch_size"], 4)
            self.assertEqual(submitted.metadata["resolved_batch_size"], 4)
            self.assertEqual(submitted.metadata["batch_probe_source"], "model_family_profile")

    def test_scheduler_submission_probes_unseen_model_family_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(runtime_root=runtime_root)
            cfg = SimpleNamespace(
                start_cpu_id=0,
                cpu_number=1,
                exp_id="unit-task",
                exp_name="unit-exp",
                experiment=SimpleNamespace(mode="hardware_aware"),
                agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=1)),
            )
            interpreter = Interpreter(working_dir=workdir, timeout=10, max_parallel_run=1, cfg=cfg)
            fake_api = _FakeSchedulerClient(settings)
            interpreter.attach_scheduler(fake_api, SimpleNamespace(wait_timeout_seconds=5, wait_poll_interval_seconds=0.01))

            result = interpreter._run_scheduler_job(
                code="MODEL_FAMILY = 'new-family'\nbatch_size = 6\nprint('new family')\n",
                id="node-new-family",
                working_dir=str(workdir),
            )

            self.assertIsNone(result.exc_type)
            self.assertEqual([job.task_type for job in fake_api.submitted_jobs], ["mlevolve_model_family_probe", "mlevolve_script"])
            probe_job, train_job = fake_api.submitted_jobs
            self.assertFalse(probe_job.packing.eligible)
            self.assertEqual(probe_job.packing.backend_allowlist, ["exclusive"])
            self.assertEqual(train_job.batch_probe.profile_key, probe_job.batch_probe.profile_key)
            self.assertTrue(train_job.batch_probe.reuse_only)
            self.assertEqual(train_job.metadata["batch_probe_source"], "model_family_profile")

    def test_scheduler_submission_blocks_training_when_model_family_probe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(runtime_root=runtime_root, gpu_scheduler={"model_family_probe_timeout_seconds": 1})
            cfg = SimpleNamespace(
                start_cpu_id=0,
                cpu_number=1,
                exp_id="unit-task",
                exp_name="unit-exp",
                experiment=SimpleNamespace(mode="hardware_aware"),
                agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=1)),
            )
            interpreter = Interpreter(working_dir=workdir, timeout=10, max_parallel_run=1, cfg=cfg)
            fake_api = _FakeSchedulerClient(settings, create_probe_profiles=False)
            interpreter.attach_scheduler(fake_api, SimpleNamespace(wait_timeout_seconds=5, wait_poll_interval_seconds=0.01))

            result = interpreter._run_scheduler_job(
                code="MODEL_FAMILY = 'blocked-family'\nbatch_size = 6\nprint('blocked')\n",
                id="node-blocked-family",
                working_dir=str(workdir),
            )

            self.assertEqual(result.exc_type, "RuntimeError")
            self.assertIn("model-family probe failed", "".join(result.term_out).lower())
            self.assertEqual([job.task_type for job in fake_api.submitted_jobs], ["mlevolve_model_family_probe"])

    def test_scheduler_bridge_starts_fallback_service_when_external_service_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(runtime_root=runtime_root)
            interpreter = Interpreter(working_dir=workdir, timeout=10, max_parallel_run=1)
            fake_api = _FakeSchedulerClient(settings, active_service=False)
            legacy_cfg = SimpleNamespace(start_service=False, wait_timeout_seconds=5, wait_poll_interval_seconds=0.01)
            interpreter.attach_scheduler(fake_api, legacy_cfg)

            result = interpreter._run_scheduler_job(
                code="print('bridge fallback service')\n",
                id="node-2",
                working_dir=str(workdir),
            )

            self.assertIsNone(result.exc_type)
            self.assertEqual(fake_api.create_service_calls, 1)
            self.assertTrue(fake_api.active_service)
            interpreter.cleanup_session(-1)
            self.assertFalse(fake_api.active_service)

    def test_scheduler_bridge_preserves_stream_allowlist_from_scheduler_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                gpu_scheduler={
                    "submission_defaults": {
                        "packing_eligible": True,
                        "backend_allowlist": ["stream"],
                    }
                },
            )
            interpreter = Interpreter(working_dir=workdir, timeout=10, max_parallel_run=1)
            fake_api = _FakeSchedulerClient(settings)
            interpreter.attach_scheduler(fake_api, SimpleNamespace(wait_timeout_seconds=5, wait_poll_interval_seconds=0.01))

            result = interpreter._run_scheduler_job(
                code="batch_size = 2\nprint('stream only raw packing config')\n",
                id="node-3",
                working_dir=str(workdir),
            )

            self.assertIsNone(result.exc_type)
            submitted = fake_api.submitted_jobs[0]
            self.assertTrue(submitted.packing.eligible)
            self.assertEqual(submitted.packing.backend_allowlist, ["stream"])

    def test_scheduler_bridge_leaves_empty_raw_allowlist_unrestricted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                gpu_scheduler={
                    "submission_defaults": {
                        "packing_eligible": True,
                        "backend_allowlist": [],
                    }
                },
            )
            interpreter = Interpreter(working_dir=workdir, timeout=10, max_parallel_run=1)
            fake_api = _FakeSchedulerClient(settings)
            interpreter.attach_scheduler(fake_api, SimpleNamespace(wait_timeout_seconds=5, wait_poll_interval_seconds=0.01))

            result = interpreter._run_scheduler_job(
                code="batch_size = 2\nprint('empty allowlist')\n",
                id="node-4",
                working_dir=str(workdir),
            )

            self.assertIsNone(result.exc_type)
            submitted = fake_api.submitted_jobs[0]
            self.assertTrue(submitted.packing.eligible)
            self.assertEqual(submitted.packing.backend_allowlist, [])
            self.assertIsNone(submitted.preload_source)

    def test_run_many_submits_round_before_collecting_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(runtime_root=runtime_root)
            cfg = SimpleNamespace(
                start_cpu_id=0,
                cpu_number=2,
                exp_id="unit-task",
                exp_name="unit-exp",
                experiment=SimpleNamespace(mode="hardware_aware"),
                agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=2)),
            )
            interpreter = Interpreter(working_dir=workdir, timeout=10, max_parallel_run=2, cfg=cfg)
            fake_api = _FakeSchedulerClient(settings)
            interpreter.attach_scheduler(fake_api, SimpleNamespace(wait_timeout_seconds=5, wait_poll_interval_seconds=0.01))

            results = interpreter.run_many(
                [
                    ("batch_size = 2\nprint('round a')\n", "node-a"),
                    ("batch_size = 4\nprint('round b')\n", "node-b"),
                ],
                working_dir=str(workdir),
            )

            self.assertEqual(set(results), {"node-a", "node-b"})
            self.assertIsNone(results["node-a"].exc_type)
            self.assertIsNone(results["node-b"].exc_type)
            self.assertEqual(fake_api.submit_many_calls, 1)
            self.assertEqual(len(fake_api.submitted_jobs), 2)
            self.assertEqual(
                {job.metadata["mlevolve_node_id"] for job in fake_api.submitted_jobs},
                {"node-a", "node-b"},
            )
            self.assertTrue(all(job.packing.eligible for job in fake_api.submitted_jobs))
            self.assertTrue(all(job.packing.signature for job in fake_api.submitted_jobs))
            self.assertEqual(getattr(fake_api, "last_tuning_outcome")["recommendation_source"], "scheduler_round")

    def test_run_many_uses_normalized_script_signature_for_raw_packing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                gpu_scheduler={"submission_defaults": {"backend_allowlist": ["cuda_process"]}},
            )
            cfg = SimpleNamespace(
                start_cpu_id=0,
                cpu_number=2,
                exp_id="unit-task",
                exp_name="unit-exp",
                experiment=SimpleNamespace(mode="baseline"),
                agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=2)),
            )
            interpreter = Interpreter(working_dir=workdir, timeout=10, max_parallel_run=2, cfg=cfg)
            fake_api = _FakeSchedulerClient(settings)
            interpreter.attach_scheduler(fake_api, SimpleNamespace(wait_timeout_seconds=5, wait_poll_interval_seconds=0.01))

            results = interpreter.run_many(
                [
                    ("batch_size = 2\nprint('same workload')\n", "node-a"),
                    ("batch_size = 4\nprint('same workload')\n", "node-b"),
                ],
                working_dir=str(workdir),
            )

            self.assertEqual(set(results), {"node-a", "node-b"})
            self.assertEqual(fake_api.submit_many_calls, 1)
            submitted = fake_api.submitted_jobs
            self.assertEqual(len(submitted), 2)
            self.assertTrue(all(job.packing.eligible for job in submitted))
            self.assertEqual(submitted[0].packing.signature, submitted[1].packing.signature)
            self.assertEqual(submitted[0].packing.backend_allowlist, ["cuda_process"])
            self.assertEqual(submitted[1].packing.backend_allowlist, ["cuda_process"])

    def test_run_many_scheduler_packet_is_not_capped_by_local_parallelism(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(runtime_root=runtime_root)
            cfg = SimpleNamespace(
                start_cpu_id=0,
                cpu_number=1,
                exp_id="unit-task",
                exp_name="unit-exp",
                experiment=SimpleNamespace(mode="baseline"),
                agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=1)),
            )
            interpreter = Interpreter(working_dir=workdir, timeout=None, max_parallel_run=1, cfg=cfg)
            fake_api = _FakeSchedulerClient(settings)
            interpreter.attach_scheduler(fake_api, SimpleNamespace(wait_timeout_seconds=None, wait_poll_interval_seconds=0.01))

            results = interpreter.run_many(
                [
                    ("print('a')\n", "node-a"),
                    ("print('b')\n", "node-b"),
                    ("print('c')\n", "node-c"),
                ],
                working_dir=str(workdir),
            )

            self.assertEqual(set(results), {"node-a", "node-b", "node-c"})
            self.assertEqual(len(fake_api.submitted_jobs), 3)
            self.assertTrue(all("timeout" not in job.config.runner_kwargs for job in fake_api.submitted_jobs))


if __name__ == "__main__":
    unittest.main()
