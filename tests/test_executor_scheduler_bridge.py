from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

from engine.executor import Interpreter
from localml_scheduler.schemas import JobStatus, TrainingJob
from localml_scheduler.settings import SchedulerSettings


class _FakeStore:
    def list_events(self, *, job_id: str | None = None) -> list[dict[str, object]]:
        return []


class _FakeSchedulerAPI:
    def __init__(self, settings: SchedulerSettings, *, active_service: bool = True):
        self.settings = settings
        self.store = _FakeStore()
        self.submitted_jobs: list[TrainingJob] = []
        self._jobs: dict[str, TrainingJob] = {}
        self.active_service = active_service
        self.create_scheduler_service_calls = 0

    def submit_job(self, job: TrainingJob) -> TrainingJob:
        self.submitted_jobs.append(job)
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

    def get_job(self, job_id: str) -> TrainingJob | None:
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.mark_status(JobStatus.CANCELLED, reason="cancelled")

    def scheduler_service_active(self, *, max_staleness_seconds: float | None = None) -> bool:
        return self.active_service

    def create_scheduler_service(self):
        self.create_scheduler_service_calls += 1

        api = self

        class _FakeService:
            def start(self, *, background: bool = False):
                api.active_service = True
                return self

            def stop(self):
                api.active_service = False

        return _FakeService()


class InterpreterSchedulerBridgeTest(unittest.TestCase):
    def test_scheduler_submission_defaults_override_legacy_bridge_fields(self) -> None:
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
            fake_api = _FakeSchedulerAPI(settings)
            legacy_cfg = SimpleNamespace(
                wait_timeout_seconds=5,
                wait_poll_interval_seconds=0.01,
                requires_gpu=False,
                estimated_vram_mb=1024,
                estimated_ram_mb=512,
                packing_eligible=True,
                packing_family="legacy-family",
                packing_max_slowdown_ratio=1.5,
                batch_probe_enabled=False,
                batch_probe_model_key="legacy-model-key",
                batch_probe_probe_timeout_seconds=3,
                batch_probe_poll_interval_seconds=1.0,
                batch_probe_max_multiplier=2,
                batch_probe_search_mode="binary",
            )
            interpreter.attach_scheduler(fake_api, legacy_cfg)

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
            self.assertEqual(submitted.config.runner_kwargs["probe_max_batch_size"], 12)
            self.assertTrue(interpreter._scheduler_legacy_warnings)

    def test_scheduler_bridge_starts_fallback_service_when_external_service_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(runtime_root=runtime_root)
            interpreter = Interpreter(working_dir=workdir, timeout=10, max_parallel_run=1)
            fake_api = _FakeSchedulerAPI(settings, active_service=False)
            legacy_cfg = SimpleNamespace(start_service=False, wait_timeout_seconds=5, wait_poll_interval_seconds=0.01)
            interpreter.attach_scheduler(fake_api, legacy_cfg)

            result = interpreter._run_scheduler_job(
                code="print('bridge fallback service')\n",
                id="node-2",
                working_dir=str(workdir),
            )

            self.assertIsNone(result.exc_type)
            self.assertEqual(fake_api.create_scheduler_service_calls, 1)
            self.assertTrue(fake_api.active_service)
            interpreter.cleanup_session(-1)
            self.assertFalse(fake_api.active_service)

    def test_scheduler_bridge_disables_raw_packing_for_unsupported_stream_only_allowlist(self) -> None:
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
            fake_api = _FakeSchedulerAPI(settings)
            interpreter.attach_scheduler(fake_api, SimpleNamespace(wait_timeout_seconds=5, wait_poll_interval_seconds=0.01))

            result = interpreter._run_scheduler_job(
                code="batch_size = 2\nprint('stream only raw packing config')\n",
                id="node-3",
                working_dir=str(workdir),
            )

            self.assertIsNone(result.exc_type)
            submitted = fake_api.submitted_jobs[0]
            self.assertFalse(submitted.packing.eligible)
            self.assertEqual(submitted.packing.backend_allowlist, [])

    def test_scheduler_bridge_uses_process_backends_when_raw_allowlist_is_empty(self) -> None:
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
            fake_api = _FakeSchedulerAPI(settings)
            interpreter.attach_scheduler(fake_api, SimpleNamespace(wait_timeout_seconds=5, wait_poll_interval_seconds=0.01))

            result = interpreter._run_scheduler_job(
                code="batch_size = 2\nprint('empty allowlist')\n",
                id="node-4",
                working_dir=str(workdir),
            )

            self.assertIsNone(result.exc_type)
            submitted = fake_api.submitted_jobs[0]
            self.assertTrue(submitted.packing.eligible)
            self.assertEqual(submitted.packing.backend_allowlist, ["mps", "cuda_process"])


if __name__ == "__main__":
    unittest.main()
