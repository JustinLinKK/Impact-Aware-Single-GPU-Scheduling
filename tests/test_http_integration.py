from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event, Thread
import unittest
from unittest.mock import Mock, patch

from localml_scheduler import SchedulerHTTPClient
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SchedulerConfig
from localml_scheduler.domain import JobStatus, ResourceRequirements, TrainingJob
from localml_scheduler.http_server import SchedulerHTTPServer
from localml_scheduler.scheduler.supervisor import WorkerSnapshot


class SchedulerHTTPIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.settings = SchedulerConfig(
            runtime_root=self.temporary.name,
            scheduler_poll_interval_seconds=0.05,
            graph_db={"enabled": False},
            hardware_feature_db={"enabled": False},
            baseline_cache={"warm_queue_top_k": 0},
        )
        self.local = SchedulerClient(self.settings)
        self.service = self.local.create_service(telemetry_sampler=Mock(sample=Mock(return_value=None)))
        self.server = SchedulerHTTPServer(self.service, token="test-token", port=0)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.client = SchedulerHTTPClient(self.url, token="test-token")

    def tearDown(self) -> None:
        self.client.close()
        self.service.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.temporary.cleanup()

    def job(self, job_id: str = "http-job") -> TrainingJob:
        return TrainingJob.create(
            "localml_scheduler.examples.toy_pytorch_runner:run_toy_training_job",
            "tiny",
            str(Path(self.temporary.name) / "baseline.pt"),
            job_id=job_id,
            max_steps=2,
            resource_requirements=ResourceRequirements(requires_gpu=False),
            metadata={"perfseer_model": {"entry": "build_model", "input_shapes": [["$batch", 16]]}},
        )

    def test_submit_inspect_commands_and_connection_reuse(self) -> None:
        seed = self.job()
        submitted = self.client.submit(seed)
        connection = self.client._connection.sock
        self.assertEqual(submitted.metadata["perfseer_model"], seed.metadata["perfseer_model"])
        self.assertEqual(self.client.inspect(seed.job_id).job_id, seed.job_id)
        self.assertEqual([job.job_id for job in self.client.list_jobs()], [seed.job_id])
        self.assertIsNone(self.client.inspect("unknown"))
        for command in (self.client.pause, self.client.resume, self.client.cancel):
            self.assertGreater(command(seed.job_id), 0)
        self.assertIs(self.client._connection.sock, connection)
        self.assertEqual(
            [command.command_type.value for command in self.local.store.fetch_pending_commands()],
            ["SUBMIT", "PAUSE", "RESUME", "CANCEL"],
        )
        with self.assertRaisesRegex(RuntimeError, "409"):
            self.client.submit(seed)
        self.assertEqual(len(self.local.list_jobs()), 1)

    def test_rejects_unauthorized_and_invalid_requests(self) -> None:
        with SchedulerHTTPClient(self.url, token="wrong") as client:
            with self.assertRaisesRegex(RuntimeError, "401"):
                client.submit(self.job())
        self.assertEqual(self.local.list_jobs(), [])
        for body in ("[]", "{", '{}'):
            with self.subTest(body=body):
                connection = HTTPConnection("127.0.0.1", self.server.server_port)
                try:
                    connection.request("POST", "/jobs", body, {
                        "Authorization": "Bearer test-token",
                        "Content-Type": "application/json",
                    })
                    response = connection.getresponse()
                    self.assertEqual(response.status, 400)
                    self.assertIn("error", json.loads(response.read()))
                finally:
                    connection.close()
        self.client.submit(self.job())
        with self.assertRaisesRegex(RuntimeError, "400"):
            self.client._request("POST", "/jobs/http-job/commands", {"command": "PRELOAD"})
        with self.assertRaisesRegex(RuntimeError, "400"):
            self.client._request("GET", "/jobs/http-job/completion?timeout=nan")

    def test_completion_timeout_and_queued_cancellation_notification(self) -> None:
        job = self.client.submit(self.job())
        self.assertIsNone(self.client.wait_for_completion(job.job_id, timeout=0.01))
        waiting = Event()
        original_wait = self.service._completion_condition.wait

        def wait(timeout):
            waiting.set()
            return original_wait(timeout)

        with patch.object(self.service._completion_condition, "wait", side_effect=wait):
            with ThreadPoolExecutor() as executor:
                result = executor.submit(self.client.wait_for_completion, job.job_id, timeout=2.0)
                self.assertTrue(waiting.wait(1.0))
                self.service._handle_cancel(job.job_id)
                self.assertEqual(result.result(timeout=1.0).status, JobStatus.CANCELLED)
        self.assertEqual(self.client.wait_for_completion(job.job_id, timeout=0).status, JobStatus.CANCELLED)

    def test_worker_completion_and_persisted_outcomes(self) -> None:
        job = self.client.submit(self.job())
        waiting = Event()
        original_wait = self.service._completion_condition.wait

        def wait(timeout):
            waiting.set()
            return original_wait(timeout)

        with patch.object(self.service._completion_condition, "wait", side_effect=wait):
            with ThreadPoolExecutor() as executor:
                result = executor.submit(self.client.wait_for_completion, job.job_id, timeout=2.0)
                self.assertTrue(waiting.wait(1.0))
                worker_store = SchedulerClient(self.settings).store
                worker_store.set_job_status(job.job_id, JobStatus.COMPLETED)
                with patch.object(self.service.supervisor, "poll", return_value=[
                    WorkerSnapshot(job.job_id, "test-group", alive=False, returncode=0),
                ]):
                    self.service._poll_active_workers()
                self.assertEqual(result.result(timeout=1.0).status, JobStatus.COMPLETED)
        restarted = SchedulerClient(self.settings).create_service(telemetry_sampler=Mock())
        try:
            self.assertEqual(restarted.wait_for_completion(job.job_id, timeout=0).status, JobStatus.COMPLETED)
            worker_store.set_job_status(job.job_id, JobStatus.FAILED)
            self.assertEqual(self.client.wait_for_completion(job.job_id, timeout=0).status, JobStatus.FAILED)
        finally:
            restarted.stop()

    def test_stop_releases_completion_waiters(self) -> None:
        job = self.client.submit(self.job())
        waiting = Event()
        original_wait = self.service._completion_condition.wait

        def wait(timeout):
            waiting.set()
            return original_wait(timeout)

        with patch.object(self.service._completion_condition, "wait", side_effect=wait):
            with ThreadPoolExecutor() as executor:
                result = executor.submit(self.client.wait_for_completion, job.job_id, timeout=2.0)
                self.assertTrue(waiting.wait(1.0))
                self.service.stop()
                with self.assertRaisesRegex(RuntimeError, "503"):
                    result.result(timeout=1.0)

    def test_cpu_worker_completes_through_http(self) -> None:
        from localml_scheduler.examples.toy_pytorch_runner import create_toy_baseline_checkpoint

        job = self.job()
        create_toy_baseline_checkpoint(job.baseline_model_path)
        self.service.start(background=True)
        submitted = self.client.submit(job)
        completed = self.client.wait_for_completion(submitted.job_id, timeout=20.0)
        self.assertIsNotNone(completed)
        self.assertEqual(completed.status, JobStatus.COMPLETED, completed.status_reason)


def test_http_sdk_does_not_load_torch_or_create_runtime(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", (
            f"import sys; sys.path.insert(0, {str(root)!r}); "
            "from localml_scheduler import SchedulerHTTPClient; "
            "assert 'torch' not in sys.modules; "
            "assert 'localml_scheduler.scheduler.service' not in sys.modules"
        )],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
