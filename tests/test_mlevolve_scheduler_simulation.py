from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from engine.executor import Interpreter
from localml_scheduler.client import SchedulerClient
from localml_scheduler.config import SCHEDULER_MODE_ADAPTIVE, SchedulerSettings
from localml_scheduler.domain import JobStatus, ProfileState


SIMULATED_TRAINING_EPOCHS = 5
SIMULATED_STEPS_PER_EPOCH = 2


class _RecordingSchedulerClient(SchedulerClient):
    def __init__(self, settings: SchedulerSettings):
        super().__init__(settings)
        self.tuning_outcomes: list[dict[str, object]] = []

    def record_tuning_outcome(self, **kwargs):
        self.tuning_outcomes.append(dict(kwargs))
        return {"ok": True}


def _scheduler_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        wait_timeout_seconds=45,
        wait_poll_interval_seconds=0.1,
        start_service=True,
    )


def _interpreter_cfg(workdir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        start_cpu_id=0,
        cpu_number=1,
        exp_id="simulated-agent-task",
        exp_name="simulated-agent-workflow",
        workspace_dir=workdir,
        experiment=SimpleNamespace(mode="hardware_aware"),
        scheduler=SimpleNamespace(enabled=True),
        agent=SimpleNamespace(search=SimpleNamespace(parallel_search_num=3)),
    )


def _agent_candidate_code(label: str, *, batch_size: int, model_family: str) -> str:
    return "\n".join(
        [
            "import json",
            "import torch",
            "from torch.utils.data import TensorDataset",
            "from localml_scheduler.elastic import ElasticTrainingSession",
            f"MODEL_FAMILY = {model_family!r}",
            f"batch_size = {batch_size}",
            f"epochs = {SIMULATED_TRAINING_EPOCHS}",
            "session = ElasticTrainingSession.from_env()",
            "features = torch.arange(2048, dtype=torch.float32).reshape(1024, 2)",
            "targets = features.sum(dim=1, keepdim=True)",
            "train_loader = session.make_dataloader(TensorDataset(features, targets), shuffle=True)",
            "model = torch.nn.Linear(2, 1)",
            "optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)",
            "session.register_training_state(model, optimizer, extra_state={'label': MODEL_FAMILY})",
            "progress = session.restore_if_present()",
            "global_step = progress['global_step']",
            "for epoch in range(progress['epoch'], epochs):",
            "    for step, (inputs, labels) in enumerate(train_loader):",
            f"        if step >= {SIMULATED_STEPS_PER_EPOCH}:",
            "            break",
            "        optimizer.zero_grad()",
            "        loss = ((model(inputs) - labels) ** 2).mean()",
            "        loss.backward()",
            "        optimizer.step()",
            "        global_step += 1",
            "        session.optimizer_step_completed(len(inputs), epoch, step, global_step, metrics={'loss': float(loss.item())})",
            "        print(",
            "            'MLEVOLVE_METRIC: ' + json.dumps({",
            "                'loss': 1.0 / (step + 1),",
            "                'accuracy': 0.70 + step * 0.01,",
            "                'epoch': epoch,",
            "                'global_step': global_step,",
            "            }),",
            "            flush=True,",
            "        )",
            f"print('agent candidate {label} completed batch_size=' + str(session.batch_size), flush=True)",
        ]
    )


class MLEvolveSchedulerSimulationTest(unittest.TestCase):
    def test_real_scheduler_executes_simulated_agent_round_with_batch_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_root = root / "runtime"
            workdir = root / "workspace"
            workdir.mkdir(parents=True, exist_ok=True)
            settings = SchedulerSettings(
                runtime_root=runtime_root,
                scheduler_poll_interval_seconds=0.05,
                gpu_scheduler={
                    "mode": SCHEDULER_MODE_ADAPTIVE,
                    "backend_priority": ["exclusive"],
                    "adaptive": {"replan_debounce_seconds": 0.0},
                    "batch_probe_max_search_rounds": 4,
                    "profiling": {"warmup_steps": 1, "solo_probe_steps": 1},
                    "submission_defaults": {
                        "requires_gpu": True,
                        "packing_eligible": True,
                        "backend_allowlist": ["exclusive"],
                        "batch_probe_enabled": True,
                        "batch_probe_probe_timeout_seconds": 5,
                        "batch_probe_poll_interval_seconds": 0.05,
                        "batch_probe_max_multiplier": 2,
                        "runtime_probe_enabled": True,
                    },
                },
            )
            api = _RecordingSchedulerClient(settings)
            service = api.create_service().start(background=True)
            interpreter = Interpreter(
                working_dir=workdir,
                timeout=10,
                max_parallel_run=3,
                cfg=_interpreter_cfg(workdir),
            )
            interpreter.attach_scheduler(api, _scheduler_cfg())

            try:
                candidates = [
                    {
                        "id": "node-alpha",
                        "code": _agent_candidate_code("alpha", batch_size=2, model_family="sim-agent-alpha"),
                        "branch_id": "branch-a",
                    },
                    {
                        "id": "node-beta",
                        "code": _agent_candidate_code("beta", batch_size=4, model_family="sim-agent-beta"),
                        "branch_id": "branch-b",
                    },
                    {
                        "id": "node-gamma",
                        "code": _agent_candidate_code("gamma", batch_size=8, model_family="sim-agent-gamma"),
                        "branch_id": "branch-c",
                    },
                ]

                handles = interpreter.submit_scheduler(candidates, working_dir=str(workdir), submission_label="round")
                self.assertEqual(set(handles), {"node-alpha", "node-beta", "node-gamma"})
                job_ids_by_node = {node_id: handle.job_id for node_id, handle in handles.items()}
                self.assertTrue(all(job_ids_by_node.values()))

                results = interpreter.collect_scheduler(handles, wait=True, timeout=45)

                self.assertEqual(set(results), set(handles))
                for node_id, result in results.items():
                    self.assertIsNone(result.exc_type, f"{node_id}: {result.term_out}")
                    self.assertTrue(any(f"agent candidate {node_id.split('-')[1]} completed" in line for line in result.term_out))
                    self.assertIsNotNone(result.phase_timings)

                for node_id, job_id in job_ids_by_node.items():
                    final_job = api.inspect(str(job_id))
                    self.assertIsNotNone(final_job)
                    assert final_job is not None
                    self.assertEqual(final_job.status, JobStatus.COMPLETED)
                    self.assertEqual(final_job.workflow_id, "simulated-agent-workflow")
                    self.assertEqual(final_job.task_type, "mlevolve_script")
                    self.assertEqual(final_job.metadata["node_id"], node_id)
                    self.assertEqual(final_job.metadata["proposed_epochs"], SIMULATED_TRAINING_EPOCHS)
                    self.assertEqual(final_job.metadata["placement_backend"], "exclusive")
                    self.assertEqual(final_job.metadata["placement_mode"], "exclusive")
                    self.assertEqual(final_job.profile_state, ProfileState.READY)
                    self.assertEqual(final_job.authored_batch_size, int(final_job.metadata["detected_batch_size"]))
                    self.assertIn("scheduler_session_id", final_job.metadata)
                    curves = [
                        curve
                        for curve in api.store.list_batch_profile_curves()
                        if curve.profile_namespace == final_job.batch_probe.profile_namespace
                    ]
                    self.assertEqual(len(curves), 1)
                    self.assertEqual(curves[0].contract_version, 3)
                    self.assertTrue(curves[0].points)
                    self.assertGreaterEqual(
                        len(api.list_job_metric_samples(final_job.job_id)),
                        SIMULATED_TRAINING_EPOCHS * SIMULATED_STEPS_PER_EPOCH,
                    )

                report = api.report()
                self.assertEqual(report["total_jobs"], 6)
                self.assertEqual(report["completed_jobs"], 6)
                self.assertEqual(report["failed_jobs"], 0)
                self.assertEqual(len(api.tuning_outcomes), 3)
                self.assertGreaterEqual(len(api.list_events(event_type="branch_profile_probe_queued")), 3)
                self.assertGreaterEqual(len(api.list_events(event_type="batch_probe_started")), 3)
                self.assertGreaterEqual(len(api.list_events(event_type="job_dispatched")), 3)
            finally:
                interpreter.terminate_all_subprocesses()
                service.stop()
