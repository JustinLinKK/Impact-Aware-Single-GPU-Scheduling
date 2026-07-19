from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from localml_scheduler.adapters.mlevolve_runner import (
    _materialize_instrumented_script,
    classify_mlevolve_probe_failure,
    probe_mlevolve_script_job,
    run_mlevolve_script_job,
)
from localml_scheduler.checkpointing.manager import CheckpointManager
from localml_scheduler.execution.control import ControlPlane, TrainingControlHook
from localml_scheduler.execution.runner_protocol import RunnerContext
from localml_scheduler.execution.worker_runtime import mark_job_completed
from localml_scheduler.observability.events import EventLogger
from localml_scheduler.domain import BatchProbeSpec, CheckpointPolicy, JobStatus, SafePointType, TrainingJob
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def _build_context(settings: SchedulerSettings, job: TrainingJob) -> RunnerContext:
    store = SQLiteStateStore(settings)
    store.save_job(job)
    event_logger = EventLogger(store, settings.events_jsonl_path)
    checkpoint_manager = CheckpointManager(settings, store, event_logger)
    control_plane = ControlPlane(settings)
    control_plane.initialize_job(job.job_id)
    control_hook = TrainingControlHook(job, control_plane, checkpoint_manager, store, event_logger)
    return RunnerContext(
        job=job,
        settings=settings,
        store=store,
        event_logger=event_logger,
        control_hook=control_hook,
        checkpoint_manager=checkpoint_manager,
        cache_client=None,
    )


class MLEvolveRunnerTest(unittest.TestCase):
    def test_candidate_failure_marks_scheduler_job_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = SchedulerSettings(runtime_root=Path(tmpdir) / "runtime")
            job = TrainingJob.create("pkg.runner:run", "model", "/tmp/model")
            context = _build_context(settings, job)

            code = mark_job_completed(
                settings,
                context.store,
                context.event_logger,
                job.job_id,
                {"success": False, "outcome": "candidate_exception", "candidate_exc_type": "RuntimeError"},
            )

            self.assertEqual(code, 0)
            self.assertEqual(context.store.get_job(job.job_id).status, JobStatus.FAILED)
            self.assertTrue(any(event["event_type"] == "job_candidate_failed" for event in context.store.list_events(job_id=job.job_id)))

    def test_probe_failure_classification_uses_terminal_exception(self) -> None:
        kind, message = classify_mlevolve_probe_failure(
            stderr_text=(
                "TensorFlow binary supports AVX512_BF16 instructions.\n"
                "Traceback (most recent call last):\n"
                "RuntimeError: fallback loop never executed\n"
            ),
            returncode=1,
        )

        self.assertEqual(kind, "script_exception")
        self.assertEqual(message, "fallback loop never executed")

    def test_materialized_script_preserves_future_import_preamble(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                '\"\"\"candidate module\"\"\"\nfrom __future__ import annotations\n'
                'from torch.utils.data import DataLoader\nbatch_size = 4\n'
                'train_loader = DataLoader(range(8), batch_size=batch_size, shuffle=True)\n',
                encoding="utf-8",
            )

            instrumented = _materialize_instrumented_script(script_path, working_dir)
            materialized = instrumented.path.read_text(encoding="utf-8")

            compile(materialized, str(instrumented.path), "exec")
            self.assertLess(materialized.index("from __future__ import annotations"), materialized.index("import os"))

    def test_materialized_script_repairs_low_precision_numpy_export_with_batch_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import torch",
                        "batch_size = 4",
                        "from torch.utils.data import DataLoader",
                        "train_loader = DataLoader(range(8), batch_size=batch_size, shuffle=True)",
                        "AMP_DTYPE = torch.bfloat16",
                        "preds_np = preds.cpu().numpy().flatten()",
                        "labels_np = labels.cpu().numpy()",
                    ]
                ),
                encoding="utf-8",
            )

            instrumented = _materialize_instrumented_script(script_path, working_dir)
            materialized = instrumented.path.read_text(encoding="utf-8")

            self.assertTrue(instrumented.had_batch_rewrite)
            self.assertEqual(instrumented.precision_repair_count, 1)
            self.assertIn("preds.detach().to(torch.float32).cpu().numpy().flatten()", materialized)
            self.assertIn("labels.cpu().numpy()", materialized)

    def test_materialized_script_repairs_low_precision_numpy_export_without_batch_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import transformer_engine.pytorch as te",
                        "PRECISION = 'nvfp4'",
                        "logits_np = logits.cpu().numpy()",
                    ]
                ),
                encoding="utf-8",
            )

            instrumented = _materialize_instrumented_script(script_path, working_dir)
            materialized = instrumented.path.read_text(encoding="utf-8")

            self.assertFalse(instrumented.had_batch_rewrite)
            self.assertEqual(instrumented.precision_repair_count, 1)
            self.assertNotEqual(instrumented.path, script_path)
            self.assertIn("logits.detach().to(torch.float32).cpu().numpy()", materialized)

    def test_run_script_job_honors_resolved_batch_size_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import json",
                        "from pathlib import Path",
                        "from torch.utils.data import DataLoader",
                        "batch_size = 4",
                        "train_loader = DataLoader(range(9), batch_size=batch_size, shuffle=True)",
                        "test_loader = DataLoader(range(8), batch_size=batch_size)",
                        "trace = {'train': [len(x) for x in train_loader], 'test': [len(x) for x in test_loader]}",
                        "Path('batch_sizes.json').write_text(json.dumps(trace), encoding='utf-8')",
                    ]
                ),
                encoding="utf-8",
            )
            result_path = working_dir / "result.json"
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(result_path),
                    "timeout": 30,
                },
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
                ),
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, pause_mode=SafePointType.STEP),
                metadata={"resolved_batch_size": 9, "placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            result = run_mlevolve_script_job(context)

            self.assertEqual(result["candidate_returncode"], 0)
            self.assertEqual(result["batch_size_override"], 9)
            trace = json.loads((working_dir / "batch_sizes.json").read_text(encoding="utf-8"))
            self.assertEqual(trace["train"], [9])
            self.assertEqual(trace["test"], [4, 4])

    def test_run_script_job_records_phase_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import time",
                        "def train_model():",
                        "    for epoch in range(2):",
                        "        time.sleep(0.01)",
                        "def generate_submission():",
                        "    time.sleep(0.01)",
                        "train_model()",
                        "generate_submission()",
                    ]
                ),
                encoding="utf-8",
            )
            result_path = working_dir / "result.json"
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(result_path),
                    "timeout": 30,
                },
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, pause_mode=SafePointType.STEP),
                metadata={"placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            result = run_mlevolve_script_job(context)

            self.assertEqual(result["candidate_returncode"], 0)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["instrumentation"]["phase_instrumented"])
            durations = payload["phase_timings"]["phase_durations_seconds"]
            self.assertGreater(durations["training"], 0)
            self.assertGreater(durations["inference"], 0)

    def test_probe_script_job_runs_successfully_with_batch_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import torch",
                        "from torch.utils.data import DataLoader, TensorDataset",
                        "batch_size = 2",
                        "x = torch.randn(8, 2)",
                        "y = torch.randn(8, 1)",
                        "train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)",
                        "model = torch.nn.Linear(2, 1)",
                        "optimizer = torch.optim.AdamW(model.parameters())",
                        "for xb, yb in train_loader:",
                        "    loss = ((model(xb) - yb) ** 2).mean()",
                        "    loss.backward()",
                        "    optimizer.step()",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(working_dir / "result.json"),
                    "timeout": 30,
                    "probe_timeout_seconds": 5,
                    "probe_poll_interval_seconds": 0.2,
                },
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
                ),
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, pause_mode=SafePointType.STEP),
                metadata={"placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            result = probe_mlevolve_script_job(context, batch_size=5, warmup_steps=1, measure_steps=1)

            self.assertTrue(result.fits)
            self.assertTrue(result.probe_completed)
            self.assertIn("optimizer step", result.message or "")

    def test_probe_script_job_classifies_timeout_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import torch",
                        "import time",
                        "from torch.utils.data import DataLoader, TensorDataset",
                        "batch_size = 2",
                        "x = torch.randn(8, 2)",
                        "y = torch.randn(8, 1)",
                        "train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)",
                        "time.sleep(3)",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(working_dir / "result.json"),
                    "timeout": 30,
                    "probe_timeout_seconds": 1,
                    "probe_poll_interval_seconds": 0.2,
                },
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
                ),
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, pause_mode=SafePointType.STEP),
                metadata={"placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            result = probe_mlevolve_script_job(context, batch_size=5, warmup_steps=1, measure_steps=1)

            self.assertFalse(result.fits)
            self.assertEqual(result.failure_kind, "timeout")
            self.assertIn("timed out", result.message or "")
            self.assertEqual(result.diagnostic.phase, "startup")

    def test_probe_script_job_requires_optimizer_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "from torch.utils.data import DataLoader\n"
                "batch_size = 2\n"
                "train_loader = DataLoader(range(8), batch_size=batch_size, shuffle=True)\n"
                "for batch in train_loader:\n"
                "    pass\n",
                encoding="utf-8",
            )
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(working_dir / "result.json"),
                    "probe_timeout_seconds": 5,
                },
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
                ),
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, pause_mode=SafePointType.STEP),
                metadata={"placement_backend": "exclusive"},
            )

            result = probe_mlevolve_script_job(_build_context(settings, job), 4, 1, 1)

            self.assertFalse(result.fits)
            self.assertEqual(result.failure_kind, "probe_incomplete")
            self.assertFalse(result.probe_completed)

    def test_probe_script_job_exits_after_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import torch",
                        "from pathlib import Path",
                        "from torch.utils.data import DataLoader, TensorDataset",
                        "batch_size = 2",
                        "num_epochs = 5",
                        "x = torch.randn(20, 2)",
                        "y = torch.randn(20, 1)",
                        "train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)",
                        "model = torch.nn.Linear(2, 1)",
                        "optimizer = torch.optim.AdamW(model.parameters())",
                        "for epoch in range(num_epochs):",
                        "    for xb, yb in train_loader:",
                        "        Path('train_batch_size.txt').write_text(str(len(xb)), encoding='utf-8')",
                        "        loss = ((model(xb) - yb) ** 2).mean()",
                        "        loss.backward()",
                        "        optimizer.step()",
                        "        Path('after_step.txt').write_text('unexpected', encoding='utf-8')",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(working_dir / "result.json"),
                    "timeout": 30,
                    "probe_timeout_seconds": 5,
                    "probe_poll_interval_seconds": 0.2,
                    "probe_max_epochs": 1,
                    "probe_max_train_batches": 3,
                },
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
                ),
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, pause_mode=SafePointType.STEP),
                metadata={"placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            result = probe_mlevolve_script_job(context, batch_size=7, warmup_steps=1, measure_steps=1)

            self.assertTrue(result.fits)
            self.assertEqual((working_dir / "train_batch_size.txt").read_text(encoding="utf-8"), "7")
            self.assertFalse((working_dir / "after_step.txt").exists())

    def test_probe_script_job_never_enters_test_submission_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import torch",
                        "from pathlib import Path",
                        "from torch.utils.data import DataLoader, TensorDataset",
                        "batch_size = 2",
                        "x = torch.randn(12, 2)",
                        "y = torch.randn(12, 1)",
                        "train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)",
                        "test_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)",
                        "model = torch.nn.Linear(2, 1)",
                        "optimizer = torch.optim.AdamW(model.parameters())",
                        "for xb, yb in train_loader:",
                        "    loss = ((model(xb) - yb) ** 2).mean()",
                        "    loss.backward()",
                        "    optimizer.step()",
                        "Path('test_started.txt').write_text('unexpected', encoding='utf-8')",
                        "for batch in test_loader:",
                        "    pass",
                    ]
                ),
                encoding="utf-8",
            )
            settings = SchedulerSettings(runtime_root=runtime_root)
            job = TrainingJob.create(
                runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
                baseline_model_id="script",
                baseline_model_path=str(script_path),
                runner_kwargs={
                    "script_path": str(script_path),
                    "working_dir": str(working_dir),
                    "result_path": str(working_dir / "result.json"),
                    "timeout": 30,
                    "probe_timeout_seconds": 5,
                    "probe_poll_interval_seconds": 0.2,
                    "probe_max_epochs": 1,
                    "probe_max_train_batches": 2,
                },
                batch_probe=BatchProbeSpec(
                    enabled=True,
                    probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
                ),
                checkpoint_policy=CheckpointPolicy(save_every_n_steps=1, pause_mode=SafePointType.STEP),
                metadata={"placement_backend": "exclusive"},
            )
            context = _build_context(settings, job)

            result = probe_mlevolve_script_job(context, batch_size=3, warmup_steps=1, measure_steps=1)

            self.assertTrue(result.fits)
            self.assertFalse((working_dir / "test_started.txt").exists())


if __name__ == "__main__":
    unittest.main()
