from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from localml_scheduler.adapters.mlevolve_runner import (
    _materialize_instrumented_script,
    probe_mlevolve_script_job,
    run_mlevolve_script_job,
)
from localml_scheduler.checkpointing.manager import CheckpointManager
from localml_scheduler.execution.control import ControlPlane, TrainingControlHook
from localml_scheduler.execution.runner_protocol import RunnerContext
from localml_scheduler.observability.events import EventLogger
from localml_scheduler.domain import BatchProbeSpec, CheckpointPolicy, SafePointType, TrainingJob
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
    def test_materialized_script_preserves_future_import_preamble(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                '\"\"\"candidate module\"\"\"\nfrom __future__ import annotations\nbatch_size = 4\n',
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
                        "from pathlib import Path",
                        "batch_size = 4",
                        "Path('batch_size.txt').write_text(str(batch_size), encoding='utf-8')",
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
            self.assertEqual((working_dir / "batch_size.txt").read_text(encoding="utf-8"), "9")

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
                        "import time",
                        "batch_size = 2",
                        "time.sleep(1)",
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
            self.assertIn("probe", result.message or "")

    def test_probe_script_job_classifies_timeout_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            working_dir = Path(tmpdir) / "workspace"
            working_dir.mkdir(parents=True, exist_ok=True)
            script_path = working_dir / "candidate.py"
            script_path.write_text(
                "\n".join(
                    [
                        "import time",
                        "batch_size = 2",
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

    def test_probe_script_job_limits_epochs_and_batches(self) -> None:
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
                        "batch_size = 2",
                        "num_epochs = 5",
                        "trace = []",
                        "loader = DataLoader(list(range(20)), batch_size=batch_size)",
                        "for epoch in range(num_epochs):",
                        "    for step, batch in enumerate(loader):",
                        "        trace.append({'epoch': epoch, 'step': step, 'size': int(len(batch))})",
                        "Path('probe_trace.json').write_text(json.dumps(trace), encoding='utf-8')",
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
            trace = json.loads((working_dir / "probe_trace.json").read_text(encoding="utf-8"))
            self.assertEqual(len(trace), 3)
            self.assertEqual({entry["epoch"] for entry in trace}, {0})
            self.assertEqual([entry["step"] for entry in trace], [0, 1, 2])
            self.assertEqual([entry["size"] for entry in trace], [7, 7, 6])

    def test_probe_script_job_does_not_limit_test_submission_loader(self) -> None:
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
                        "from torch.utils.data import DataLoader, Dataset",
                        "class DemoDataset(Dataset):",
                        "    def __init__(self, size, is_test=False):",
                        "        self.size = size",
                        "        self.is_test = is_test",
                        "    def __len__(self):",
                        "        return self.size",
                        "    def __getitem__(self, idx):",
                        "        return idx",
                        "batch_size = 2",
                        "train_loader = DataLoader(DemoDataset(20), batch_size=batch_size)",
                        "test_loader = DataLoader(DemoDataset(12, is_test=True), batch_size=batch_size)",
                        "trace = {'train': [], 'test': []}",
                        "for batch in train_loader:",
                        "    trace['train'].append(int(len(batch)))",
                        "for batch in test_loader:",
                        "    trace['test'].append(int(len(batch)))",
                        "Path('probe_trace.json').write_text(json.dumps(trace), encoding='utf-8')",
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
            trace = json.loads((working_dir / "probe_trace.json").read_text(encoding="utf-8"))
            self.assertEqual(trace["train"], [3, 3])
            self.assertEqual(trace["test"], [3, 3, 3, 3])


if __name__ == "__main__":
    unittest.main()
