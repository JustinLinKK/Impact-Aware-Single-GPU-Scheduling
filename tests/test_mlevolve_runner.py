from __future__ import annotations

import json
from pathlib import Path

import pytest

from localml_scheduler.adapters.mlevolve_runner import (
    classify_mlevolve_probe_failure,
    probe_mlevolve_script_job,
    run_mlevolve_script_job,
)
from localml_scheduler.checkpointing.manager import CheckpointManager
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import BatchProbeSpec, CheckpointPolicy, JobStatus, TrainingJob
from localml_scheduler.execution.control import ControlPlane, TrainingControlHook
from localml_scheduler.execution.runner_protocol import RunnerContext
from localml_scheduler.execution.worker_runtime import mark_job_completed
from localml_scheduler.observability.events import EventLogger
from localml_scheduler.runtime_environment import repair_generated_training_code, validate_generated_training_code
from localml_scheduler.storage import StateStore


def _context(settings: SchedulerSettings, job: TrainingJob) -> RunnerContext:
    store = StateStore(settings)
    store.save_job(job)
    event_logger = EventLogger(store, settings.events_jsonl_path)
    checkpoint_manager = CheckpointManager(settings, store, event_logger)
    control_plane = ControlPlane(settings)
    control_plane.initialize_job(job.job_id)
    return RunnerContext(
        job=job,
        settings=settings,
        store=store,
        event_logger=event_logger,
        control_hook=TrainingControlHook(job, control_plane, checkpoint_manager, store, event_logger),
        checkpoint_manager=checkpoint_manager,
        cache_client=None,
    )


def _elastic_script(path: Path, *, step_count: int = 10) -> None:
    path.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "import torch",
                "from torch.utils.data import TensorDataset",
                "from localml_scheduler.elastic import ElasticTrainingSession",
                "session = ElasticTrainingSession.from_env()",
                f"features = torch.arange({step_count * 16}, dtype=torch.float32).reshape({step_count * 8}, 2)",
                f"targets = torch.arange({step_count * 8}, dtype=torch.float32).reshape(-1, 1)",
                "loader = session.make_dataloader(TensorDataset(features, targets), shuffle=False)",
                "model = torch.nn.Linear(2, 1)",
                "optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)",
                "session.register_training_state(model, optimizer, extra_state={'metric': 0.0})",
                "progress = session.restore_if_present()",
                "global_step = progress['global_step']",
                "for batch_index, (inputs, labels) in enumerate(loader):",
                "    optimizer.zero_grad()",
                "    loss = ((model(inputs) - labels) ** 2).mean()",
                "    loss.backward()",
                "    optimizer.step()",
                "    global_step += 1",
                "    session.optimizer_step_completed(len(inputs), 0, batch_index, global_step, metrics={'loss': float(loss.item())})",
                "Path('elastic_result.json').write_text(json.dumps({'batch_size': session.batch_size, 'global_step': global_step}), encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _script_job(script: Path, result_path: Path, *, batch_size: int = 4) -> TrainingJob:
    return TrainingJob.create(
        runner_target="localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job",
        baseline_model_id="elastic-script",
        baseline_model_path=str(script),
        runner_kwargs={
            "script_path": str(script),
            "working_dir": str(script.parent),
            "result_path": str(result_path),
            "batch_size": batch_size,
            "timeout": 30,
        },
        batch_probe=BatchProbeSpec(
            enabled=True,
            probe_target="localml_scheduler.adapters.mlevolve_runner:probe_mlevolve_script_job",
            contract_version=3,
        ),
        checkpoint_policy=CheckpointPolicy(save_every_epoch=False),
        metadata={"placement_backend": "exclusive", "elastic_contract_validated": True},
    )


def test_missing_elastic_contract_is_rejected_and_one_repair_still_fails() -> None:
    source = "from torch.utils.data import DataLoader\nloader = DataLoader(range(8), batch_size=4)\n"
    validation = validate_generated_training_code(source, require_elastic_contract=True)
    assert validation["ok"] is False
    assert {issue["code"] for issue in validation["issues"]} == {
        "elastic_training_contract_missing",
        "elastic_training_loader_bypasses_session",
    }
    repaired = repair_generated_training_code(source, require_elastic_contract=True)
    assert repaired["validation"]["ok"] is False


def test_candidate_failure_marks_scheduler_job_failed(tmp_path: Path) -> None:
    settings = SchedulerSettings(runtime_root=tmp_path / "runtime")
    job = TrainingJob.create("pkg.runner:run", "model", "/tmp/model")
    context = _context(settings, job)
    code = mark_job_completed(
        settings,
        context.store,
        context.event_logger,
        job.job_id,
        {"success": False, "outcome": "candidate_exception", "candidate_exc_type": "RuntimeError"},
    )
    assert code == 0
    assert context.store.get_job(job.job_id).status == JobStatus.FAILED


def test_probe_failure_classification_uses_terminal_exception() -> None:
    kind, message = classify_mlevolve_probe_failure(
        stderr_text="Traceback (most recent call last):\nRuntimeError: fallback loop never executed\n",
        returncode=1,
    )
    assert kind == "script_exception"
    assert message == "fallback loop never executed"


def test_runner_executes_original_elastic_script_at_current_batch(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = workspace / "candidate.py"
    _elastic_script(script)
    settings = SchedulerSettings(runtime_root=tmp_path / "runtime")
    job = _script_job(script, workspace / "result.json", batch_size=4)
    job.current_batch_size = 8
    context = _context(settings, job)

    result = run_mlevolve_script_job(context)

    assert result["success"] is True
    assert result["candidate_returncode"] == 0
    assert result["batch_size_override"] == 8
    assert result["instrumentation"] == {"elastic_contract_version": 1}
    payload = json.loads((workspace / "elastic_result.json").read_text(encoding="utf-8"))
    assert payload["batch_size"] == 8
    assert context.store.get_job(job.job_id).config.runner_kwargs["batch_size"] == 4


def test_probe_uses_clean_process_and_requested_batch_with_end_to_end_throughput(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = workspace / "candidate.py"
    _elastic_script(script, step_count=20)
    settings = SchedulerSettings(runtime_root=tmp_path / "runtime")
    job = _script_job(script, workspace / "result.json", batch_size=8)
    context = _context(settings, job)

    result = probe_mlevolve_script_job(context, batch_size=2, warmup_steps=2, measure_steps=5)

    assert result.fits is True
    assert result.probe_completed is True
    assert result.avg_step_time_ms is not None and result.avg_step_time_ms > 0
    assert result.samples_per_second is not None and result.samples_per_second > 0
    assert result.peak_vram_mb is not None


def test_runner_rejects_noncontract_script_before_subprocess(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = workspace / "bad.py"
    script.write_text("print('not elastic')\n", encoding="utf-8")
    settings = SchedulerSettings(runtime_root=tmp_path / "runtime")
    job = _script_job(script, workspace / "result.json")
    context = _context(settings, job)
    with pytest.raises(RuntimeError, match="elastic training contract validation failed"):
        run_mlevolve_script_job(context)
