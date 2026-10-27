import concurrent.futures
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from engine.script_introspection import supports_cooperative_trial
from localml_scheduler.adapters.mlevolve_runner import run_mlevolve_script_job
from localml_scheduler.adapters.shared_training import _check_runtime_recipe, run_candidate_training
from localml_scheduler.domain import JobStatus
from localml_scheduler.execution.control import PauseRequested
from localml_scheduler.tests.test_cold_start_trials import setup
from localml_scheduler.tests.test_mlevolve_runner import _build_context
from localml_scheduler.tests.fixtures import shared_training_candidate as fixture


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def test_shared_entry_recognition_rejects_missing_or_shadowed_loop():
    code = Path(fixture.__file__).read_text()
    assert supports_cooperative_trial(code)
    assert not supports_cooperative_trial(code.replace('run_candidate_training(prepare)', 'print(prepare)'))
    assert not supports_cooperative_trial(code.replace('if __name__', 'run_candidate_training = print\nif __name__'))
    assert not supports_cooperative_trial(code.replace('localml_scheduler.adapters.shared_training', 'untrusted.shared_training'))
    assert not supports_cooperative_trial(code.replace('run_candidate_training(prepare)', 'run_candidate_training(prepare, context=object())'))


def test_shared_real_child_pause_resume_exact_state(tmp_path):
    settings, _, _, jobs = setup(tmp_path / "runtime")
    settings.ensure_runtime_layout()
    (settings.runtime_root / "scheduler_settings.json").write_text(json.dumps(settings.to_dict()))
    script = tmp_path / "generated.py"
    script.write_text(Path(fixture.__file__).read_text())

    def context_for(name):
        work = tmp_path / name
        work.mkdir()
        job = jobs[0].copy()
        job.job_id = name
        job.max_epochs = job.config.max_epochs = 2
        job.config.runner_target = "localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job"
        job.config.runner_kwargs = {"script_path": str(script), "working_dir": str(work),
            "result_path": str(work / "execution.json"), "batch_size": 4}
        return _build_context(settings, job)

    context = context_for("resume")
    with concurrent.futures.ThreadPoolExecutor(1) as executor:
        future = executor.submit(run_mlevolve_script_job, context)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            current = context.store.get_job("resume")
            if int(current.metadata.get("runtime_global_step", 0)) >= 4:
                break
            assert not future.done(), future.result() if future.done() else ""
            time.sleep(0.005)
        else:
            pytest.fail("runner did not publish progress")
        context.control_hook.control_plane.request_pause("resume", reason="shared-loop-regression", hold=True)
        with pytest.raises(PauseRequested):
            future.result(timeout=15)
    paused = context.store.get_job("resume")
    saved = torch.load(paused.latest_checkpoint_path, weights_only=False)["state"]
    assert 4 <= saved["global_step"] < 24
    paused.resume_from_checkpoint = paused.latest_checkpoint_path
    paused.status, paused.hold = JobStatus.RUNNING, False
    resumed = _build_context(settings, paused)
    assert run_mlevolve_script_job(resumed)["candidate_returncode"] == 0
    control = context_for("control")
    assert run_mlevolve_script_job(control)["candidate_returncode"] == 0
    final = torch.load(tmp_path / "resume/working/shared_training/training_state.pt", weights_only=False)
    expected = torch.load(tmp_path / "control/working/shared_training/training_state.pt", weights_only=False)
    equal(final, expected)
    assert final["global_step"] == 24
    assert all(int(state["step"]) == 24 for state in final["optimizer_state"]["state"].values())


def test_last_batch_pause_keeps_pending_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from localml_scheduler.domain import SafePointType
    import copy
    monkeypatch.chdir(tmp_path)
    checkpoint = None
    pause = True
    def safe_point(kind, **kwargs):
        nonlocal checkpoint, pause
        if pause and kind == SafePointType.STEP and kwargs["global_step"] == 12:
            checkpoint = {"state": copy.deepcopy(kwargs["state_factory"]())}
            pause = False
            raise PauseRequested()
    context = SimpleNamespace(job=SimpleNamespace(resource_requirements=SimpleNamespace(requires_gpu=False)),
        load_resume_checkpoint=lambda: checkpoint, control_hook=SimpleNamespace(safe_point=safe_point))
    monkeypatch.setattr("localml_scheduler.adapters.shared_training._check_runtime_recipe", lambda *args: None)
    with pytest.raises(PauseRequested):
        run_candidate_training(fixture.prepare, context=context)
    assert checkpoint["state"]["completed_epochs"] == 0
    run_candidate_training(fixture.prepare, context=context)
    final = torch.load("working/shared_training/training_state.pt", weights_only=False)
    checkpoint = None
    run_candidate_training(fixture.prepare, context=context)
    expected = torch.load("working/shared_training/training_state.pt", weights_only=False)
    equal(final, expected)
    assert len(final["history"]) == 2


def test_prediction_recipe_rejects_wrong_batch_and_context():
    from types import SimpleNamespace
    from localml_scheduler.domain import TrainingJob
    workload = fixture.prepare()
    job = TrainingJob.create("test:runner", "test", "/tmp/none", max_epochs=2, runner_kwargs={"batch_size": 8})
    with pytest.raises(ValueError, match="batch"):
        _check_runtime_recipe(workload, SimpleNamespace(job=job))
    job.config.runner_kwargs["batch_size"] = 4
    job.metadata["perfseer_model"] = {"workload_contract": {
        "training": workload.recipe.prediction_training(), "adapter": {"context": {"wrong": 1}}}}
    with pytest.raises(ValueError, match="context"):
        _check_runtime_recipe(workload, SimpleNamespace(job=job))


def test_immediate_resume_waits_for_parent_exit_and_can_be_cancelled(tmp_path, monkeypatch):
    from localml_scheduler.scheduler.service import SchedulerService
    settings, store, _, jobs = setup(tmp_path)
    service = SchedulerService(settings, store=store)
    job = jobs[0]
    store.set_job_status(job.job_id, JobStatus.PAUSED, reason="checkpoint saved", hold=True)
    monkeypatch.setattr(service.supervisor, "has_live_process", lambda job_id: True)
    service._handle_resume(job.job_id)
    assert store.get_job(job.job_id).status == JobStatus.PAUSED
    assert store.get_job(job.job_id).metadata["resume_after_worker_exit"]
    service._release_deferred_resumes()
    assert store.get_job(job.job_id).status == JobStatus.PAUSED
    monkeypatch.setattr(service.supervisor, "has_live_process", lambda job_id: False)
    service._release_deferred_resumes()
    assert store.get_job(job.job_id).status == JobStatus.READY
    assert not store.get_job(job.job_id).hold
    store.set_job_status(job.job_id, JobStatus.PAUSED, reason="second pause", hold=True)
    monkeypatch.setattr(service.supervisor, "has_live_process", lambda job_id: True)
    service._handle_resume(job.job_id)
    service._handle_pause(job.job_id)  # A newer manual pause cancels queued resume.
    monkeypatch.setattr(service.supervisor, "has_live_process", lambda job_id: False)
    service._release_deferred_resumes()
    assert store.get_job(job.job_id).status == JobStatus.PAUSED
    assert store.get_job(job.job_id).hold
