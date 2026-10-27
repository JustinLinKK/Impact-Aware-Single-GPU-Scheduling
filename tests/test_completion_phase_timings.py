"""CPU regressions for observed completion work, separate from scheduler waits."""
import builtins
import json
import time
from pathlib import Path

import pytest
import torch

from localml_scheduler.adapters import shared_training
from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import CheckpointPolicy, ResourceRequirements, SafePointType, TrainingJob
from localml_scheduler.execution.control import PauseRequested
from localml_scheduler.tests.fixtures import shared_training_candidate as fixture
from localml_scheduler.tests.test_mlevolve_runner import _build_context
from localml_scheduler.tests.test_shared_training import equal


def context_for(tmp_path, name="candidate", *, identity=None):
    settings = SchedulerSettings(runtime_root=tmp_path / "runtime",
        graph_db={"enabled": False}, log_db={"enabled": False})
    job = TrainingJob.create("test:runner", name, "/tmp/unused", job_id=name,
        max_epochs=2, runner_kwargs={"batch_size": 4},
        resource_requirements=ResourceRequirements(requires_gpu=False),
        checkpoint_policy=CheckpointPolicy(save_every_epoch=True, keep_last_n=1),
        metadata={"runtime_completion_identity": identity} if identity else {})
    return _build_context(settings, job)


def test_validation_and_finalization_measure_work_without_scheduler_yields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = context_for(tmp_path)
    workload = fixture.prepare()
    validate, save, clone = workload.validate, shared_training._atomic_save, torch.Tensor.clone
    validation_spans, copied, safe_points = [], [], []
    validation_started = None

    def slow_validate(model, device):
        nonlocal validation_started
        assert device.type == "cpu"
        validation_started = time.perf_counter()
        time.sleep(0.03)
        return validate(model, device)

    def slow_clone(tensor, *args, **kwargs):
        if validation_started is not None:
            time.sleep(0.01)
            copied.append(tensor.shape)
        return clone(tensor, *args, **kwargs)

    def slow_print(*args, **kwargs):
        nonlocal validation_started
        if args and str(args[0]).startswith("MLEVOLVE_EPOCH_METRIC"):
            time.sleep(0.02)
            validation_spans.append(time.perf_counter() - validation_started)
            validation_started = None
        builtins.print(*args, **kwargs)

    def slow_save(value, path):
        time.sleep(0.02)
        save(value, path)

    def finalize(model, device):
        metadata = context.store.get_job(context.job.job_id).metadata
        assert metadata["runtime_final_checkpoint_complete"]
        assert not metadata["runtime_finalization_complete"]
        assert (workload.output_dir / "selected_model.pt").exists()
        time.sleep(0.04)
        (workload.output_dir / "finalizer.txt").write_text("finished")

    original_safe_point = context.control_hook.safe_point

    def yielding_safe_point(kind, **kwargs):
        metadata = context.store.get_job(context.job.job_id).metadata
        safe_points.append(kind)
        assert metadata["runtime_total_optimizer_updates"] == 24
        assert metadata["runtime_planned_epochs"] == 2
        if kind == SafePointType.EPOCH:
            observation = metadata["runtime_phase_timing_history"][-1]
            assert observation["phase"] == "validation"
            assert observation["epoch"] == kwargs["epoch"]
            assert observation["global_step"] == kwargs["global_step"]
            # Waiting at a scheduler barrier must not enter any phase duration.
            time.sleep(0.2)
        original_safe_point(kind, **kwargs)

    def no_cuda(*args, **kwargs):
        pytest.fail("CPU instrumentation must not synchronize or allocate CUDA")

    workload.validate, workload.finalize = slow_validate, finalize
    monkeypatch.setattr(torch.Tensor, "clone", slow_clone)
    monkeypatch.setattr(torch.cuda, "synchronize", no_cuda)
    monkeypatch.setattr(shared_training, "print", slow_print, raising=False)
    monkeypatch.setattr(shared_training, "_atomic_save", slow_save)
    monkeypatch.setattr(context.control_hook, "safe_point", yielding_safe_point)
    result = shared_training.run_candidate_training(lambda: workload, context=context)

    metadata = context.store.get_job(context.job.job_id).metadata
    observations = metadata["runtime_phase_timing_history"]
    validation = [sample for sample in observations if sample["phase"] == "validation"]
    finalization = [sample for sample in observations if sample["phase"] == "finalization"]
    assert [sample["epoch"] for sample in validation] == [1, 2]
    assert copied  # Best-state CPU copies are inside the measured validation work.
    for sample, measured_work in zip(validation, validation_spans):
        assert sample["seconds"] >= measured_work
        assert sample["seconds"] < measured_work + 0.1
        assert sample["seconds"] >= 0.05
    assert len(finalization) == 1
    assert finalization[0]["seconds"] >= 0.08  # Both output saves plus finalizer.
    assert metadata["runtime_finalization_complete"]
    assert metadata["runtime_has_finalizer"]
    assert metadata["runtime_checkpoint_completed_step"] == 24
    assert metadata["runtime_checkpoint_completed_epoch"] == 2
    assert metadata["runtime_nonupdate_phase_counter"] == len(observations) == 6
    assert all(sample["identity"] == result["signature"] for sample in observations)
    assert all(sample["hardware_key"] == context.store.hardware_key() for sample in observations)
    assert all(sample["finished_at"] for sample in observations)
    assert {sample["source"] for sample in observations} == {
        "shared_training.validation", "checkpoint_manager.save_checkpoint", "shared_training.finalization"}
    assert safe_points.count(SafePointType.STEP) == 24
    for index, kind in enumerate(safe_points):
        if kind == SafePointType.EPOCH:
            assert safe_points[index + 1] == SafePointType.MANUAL
    state = torch.load(workload.output_dir / "training_state.pt", weights_only=False)
    assert state["global_step"] == 24
    assert all(int(value["step"]) == 24 for value in state["optimizer_state"]["state"].values())
    assert not any(key.startswith("runtime_") for key in state)
    assert json.loads((workload.output_dir / "result.json").read_text()) == result


def test_checkpoint_duration_includes_save_record_and_prune(tmp_path, monkeypatch):
    context = context_for(tmp_path, identity="runner-signature")
    manager = context.checkpoint_manager
    original_save, original_record, original_prune = torch.save, context.store.record_checkpoint, manager._prune
    delays = []

    def delayed(name, operation):
        def call(*args, **kwargs):
            time.sleep(0.02)
            delays.append(name)
            return operation(*args, **kwargs)
        return call

    monkeypatch.setattr(torch, "save", delayed("save", original_save))
    monkeypatch.setattr(context.store, "record_checkpoint", delayed("record", original_record))
    monkeypatch.setattr(manager, "_prune", delayed("prune", original_prune))
    first = None
    for kind, epoch, step in [(SafePointType.STEP, 0, 4), (SafePointType.EPOCH, 1, 12),
                              (SafePointType.EXPLICIT, 2, 24)]:
        path = manager.save_checkpoint(context.store.get_job(context.job.job_id),
            state={"global_step": step}, safe_point_type=kind, epoch=epoch, global_step=step, reason="test")
        first = first or path
        metadata = context.store.get_job(context.job.job_id).metadata
        sample = metadata["runtime_phase_timing_history"][-1]
        assert sample["seconds"] >= 0.06
        assert sample["identity"] == "runner-signature"
        assert sample["hardware_key"] == context.store.hardware_key()
        assert metadata["runtime_checkpoint_completed_step"] == step
        assert metadata.get("runtime_checkpoint_completed_epoch") == (None if epoch == 0 else 1)
        assert bool(metadata.get("runtime_final_checkpoint_complete")) == (kind == SafePointType.EXPLICIT)
        assert manager.load_checkpoint(path)["state"] == {"global_step": step}
    assert delays == ["save", "record", "prune"] * 3
    assert not Path(first).exists()
    events = context.store.list_events(job_id=context.job.job_id, event_type="checkpoint_saved")
    samples = metadata["runtime_phase_timing_history"]
    assert sorted(event["payload"]["checkpoint_seconds"] for event in events) == sorted(sample["seconds"] for sample in samples)
    assert metadata["runtime_nonupdate_phase_counter"] == 3


def test_unidentified_checkpoints_do_not_publish_completion_evidence(tmp_path):
    context = context_for(tmp_path)
    context.checkpoint_manager.save_checkpoint(context.job, state={"x": 1},
        safe_point_type=SafePointType.EXPLICIT, epoch=2, global_step=24, reason="test")
    metadata = context.store.get_job(context.job.job_id).metadata
    assert "runtime_phase_timing_history" not in metadata
    assert "runtime_final_checkpoint_complete" not in metadata


def test_failed_finalizer_does_not_claim_completion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = context_for(tmp_path)
    workload = fixture.prepare()

    def fail(model, device):
        raise RuntimeError("final output failed")

    workload.finalize = fail
    with pytest.raises(RuntimeError, match="final output failed"):
        shared_training.run_candidate_training(lambda: workload, context=context)
    metadata = context.store.get_job(context.job.job_id).metadata
    assert metadata["runtime_final_checkpoint_complete"]
    assert not metadata["runtime_finalization_complete"]
    assert not any(sample["phase"] == "finalization" for sample in metadata["runtime_phase_timing_history"])
    assert not (workload.output_dir / "result.json").exists()


def test_phase_clock_synchronizes_before_each_cuda_boundary_without_allocating(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("synchronize", str(device))))

    def clock():
        calls.append(("clock", None))
        return 123.0

    monkeypatch.setattr(shared_training.time, "perf_counter", clock)
    assert shared_training._phase_clock(torch.device("cuda:0")) == 123.0
    assert shared_training._phase_clock(torch.device("cuda:0")) == 123.0
    assert calls == [("synchronize", "cuda:0"), ("clock", None)] * 2


def test_completion_observations_are_bounded_finite_and_identity_bound(tmp_path):
    context = context_for(tmp_path, identity="current")
    manager = context.checkpoint_manager
    for step in range(70):
        shared_training._record_completion_phase(context, "validation", 0.01,
            identity="current", epoch=step, global_step=step)
        manager._record_completion_timing(context.job, 0.02, SafePointType.STEP, step, step)
    before = context.store.get_job(context.job.job_id).metadata
    assert len(before["runtime_phase_timing_history"]) == 64
    assert before["runtime_nonupdate_phase_counter"] == 140
    for seconds in [-1, float("nan"), float("inf")]:
        shared_training._record_completion_phase(context, "validation", seconds,
            identity="current", epoch=70, global_step=70)
        manager._record_completion_timing(context.job, seconds, SafePointType.EXPLICIT, 70, 70)
    shared_training._record_completion_phase(context, "validation", 1,
        identity="stale", epoch=70, global_step=70)
    shared_training._record_completion_phase(context, "untrusted", 1,
        identity="current", epoch=70, global_step=70)
    stale_job = context.job.copy()
    stale_job.metadata["runtime_completion_identity"] = "stale"
    manager._record_completion_timing(stale_job, 1, SafePointType.EXPLICIT, 70, 70)
    assert context.store.get_job(context.job.job_id).metadata == before


def test_pause_after_epoch_checkpoint_resumes_exact_updates_and_rng(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = context_for(tmp_path, "resumed")
    original_safe_point = context.control_hook.safe_point

    def pause_after_epoch(kind, **kwargs):
        original_safe_point(kind, **kwargs)
        if kind == SafePointType.EPOCH and kwargs["epoch"] == 1:
            metadata = context.store.get_job(context.job.job_id).metadata
            assert metadata["runtime_checkpoint_completed_epoch"] == 1
            context.control_hook.control_plane.request_pause(context.job.job_id, reason="after epoch", hold=True)

    monkeypatch.setattr(context.control_hook, "safe_point", pause_after_epoch)
    with pytest.raises(PauseRequested):
        shared_training.run_candidate_training(fixture.prepare, context=context)
    paused = context.store.get_job(context.job.job_id)
    checkpoint = context.checkpoint_manager.load_checkpoint(paused.latest_checkpoint_path)
    assert checkpoint["safe_point_type"] == SafePointType.MANUAL.value
    assert checkpoint["state"]["global_step"] == 12
    assert checkpoint["state"]["completed_epochs"] == 1
    paused.resume_from_checkpoint = paused.latest_checkpoint_path
    resumed = _build_context(context.settings, paused)
    shared_training.run_candidate_training(fixture.prepare, context=resumed)
    actual = torch.load("working/shared_training/training_state.pt", weights_only=False)
    metadata = resumed.store.get_job(paused.job_id).metadata
    validation = [sample for sample in metadata["runtime_phase_timing_history"] if sample["phase"] == "validation"]
    assert [sample["epoch"] for sample in validation] == [1, 2]
    assert metadata["runtime_finalization_complete"]
    assert not metadata["runtime_has_finalizer"]
    shared_training.run_candidate_training(fixture.prepare, context=context_for(tmp_path, "uninterrupted"))
    expected = torch.load("working/shared_training/training_state.pt", weights_only=False)
    equal(actual, expected)
    assert actual["global_step"] == 24
    assert all(int(state["step"]) == 24 for state in actual["optimizer_state"]["state"].values())


@pytest.mark.parametrize("resume_older", [True, False], ids=["older-checkpoint", "fresh-restart"])
def test_restart_does_not_reuse_completion_markers_ahead_of_restored_state(tmp_path, monkeypatch, resume_older):
    monkeypatch.chdir(tmp_path)
    context = context_for(tmp_path)
    original_safe_point = context.control_hook.safe_point
    older_checkpoint = None

    def remember_first_epoch(kind, **kwargs):
        nonlocal older_checkpoint
        original_safe_point(kind, **kwargs)
        if kind == SafePointType.EPOCH and kwargs["epoch"] == 1:
            job = context.store.get_job(context.job.job_id)
            older_checkpoint = context.checkpoint_manager.load_checkpoint(job.latest_checkpoint_path)

    monkeypatch.setattr(context.control_hook, "safe_point", remember_first_epoch)
    shared_training.run_candidate_training(fixture.prepare, context=context)
    expected = torch.load("working/shared_training/training_state.pt", weights_only=False)
    prior = context.store.get_job(context.job.job_id).metadata
    assert prior["runtime_checkpoint_completed_step"] == 24
    assert prior["runtime_checkpoint_completed_epoch"] == 2
    assert prior["runtime_final_checkpoint_complete"]
    assert prior["runtime_finalization_complete"]
    assert older_checkpoint["state"]["global_step"] == 12
    resumed_epochs, initial_metadata = [], []

    def check_restored_progress(kind, **kwargs):
        if kind == SafePointType.BEFORE_TRAIN:
            metadata = context.store.get_job(context.job.job_id).metadata
            initial_metadata.append(metadata)
            assert metadata["runtime_checkpoint_completed_step"] == (12 if resume_older else 0)
            assert metadata["runtime_checkpoint_completed_epoch"] == (1 if resume_older else 0)
            assert not metadata["runtime_final_checkpoint_complete"]
            assert not metadata["runtime_finalization_complete"]
            assert metadata["runtime_nonupdate_phase_counter"] == prior["runtime_nonupdate_phase_counter"]
            assert metadata["runtime_phase_timing_history"] == prior["runtime_phase_timing_history"]
        if kind == SafePointType.EPOCH:
            resumed_epochs.append(kwargs["epoch"])
        original_safe_point(kind, **kwargs)

    monkeypatch.setattr(context.control_hook, "safe_point", check_restored_progress)
    monkeypatch.setattr(type(context), "load_resume_checkpoint", lambda self: older_checkpoint if resume_older else None)
    shared_training.run_candidate_training(fixture.prepare, context=context)
    assert len(initial_metadata) == 1
    assert resumed_epochs == ([2] if resume_older else [1, 2])
    actual = torch.load("working/shared_training/training_state.pt", weights_only=False)
    equal(actual, expected)
    assert actual["global_step"] == 24
    final_metadata = context.store.get_job(context.job.job_id).metadata
    assert final_metadata["runtime_nonupdate_phase_counter"] > prior["runtime_nonupdate_phase_counter"]
    assert final_metadata["runtime_final_checkpoint_complete"]
    assert final_metadata["runtime_finalization_complete"]
