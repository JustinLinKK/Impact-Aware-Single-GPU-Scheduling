from datetime import datetime, timezone
from pathlib import Path
import time
from unittest.mock import Mock
import json

import pytest

from localml_scheduler.config import SchedulerSettings
from localml_scheduler.domain import JobStatus, PackingSpec, ResourceRequirements, TrainingJob
from localml_scheduler.scheduler.cold_start import normalized_throughput
from localml_scheduler.scheduler.placement_planner import PlacementPlanner
from localml_scheduler.scheduler.policies import PriorityFifoPolicy
from localml_scheduler.scheduler.service import SchedulerService
from localml_scheduler.scheduler.service_state import ColocationTrialState
from localml_scheduler.storage.sqlite_store import SQLiteStateStore


def setup(tmp_path, backend="cuda_process"):
    settings = SchedulerSettings(runtime_root=tmp_path, gpu_scheduler={
        "packing_backend": backend, "memory": {"gpu_vram_gib": 31},
    }, graph_db={"enabled": False}, log_db={"enabled": False})
    store = SQLiteStateStore(settings)
    planner = PlacementPlanner(settings, store, PriorityFifoPolicy())
    jobs = []
    for name in ("a", "b", "c"):
        job = TrainingJob.create(
            "pkg:runner", name, "/tmp/unused.pt", job_id=name,
            max_epochs=3, runner_kwargs={"batch_size": 64},
            packing=PackingSpec(eligible=True, signature=name, backend_allowlist=[backend]),
            resource_requirements=ResourceRequirements(requires_gpu=False),
            metadata={"cooperative_trial": True, "placement_backend": backend},
        )
        store.submit_job(job)
        jobs.append(job)
    return settings, store, planner, jobs


@pytest.mark.parametrize("backend", ["cuda_process", "mps_process"])
def test_empty_profiles_start_packed_anchor_then_addition(tmp_path, backend):
    settings, store, planner, jobs = setup(tmp_path, backend)
    available = {backend: True, "exclusive": True}
    anchor = planner.choose_plan(jobs, backend_available=available)
    assert anchor.mode == "stack_anchor"
    assert anchor.backend_name == backend
    assert not anchor.batch_overrides
    assert settings.gpu_scheduler.parallel_job_cap is None
    for missing_epochs in (False, True):
        if missing_epochs:
            jobs[0].max_epochs = jobs[0].config.max_epochs = None
            jobs[1].max_epochs = jobs[1].config.max_epochs = None
        plan = planner.choose_plan(jobs[1:], active_jobs=jobs[:1], backend_available=available)
        assert plan.trial_metadata["cold_start"]
        assert plan.trial_metadata["requires_live_trial"]
        assert plan.trial_metadata["memory_limit_mb"] > 0
        assert plan.objective_breakdown["gain"] is None


def test_guards_and_combination_local_backoff(tmp_path):
    settings, store, planner, jobs = setup(tmp_path)
    kwargs = {"active_jobs": jobs[:1], "backend_available": {"cuda_process": True}}
    plan = planner.choose_plan(jobs[1:2], **kwargs)
    jobs[1].metadata["cold_start_blocked_groups"] = [plan.trial_metadata["profile_key"]]
    assert planner.choose_plan(jobs[1:], **kwargs).job_ids == ("c",)
    assert planner.choose_plan(jobs[1:], **kwargs, trial_pending=True) is None
    assert planner.choose_plan(jobs[1:], **kwargs, admission_open=False) is None
    assert planner.choose_plan(jobs[1:], **kwargs, active_vram_mb=31 * 1024) is None
    settings.gpu_scheduler.colocation.live_trial_enabled = False
    assert planner.choose_plan(jobs[1:], **kwargs) is None


@pytest.mark.parametrize("packed,decision", [
    ({"a": [1.1, 1.1], "b": [1.1, 1.1]}, "accepted"),
    ({"a": [3, 3], "b": [3, 3]}, "rejected"),
    ({"a": [2, 2], "b": [2, 2]}, "inconclusive"),
    ({"a": [1, 3], "b": [1, 3]}, "inconclusive"),
    ({"a": [float("nan"), 1], "b": [1, 1]}, "inconclusive"),
])
def test_throughput_uses_both_halves(packed, decision):
    assert normalized_throughput({"a": [1, 1], "b": [1, 1]}, {"a": [1, 1]}, packed, "b", 0.03)[0] == decision


@pytest.mark.parametrize("profiled", ["a", "b"])
def test_one_profile_is_not_required_for_the_other_job(tmp_path, profiled):
    from localml_scheduler.domain import RuntimeProfile
    settings, store, planner, jobs = setup(tmp_path)
    store.upsert_runtime_profile(RuntimeProfile.create(
        signature=profiled, hardware_key=store.hardware_key(), backend_name="cuda_process",
        resolved_batch_size=64, avg_step_time_ms=2, steps_per_epoch=100,
        epoch_1_seconds=0.2, estimated_total_runtime_seconds=0.6, strategy="epoch_1",
    ))
    plan = planner.choose_plan([jobs[1]], active_jobs=[jobs[0]], backend_available={"cuda_process": True})
    assert plan.trial_metadata["cold_start"]
    assert not plan.batch_overrides


def test_missing_or_stale_window_is_inconclusive(tmp_path):
    from localml_scheduler.scheduler.cold_start import ColdStartTrialMixin
    _, _, _, jobs = setup(tmp_path)
    jobs[0].metadata.update({"trial_step_command": {"token": "new"}, "trial_step_window": {"token": "old", "complete": True}})
    assert ColdStartTrialMixin._cold_window(jobs[0]) is None
    jobs[0].metadata["trial_step_window"].update({"token": "new", "halves": [float("nan"), 1]})
    assert ColdStartTrialMixin._cold_window(jobs[0]) is None
    assert normalized_throughput({"a": [1, 1]}, {"a": [1, 1]}, {"b": [1, 1]}, "b", 0.03)[0] == "inconclusive"


def test_mps_unprofiled_anchor_and_newcomers_keep_startup_ceilings(tmp_path):
    from unittest.mock import patch
    from localml_scheduler.tests.test_process_backends import _RecordingExecutor
    from localml_scheduler.execution.backends import MPSBackend
    settings, _, planner, jobs = setup(tmp_path, "mps_process")
    executor = _RecordingExecutor(tmp_path)
    backend = MPSBackend(settings, executor, mps_binary="mps")
    for index, job in enumerate(jobs):
        if index == 1:
            job.metadata["mps_active_thread_pct"] = 60
        plan = planner.choose_plan([job], active_jobs=jobs[:index], backend_available={"mps_process": True})
        assert plan.backend_config["allocation_percentages"] == [60 if index == 1 else 100]
        job.metadata["placement_backend_config"] = plan.backend_config
        job.metadata["mps_active_thread_pct"] = plan.backend_config["allocation_percentages"][0]
        with patch.object(MPSBackend, "_ensure_runtime"):
            backend.launch([job])
    assert [env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] for _, env in executor.calls] == ["100", "60", "100"]
    assert jobs[0].metadata["mps_active_thread_pct"] == 100


def test_allocator_limit_respects_live_free_memory(tmp_path):
    from unittest.mock import patch
    from localml_scheduler.execution.trial_control import configure_trial_memory
    _, store, _, jobs = setup(tmp_path)
    job = jobs[0]
    job.resource_requirements.requires_gpu = True
    job.metadata["trial_memory_limit_mb"] = 2048
    context = Mock(job=job, store=store)
    with patch("torch.cuda.mem_get_info", return_value=(1024 * 1024**2, 32 * 1024**3)), patch("torch.cuda.set_per_process_memory_fraction") as guard:
        configure_trial_memory(context)
    assert guard.call_args.args[0] == 768 / (32 * 1024)


@pytest.mark.parametrize("mode,expected", [("origin", True), ("baseline", False)])
def test_origin_reuses_compatible_branch_profile_but_baseline_does_not(tmp_path, mode, expected):
    from localml_scheduler.domain import RuntimeProfile
    _, store, planner, jobs = setup(tmp_path)
    for job in jobs[:2]:
        job.workflow_id = "workflow"
        job.packing.family = "mlp"
        job.metadata.update({"branch_id": "branch", "experiment_mode": mode})
        store.save_job(job)
    store.upsert_runtime_profile(RuntimeProfile.create(
        signature="a", hardware_key=store.hardware_key(), backend_name="cuda_process",
        resolved_batch_size=64, strategy="epoch_1", epoch_1_seconds=1,
        estimated_total_runtime_seconds=3, last_job_id="a",
    ))
    profile = planner.estimator._compatible_branch_runtime_profile(jobs[1], batch_size=64, backend_name="cuda_process")
    assert (profile is not None) == expected


def test_state_machine_collects_missing_references_and_accepts(tmp_path):
    settings, store, planner, jobs = setup(tmp_path)
    active = jobs[:2]
    for job in active:
        store.set_job_status(job.job_id, JobStatus.RUNNING, reason="fixture")
        store.update_job(job.job_id, metadata_updates={"trial_runner_ready": True})
    supervisor = Mock()
    supervisor.active_job_ids.return_value = ["a", "b"]
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    plan = planner.choose_plan([jobs[1]], active_jobs=[jobs[0]], backend_available={"cuda_process": True})
    trial = service._prepare_colocation_trial(plan, jobs[1])
    assert trial.target_epoch == 1
    service._evaluate_colocation_trial()
    assert trial.phase == "pre_add"

    def window(name, halves):
        job = store.get_job(name)
        token = job.metadata["trial_step_command"]["token"]
        store.update_job(name, metadata_updates={"trial_step_window": {"token": token, "complete": True, "halves": halves}})

    window("a", [1, 1])
    service._evaluate_colocation_trial()
    assert trial.phase == "reference_barrier"
    for name in ("a", "b"):
        token = store.get_job(name).metadata["trial_step_command"]["token"]
        store.update_job(name, metadata_updates={"trial_yield_ack": token})
    service._evaluate_colocation_trial()
    assert trial.phase == "reference"
    window("b", [1, 1])
    service._evaluate_colocation_trial()
    assert trial.phase == "measure_overlap"
    window("a", [1.1, 1.1])
    window("b", [1.1, 1.1])
    service._evaluate_colocation_trial()
    assert service._colocation_trial is None
    result = store.get_job("b").metadata["colocation_trial"]
    assert result["decision"] == "accepted"
    assert result["result"]["objective"] == "normalized_throughput"
    assert not store.get_job("a").metadata["trial_step_command"]


def test_controller_restart_releases_all_owned_yields(tmp_path):
    settings, store, planner, jobs = setup(tmp_path)
    supervisor = Mock()
    supervisor.request_pause.return_value = True
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    plan = planner.choose_plan([jobs[1]], active_jobs=[jobs[0]], backend_available={"cuda_process": True})
    trial = service._prepare_colocation_trial(plan, jobs[1])
    for job in jobs[:2]:
        service._cold_command(trial, job, "yield", "reference_barrier")
    restored = SchedulerService(settings, store=store, supervisor=supervisor)
    assert restored._colocation_trial is None
    assert store.get_job("b").metadata["colocation_trial"]["decision"] == "inconclusive"
    assert all(not store.get_job(job.job_id).metadata["trial_step_command"] for job in jobs[:2])


def test_missing_subgroup_uses_throughput_without_calibration(tmp_path):
    settings, store, planner, jobs = setup(tmp_path)
    for index, job in enumerate(jobs):
        store.update_job(job.job_id, status=JobStatus.RUNNING, metadata_updates={
            "trial_runner_ready": True, "runtime_steps_per_epoch": 5000 if index == 0 else 100,
            "last_completed_epoch": 0,
        })
    supervisor = Mock()
    supervisor.active_job_ids.return_value = [job.job_id for job in jobs]
    service = SchedulerService(settings, store=store, supervisor=supervisor)
    plan = planner.choose_plan([jobs[2]], active_jobs=jobs[:2], backend_available={"cuda_process": True})
    trial = service._prepare_colocation_trial(plan, jobs[2])
    trial.cold_start.update({"references": {job.job_id: [1, 1] for job in jobs}, "pre_add": {job.job_id: [1, 1] for job in jobs[:2]}})
    service._cold_phase(trial, "measure_overlap", jobs)
    for job in jobs:
        current = store.get_job(job.job_id)
        store.update_job(job.job_id, metadata_updates={"trial_step_window": {
            "token": current.metadata["trial_step_command"]["token"], "complete": True, "halves": [1.2, 1.2],
        }})
    service._evaluate_colocation_trial()
    decision = store.get_job("c").metadata["colocation_trial"]
    assert decision["decision"] == "accepted"
    assert decision["result"]["objective"] == "normalized_throughput"
    assert decision["target_epoch"] == 1


def test_generated_fixture_declares_real_control_calls():
    from engine.script_introspection import supports_cooperative_trial
    from localml_scheduler.examples import cold_start_runner
    assert supports_cooperative_trial(Path(cold_start_runner.__file__).read_text())
    assert not supports_cooperative_trial("print('MLEVOLVE_EPOCH_METRIC')")
    code = Path(cold_start_runner.__file__).read_text()
    assert not supports_cooperative_trial(code.replace("optimizer.load_state_dict", "optimizer.other"))
    assert not supports_cooperative_trial(code.replace('"step_in_epoch"', '"missing_data_position"'))


def test_expired_yield_releases_without_checkpoint(tmp_path):
    from localml_scheduler.tests.test_mlevolve_runner import _build_context
    from localml_scheduler.domain import SafePointType
    settings, store, planner, jobs = setup(tmp_path)
    job = jobs[0]
    job.metadata["trial_step_command"] = {"owner": "dead", "token": "lease", "action": "yield", "expires_at": time.time() - 1}
    context = _build_context(settings, job)
    factory = Mock()
    context.control_hook.safe_point(SafePointType.STEP, epoch=0, global_step=1, state_factory=factory)
    factory.assert_not_called()
    assert store.get_job(job.job_id).metadata["trial_lease_expired"] == "dead"


def test_generated_child_pause_and_resume_preserve_exact_updates(tmp_path):
    import concurrent.futures
    import torch
    from localml_scheduler.tests.test_mlevolve_runner import _build_context
    from localml_scheduler.adapters.mlevolve_runner import run_mlevolve_script_job
    from localml_scheduler.examples import cold_start_runner
    from localml_scheduler.execution.control import PauseRequested

    settings, store, planner, jobs = setup(tmp_path / "runtime")
    settings.ensure_runtime_layout()
    (settings.runtime_root / "scheduler_settings.json").write_text(json.dumps(settings.to_dict()))
    script = tmp_path / "generated.py"
    script.write_text(Path(cold_start_runner.__file__).read_text())

    def context_for(name):
        job = jobs[0].copy()
        job.job_id = name
        job.config.runner_target = "localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job"
        job.config.runner_kwargs = {
            "script_path": str(script), "working_dir": str(tmp_path),
            "result_path": str(tmp_path / (name + ".json")),
            "batch_size": 4, "width": 16, "steps_per_epoch": 40, "input_delay_seconds": 0.003,
        }
        return _build_context(settings, job)

    context = context_for("resume")
    with concurrent.futures.ThreadPoolExecutor(1) as executor:
        future = executor.submit(run_mlevolve_script_job, context)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            current = context.store.get_job("resume")
            if int(current.metadata.get("runtime_global_step", 0)) >= 12:
                break
            assert not future.done(), future.result() if future.done() else ""
            time.sleep(0.01)
        else:
            pytest.fail("generated script did not publish step progress")
        context.control_hook.control_plane.request_pause("resume", reason="regression", hold=False)
        with pytest.raises(PauseRequested):
            future.result(timeout=10)
    paused = context.store.get_job("resume")
    assert paused.status == JobStatus.PAUSED
    checkpoint = context.checkpoint_manager.load_checkpoint(paused.latest_checkpoint_path)["state"]
    assert 12 <= checkpoint["global_step"] < 120
    paused.resume_from_checkpoint = paused.latest_checkpoint_path
    paused.status = JobStatus.RUNNING
    resumed = _build_context(settings, paused)
    assert run_mlevolve_script_job(resumed)["candidate_returncode"] == 0
    final = resumed.checkpoint_manager.load_checkpoint(resumed.store.get_job("resume").latest_checkpoint_path)["state"]
    control = context_for("control")
    assert run_mlevolve_script_job(control)["candidate_returncode"] == 0
    expected = control.checkpoint_manager.load_checkpoint(control.store.get_job("control").latest_checkpoint_path)["state"]
    assert final["global_step"] == expected["global_step"] == 120
    for name, value in final["model_state"].items():
        torch.testing.assert_close(value, expected["model_state"][name], rtol=0, atol=0)


def test_concurrent_progress_preserves_controller_commands(tmp_path):
    import concurrent.futures
    _, store, _, jobs = setup(tmp_path)

    def write(prefix):
        for index in range(30):
            store.update_job("a", metadata_updates={f"{prefix}-{index}": index})

    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        list(executor.map(write, ["controller", "worker"]))
    metadata = store.get_job("a").metadata
    assert all(metadata[f"{prefix}-{index}"] == index for prefix in ("controller", "worker") for index in range(30))


def test_generated_script_submission_rejects_requeues_and_completes(tmp_path, monkeypatch):
    import torch
    from localml_scheduler.client import SchedulerClient
    from localml_scheduler.examples import cold_start_runner
    from localml_scheduler.domain import CheckpointPolicy

    settings = SchedulerSettings(runtime_root=tmp_path / "runtime", scheduler_poll_interval_seconds=0.05,
        graph_db={"enabled": False}, hardware_knowledge_graph={"enabled": False}, hardware_feature_db={"enabled": False},
        gpu_scheduler={"packing_backend": "cuda_process", "batch_probe_enabled": False})
    client = SchedulerClient(settings)
    baseline = tmp_path / "baseline.pt"
    torch.save({}, baseline)
    script = tmp_path / "generated.py"
    script.write_text(Path(cold_start_runner.__file__).read_text())
    for name in ("incumbent", "newcomer"):
        client.submit(TrainingJob.create(
            "localml_scheduler.adapters.mlevolve_runner:run_mlevolve_script_job", name, str(baseline), job_id=name, max_epochs=3,
            runner_kwargs={"script_path": str(script), "working_dir": str(tmp_path), "result_path": str(tmp_path / f"{name}.json"),
                           "width": 16, "batch_size": 4, "steps_per_epoch": 240, "input_delay_seconds": 0.003},
            resource_requirements=ResourceRequirements(requires_gpu=False),
            packing=PackingSpec(eligible=True, signature=name, backend_allowlist=["cuda_process"]),
            checkpoint_policy=CheckpointPolicy(save_every_epoch=False, keep_last_n=5),
            metadata={"cooperative_trial": True, "skip_active_scheduler_probes": True},
        ))
    service = client.create_service()
    monkeypatch.setattr("localml_scheduler.scheduler.cold_start.normalized_throughput", lambda *args: ("rejected", [0.6, 0.6]))
    monkeypatch.setattr(service.planner.time_objective, "estimate_gain", lambda *args, **kwargs: None)
    service.start(background=True)
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if all(job.status.is_terminal for job in client.store.list_jobs()):
                break
            time.sleep(0.05)
        else:
            pytest.fail("generated-script rejection left a worker blocked")
    finally:
        service.stop()
    jobs = client.store.list_jobs()
    assert all(job.status == JobStatus.COMPLETED for job in jobs)
    pauses = client.store.list_events(event_type="job_paused")
    assert [event["job_id"] for event in pauses] == ["newcomer"]
    for job in jobs:
        saved = torch.load(job.latest_checkpoint_path, map_location="cpu", weights_only=False)["state"]
        assert saved["global_step"] == 720
        assert all(int(value["step"]) == 720 for value in saved["optimizer_state"]["state"].values())
