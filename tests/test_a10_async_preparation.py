"""CPU-only concurrency and admission boundaries for the research cache."""
from copy import deepcopy
from threading import Event, Thread
from types import SimpleNamespace
import time

import pytest

from localml_scheduler.benchmarks.a10_feedback_latency.async_preparation import (
    AsyncV4Service, AsyncRevisedService, PreparedPredictor, PreparationServiceMixin,
)
from localml_scheduler.benchmarks.a10_feedback_latency.calibration import RecordingPredictor
from localml_scheduler.benchmarks.a10_feedback_latency import async_preparation as module
from localml_scheduler.prediction import JobPredictionError
from localml_scheduler.scheduler.service import SchedulerService


def job(name="first", batch=4):
    return SimpleNamespace(job_id=name, packing=SimpleNamespace(signature="signature-" + name),
        metadata={"resolved_batch_size": batch}, batch_probe=SimpleNamespace(batch_param_name="batch_size"),
        config=SimpleNamespace(runner_kwargs={"batch_size": batch}))


class Predictor:
    available = True
    artifact_hash = "artifact"
    model_id = "release/S_A10"
    hardware = SimpleNamespace(hardware_key="dedicated-a10")

    def __init__(self, *, mutate=None):
        self.last_sources, self.last_errors = {}, {}
        self.calls = []
        self.slow_started, self.release = Event(), Event()
        self.mutate = mutate

    def predict_profile_options(self, target, batches):
        self.calls.append((target.job_id, list(batches)))
        if target.job_id == "slow":
            self.slow_started.set()
            assert self.release.wait(3), "test must release blocked preparation"
        outputs = {batch: dict(kind="predicted_workload", job_signature=target.packing.signature,
            batch_size=batch, artifact_sha256=self.artifact_hash, hardware="A10", step_cost=200.,
            epoch_cost=6000., avg_vram_mib=800., peak_vram_mib=1000., busy_indicator=70.,
            derived_solo_slope={"value": 5., "artifact_sha256": self.artifact_hash}) for batch in batches}
        if self.mutate:
            self.mutate(outputs[batches[0]])
        self.last_sources[target.job_id] = "ml_predictor"
        return outputs

    def predict_avg_vram_options(self, *_):
        pytest.fail("cache must not delegate helper calls")

    def predict_epoch_cost_options(self, *_):
        pytest.fail("cache must not delegate helper calls")


def wait_finished(prepared, target):
    deadline = time.monotonic() + 2
    while target.job_id not in prepared.preparations_finished:
        assert time.monotonic() < deadline, "preparation did not finish"
        Event().wait(.002)


def test_slow_preparation_cannot_block_cached_or_missing_reads():
    underlying = Predictor()
    prepared = PreparedPredictor(RecordingPredictor(underlying, lambda *_: None))
    first, slow = job(), job("slow")
    try:
        assert prepared.enqueue(first, 4)
        wait_finished(prepared, first)
        assert not prepared.enqueue(first, 4)
        prepared.enqueue(slow, 4)
        assert underlying.slow_started.wait(1)
        # Read in another thread so a lock held during inference fails promptly.
        result, finished = {}, Event()
        def read():
            try:
                result["cached"] = prepared.predict_profile_options(first, [4])[4]
                with pytest.raises(JobPredictionError):
                    prepared.predict_profile_options(slow, [4])
                result["ready"] = prepared.ready(slow)
            finally:
                finished.set()
        reader = Thread(target=read, daemon=True)
        reader.start()
        assert finished.wait(.3), "read blocked behind CPU preparation"
        assert result["cached"]["step_cost"] == 200.
        assert result["ready"] is False
        assert slow.job_id not in prepared.records
        assert len(underlying.calls) == 2
        # API callers cannot mutate published state.
        result["cached"]["derived_solo_slope"]["value"] = -1
        assert prepared.predict_profile_options(first, [4])[4]["derived_solo_slope"]["value"] == 5.
    finally:
        underlying.release.set()
        assert prepared.stop(timeout=1)


@pytest.mark.parametrize("field,value", [
    ("artifact_sha256", "other"), ("hardware", "V100"),
    ("job_signature", "other"), ("batch_size", 8),
    ("step_cost", float("nan")), ("busy_indicator", 101.), ("peak_vram_mib", 799.),
])
def test_invalid_complete_profiles_fail_closed(field, value):
    underlying = Predictor(mutate=lambda profile: profile.update({field: value}))
    prepared = PreparedPredictor(underlying, asynchronous=False)
    target = job()
    try:
        prepared.enqueue(target, 4)
        assert not prepared.ready(target)
        assert target.job_id in prepared.failures
        assert prepared.records == {}
        with pytest.raises(JobPredictionError): prepared.predict_avg_vram_mb(target, 4)
        assert prepared.events[-1]["status"] == "failed"
    finally:
        assert prepared.stop()


def test_job_batch_artifact_and_live_hardware_changes_cannot_reuse_cache():
    underlying = Predictor()
    prepared = PreparedPredictor(underlying, asynchronous=False)
    target = job()
    try:
        prepared.enqueue(target, 4)
        assert prepared.ready(target)
        changed = deepcopy(target)
        changed.packing.signature = "new-signature"
        assert not prepared.ready(changed)
        assert not prepared.ready(target, 8)
        with pytest.raises(JobPredictionError): prepared.predict_profile_options(changed, [4])
        underlying.artifact_hash = "new-artifact"
        assert not prepared.ready(target)
        underlying.artifact_hash = "artifact"
        underlying.hardware = SimpleNamespace(hardware_key="other-a10")
        assert not prepared.ready(target)
        assert len(underlying.calls) == 1
    finally:
        prepared.stop()


def test_exception_has_no_partial_publication_and_no_fallback():
    underlying = Predictor(mutate=lambda _: (_ for _ in ()).throw(RuntimeError("conversion failed")))
    prepared = PreparedPredictor(underlying, asynchronous=False)
    target = job()
    try:
        prepared.enqueue(target, 4)
        assert prepared.failures[target.job_id] == {"type": "RuntimeError", "error": "conversion failed"}
        assert prepared.last_errors[target.job_id] == "conversion failed"
        assert not prepared.ready(target)
        assert not prepared.enqueue(target, 4)
        with pytest.raises(JobPredictionError): prepared.predict_epoch_cost_options(target, [4])
        assert len(underlying.calls) == 1
        assert not hasattr(prepared, "_released_options")
        assert not hasattr(prepared, "predictor")
    finally:
        prepared.stop()


def test_stop_is_bounded_cancels_queued_work_and_does_not_kill_running_thread():
    underlying = Predictor()
    prepared = PreparedPredictor(underlying)
    slow, queued = job("slow"), job("queued")
    prepared.enqueue(slow, 4)
    assert underlying.slow_started.wait(1)
    prepared.enqueue(queued, 4)
    try:
        started = time.monotonic()
        assert prepared.stop(timeout=.02) is False
        assert time.monotonic() - started < .3
        assert prepared.worker_alive
        assert not prepared.available
        with pytest.raises(RuntimeError): prepared.enqueue(job("late"), 4)
        assert any(e["phase"] == "cancel" and e["job_id"] == "queued" for e in prepared.events)
    finally:
        underlying.release.set()
        assert prepared.stop(timeout=1)
    assert not prepared.ready(slow)
    assert prepared.records == {}
    assert underlying.calls == [("slow", [4])]
    assert prepared.events[-1]["status"] == "cancelled"


def test_all_helpers_derive_same_profile_in_both_treatments_and_log_clocks():
    target = job()
    for asynchronous in (False, True):
        underlying = Predictor()
        prepared = PreparedPredictor(RecordingPredictor(underlying, lambda *_: None),
            asynchronous=asynchronous, prepare_on_read=not asynchronous)
        try:
            if asynchronous:
                with pytest.raises(JobPredictionError): prepared.predict_avg_vram_mb(target, 4)
                assert not underlying.calls
                prepared.enqueue(target, 4)
                wait_finished(prepared, target)
            assert prepared.predict_avg_vram_options(target, [4]) == {4: 800.}
            assert prepared.predict_avg_vram_mb(target, 4) == 800.
            assert prepared.predict_busy_percent_options(target, [4]) == {4: 70.}
            assert prepared.predict_busy_indicator_options(target, [4]) == {4: 70.}
            assert prepared.predict_epoch_cost_options(target, [4]) == {4: 6000.}
            with pytest.raises(JobPredictionError): prepared.predict_seconds_per_epoch_options(target, [4])
            assert len(underlying.calls) == 1
            events = prepared.events
            assert [event["phase"] for event in events] == ["enqueue", "start", "end"]
            assert events[0]["monotonic_seconds"] <= events[1]["monotonic_seconds"] <= events[2]["monotonic_seconds"]
            assert events[-1]["wall_seconds"] >= 0
            assert events[-1]["thread_cpu_seconds"] >= 0
            assert events[-1]["process_cpu_seconds"] >= 0
            assert "child processes" in events[-1]["cpu_clock_note"]
        finally:
            prepared.stop()
    with pytest.raises(ValueError): PreparedPredictor(Predictor(), prepare_on_read=True)


class ServiceBase:
    def _runnable_jobs(self): return list(self.jobs)
    def _supervisor_active_job_ids(self): return list(self.active)
    def _dispatch_plan(self, plan):
        self.dispatched.append(plan)
        return True


class Service(PreparationServiceMixin, ServiceBase):
    pass


def test_solo_can_run_during_preparation_but_trials_require_entire_forecast_cohort(monkeypatch):
    first, second, unaccepted = job(), job("second"), job("unaccepted")
    prepared = PreparedPredictor(Predictor(), asynchronous=False)
    prepared.enqueue(first, 4)
    service = Service()
    service.predictor_preparation = prepared
    service.jobs, service.active, service.dispatched = [first, second], [], []
    service.store = SimpleNamespace(get_job=lambda key: next(j for j in service.jobs if j.job_id == key))
    monkeypatch.setattr(module, "forecast_cohort", lambda _: ([first, second], [], {}))
    plan = SimpleNamespace(job_ids=(first.job_id,), batch_overrides={}, trial_metadata={})
    try:
        assert service._runnable_jobs() == [first]
        assert not prepared.ready(unaccepted)
        assert service._dispatch_plan(plan)  # Anchor runs while second is unprepared.
        service.active = ["incumbent"]
        assert not service._dispatch_plan(plan)
        service.active = []
        plan.trial_metadata["requires_live_trial"] = True
        assert not service._dispatch_plan(plan)
        prepared.enqueue(second, 4)
        assert service._dispatch_plan(plan)
        plan.batch_overrides = {first.job_id: 8}
        assert not service._dispatch_plan(plan)
        assert len(service.dispatched) == 2
    finally:
        prepared.stop()
    assert issubclass(AsyncV4Service, PreparationServiceMixin)
    assert issubclass(AsyncRevisedService, PreparationServiceMixin)


def test_normal_service_loop_continues_reaping_and_trial_ticks_during_preparation():
    underlying = Predictor()
    prepared = PreparedPredictor(underlying)
    slow = job("slow")
    prepared.enqueue(slow, 4)
    assert underlying.slow_started.wait(1)
    service = Service()
    service.predictor_preparation = prepared
    service.jobs = [slow]
    service._stop_event = Event()
    service.logger = SimpleNamespace(info=lambda *_: None)
    service.settings = SimpleNamespace(scheduler_poll_interval_seconds=.001)
    service._colocation_trial = object()
    ticks = {"reap": 0, "trial": 0}
    progressed = Event()
    service._write_service_heartbeat = lambda *_: None
    service._poll_active_workers = lambda: ticks.update(reap=ticks["reap"] + 1)
    def trial_tick():
        ticks["trial"] += 1
        if ticks["trial"] >= 3:
            progressed.set()
    service._evaluate_colocation_trial = trial_tick
    for method in ("_release_deferred_resumes", "_process_commands", "_warm_cache", "_poll_telemetry", "_maybe_preempt"):
        setattr(service, method, lambda: None)
    service._dispatch_pending_work = lambda: service._runnable_jobs()
    thread = Thread(target=SchedulerService.run_forever, args=(service,), daemon=True)
    thread.start()
    try:
        assert progressed.wait(.3)
        assert ticks["reap"] >= 3
        assert not prepared.ready(slow)
    finally:
        service._stop_event.set()
        thread.join(1)
        underlying.release.set()
        assert prepared.stop(timeout=1)
    assert not thread.is_alive()


def test_recipe_change_invalidates_same_job_signature_without_including_live_progress():
    target = job()
    target.metadata["perfseer_model"] = {"source_path": "/frozen.py", "steps_per_epoch": 30,
        "workload_contract": {"training": {"optimizer": "adam", "precision": "fp32_ieee"}}}
    prepared = PreparedPredictor(Predictor(), asynchronous=False)
    try:
        prepared.enqueue(target, 4)
        target.metadata.update(runtime_global_step=20, trial_step_window={"count": 8},
            feedback_request={"request_id": "new-arrival"})
        assert prepared.ready(target)
        target.metadata["perfseer_model"]["workload_contract"]["training"]["optimizer"] = "sgd"
        assert not prepared.ready(target)
        with pytest.raises(JobPredictionError): prepared.predict_profile_options(target, [4])
    finally:
        prepared.stop()


def test_frozen_workload_digest_change_invalidates_cache_without_runtime_counts():
    target = job()
    target.metadata["pilot_frozen_workload_sha256"] = "original-frozen-spec"
    prepared = PreparedPredictor(Predictor(), asynchronous=False)
    try:
        prepared.enqueue(target, 4)
        target.metadata.update(runtime_steps_per_epoch=30, runtime_global_step=20)
        assert prepared.ready(target)
        target.metadata["pilot_frozen_workload_sha256"] = "different-frozen-spec"
        assert not prepared.ready(target)
        with pytest.raises(JobPredictionError): prepared.predict_profile_options(target, [4])
    finally:
        prepared.stop()
