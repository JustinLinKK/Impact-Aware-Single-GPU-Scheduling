"""Research-only CPU preparation separated from scheduler admission.

The asynchronous proxy has no delegation escape hatch: every prediction helper
reads one atomically published profile, and a miss raises JobPredictionError.
Only enqueue may invoke the wrapped RecordingPredictor/CalibratedPredictor.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time

from ...domain import BatchResolution
from ...prediction import JobPredictionError
from ...scheduler.feedback_trials import forecast_cohort
from ...scheduler.service import SchedulerService
from .service import RevisedPilotService


@dataclass(frozen=True)
class PreparationIdentity:
    job_id: str
    job_signature: str
    batch_size: int
    artifact_sha256: str
    hardware: str
    hardware_key: str
    prediction_contract_sha256: str


class PreparedPredictor:
    """Single-worker CPU preparation with cache-only asynchronous reads.

    The explicit synchronous control ``prepare_on_read=True`` instruments first
    reads on the scheduler thread. All helpers derive values from the same
    calibrated profile in both treatments. Asynchronous reads never prepare.
    Running Python threads cannot be killed: stop cancels queued work and waits
    at most its timeout; its boolean result reports whether the worker exited.
    It cannot kill an in-flight capture child: the existing predictor conversion
    deadline and the experiment controller own that process cleanup.
    """
    uses_predicted_profiles = True
    requires_measured_runtime = True
    requires_branch_memory = False

    def __init__(self, predictor, *, asynchronous=True, hardware="A10",
                 hardware_key=None, prepare_on_read=False):
        if asynchronous and prepare_on_read:
            raise ValueError("asynchronous prediction reads must never prepare")
        if not getattr(predictor, "available", False):
            raise ValueError("preparation requires an available predictor")
        if not isinstance(getattr(predictor, "artifact_hash", None), str) or not predictor.artifact_hash:
            raise ValueError("preparation requires an artifact identity")
        self._predictor = predictor
        self.asynchronous = bool(asynchronous)
        self.prepare_on_read = bool(prepare_on_read)
        self.hardware = str(hardware)
        self.hardware_key = str(hardware_key or self._hardware_identity())
        self.model_id = getattr(predictor, "model_id", None)
        self.unavailable_reason = None
        self._lock = Lock()
        self._stop = Event()
        self._queue = Queue()
        self._profiles = {}
        self._states = {}
        self._events = []
        self._failures = {}
        self._last_sources = {}
        self._last_errors = {}
        self._finished = set()
        self._thread = None
        if self.asynchronous:
            self._thread = Thread(target=self._run, name="a10-predictor-preparation", daemon=True)
            self._thread.start()

    def _hardware_identity(self):
        hardware = getattr(self._predictor, "hardware", None)
        return str(getattr(hardware, "hardware_key", None) or getattr(hardware, "gpu_name", None) or self.hardware)

    @property
    def artifact_hash(self):
        return self._predictor.artifact_hash

    @property
    def available(self):
        return bool(getattr(self._predictor, "available", False)) and not self._stop.is_set()

    @staticmethod
    def _contract_digest(job):
        # No filesystem checks on the scheduler thread. The experiment pins
        # source/sidecar bytes separately; this key binds their declared paths,
        # recipe, precision and source hashes without including live progress.
        fields = ("perfseer_model", "architecture_source", "source_path", "source_sha256",
                  "input_shape", "precision", "precision_mode", "uses_amp",
                  "pilot_frozen_workload_sha256")
        contract = {name: job.metadata.get(name) for name in fields}
        contract["baseline_model_path"] = getattr(job, "baseline_model_path", None)
        contract["runner_target"] = getattr(job.config, "runner_target", None)
        contract["loader_target"] = getattr(job.config, "loader_target", None)
        recipe = job.metadata.get("perfseer_model") or {}
        if not isinstance(recipe, dict):
            raise JobPredictionError("perfseer_model must be an object")
        try:
            encoded = json.dumps(contract, sort_keys=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise JobPredictionError("prediction contract is not finite JSON") from exc
        return sha256(encoded.encode()).hexdigest()

    def _identity(self, job, batch_size):
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or int(batch_size) <= 0:
            raise JobPredictionError("preparation requires a positive integral batch size")
        # The live hardware identity is part of the key as well as the configured
        # hardware key, so replacing a predictor's hardware invalidates its cache.
        return PreparationIdentity(str(job.job_id), str(job.packing.signature or job.job_id),
            int(batch_size), self.artifact_hash, self.hardware,
            self.hardware_key + ":" + self._hardware_identity(), self._contract_digest(job))

    def _event(self, phase, identity, **extra):
        self._events.append({"phase": phase, **identity.__dict__,
            "monotonic_seconds": time.monotonic(), **extra})

    @property
    def events(self):
        with self._lock:
            return deepcopy(self._events)

    @property
    def failures(self):
        with self._lock:
            return deepcopy(self._failures)

    @property
    def last_sources(self):
        with self._lock:
            return dict(self._last_sources)

    @property
    def last_errors(self):
        with self._lock:
            return dict(self._last_errors)

    @property
    def preparations_finished(self):
        with self._lock:
            return set(self._finished)

    @property
    def records(self):
        with self._lock:
            records = {}
            for key, profile in self._profiles.items():
                records.setdefault(key.job_id, {})[key.batch_size] = deepcopy(profile)
            return records

    @property
    def worker_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def enqueue(self, job, batch_size):
        """Snapshot accepted work; return False for an already submitted identity."""
        identity = self._identity(job, batch_size)
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("predictor preparation is stopped")
            if identity in self._states:
                return False
        snapshot = deepcopy(job)
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("predictor preparation is stopped")
            if identity in self._states:
                return False
            self._states[identity] = "queued"
            self._event("enqueue", identity)
            if self.asynchronous:
                self._queue.put((identity, snapshot))
        if not self.asynchronous:
            self._prepare(identity, snapshot)
        return True

    def _validate(self, profile, identity):
        if (not isinstance(profile, dict) or profile.get("kind") != "predicted_workload"
                or profile.get("job_signature") != identity.job_signature
                or isinstance(profile.get("batch_size"), bool)
                or profile.get("batch_size") != identity.batch_size
                or profile.get("artifact_sha256") != identity.artifact_sha256
                or profile.get("hardware") != identity.hardware):
            raise JobPredictionError("missing or incompatible prepared profile identity")
        for field in ("step_cost", "epoch_cost", "avg_vram_mib", "peak_vram_mib", "busy_indicator"):
            value = profile.get(field)
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise JobPredictionError("invalid prepared numeric profile: " + field)
            invalid = not 0 <= float(value) <= 100 if field == "busy_indicator" else float(value) <= 0
            if invalid:
                raise JobPredictionError("invalid prepared numeric profile: " + field)
        if float(profile["peak_vram_mib"]) < float(profile["avg_vram_mib"]):
            raise JobPredictionError("prepared peak memory is below average memory")

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=.05)
            except Empty:
                continue
            try:
                self._prepare(*item)
            finally:
                self._queue.task_done()

    def _prepare(self, identity, job):
        with self._lock:
            if self._stop.is_set():
                self._states[identity] = "cancelled"
                self._event("cancel", identity)
                return
            self._states[identity] = "preparing"
            started = time.monotonic()
            thread_started, process_started = time.thread_time(), time.process_time()
            self._event("start", identity)
        profile, failure = None, None
        try:
            outputs = self._predictor.predict_profile_options(job, [identity.batch_size])
            profile = deepcopy(outputs[identity.batch_size])
            self._validate(profile, identity)
            if self._identity(job, identity.batch_size) != identity:
                raise JobPredictionError("predictor identity changed during preparation")
            sources, errors = getattr(self._predictor, "last_sources", {}), getattr(self._predictor, "last_errors", {})
            if sources.get(identity.job_id) != "ml_predictor" or errors.get(identity.job_id):
                raise JobPredictionError("preparation did not use the selected ML predictor")
        except Exception as exc:
            failure = {"type": type(exc).__name__, "error": str(exc)}
        ended = time.monotonic()
        thread_cpu, process_cpu = time.thread_time() - thread_started, time.process_time() - process_started
        with self._lock:
            cancelled = self._stop.is_set()
            status = "cancelled" if cancelled else "failed" if failure else "ready"
            if status == "ready":
                self._profiles[identity] = profile
                self._last_sources[identity.job_id] = "ml_predictor"
                self._last_errors.pop(identity.job_id, None)
                self._failures.pop(identity.job_id, None)
            elif failure:
                self._failures[identity.job_id] = failure
                self._last_errors[identity.job_id] = failure["error"]
                self._last_sources[identity.job_id] = "failed_preparation"
            self._states[identity] = status
            self._finished.add(identity.job_id)
            self._event("end", identity, status=status, started_monotonic_seconds=started,
                ended_monotonic_seconds=ended, wall_seconds=ended-started,
                thread_cpu_seconds=thread_cpu, process_cpu_seconds=process_cpu,
                cpu_clock_note="thread CPU excludes native worker threads and child processes; process CPU includes concurrent parent threads but excludes child processes",
                failure=failure)

    def ready(self, job, batch_size=None):
        batch_size = BatchResolution.resolved_batch_size(job) if batch_size is None else batch_size
        try:
            identity = self._identity(job, batch_size)
        except (ValueError, TypeError, OverflowError, JobPredictionError):
            return False
        with self._lock:
            return not self._stop.is_set() and identity in self._profiles

    def predict_profile_options(self, job, batch_sizes):
        batches = list(batch_sizes)
        if self.prepare_on_read:
            for batch in batches:
                self.enqueue(job, batch)
        identities = [self._identity(job, batch) for batch in batches]
        with self._lock:
            if self._stop.is_set() or any(key not in self._profiles for key in identities):
                raise JobPredictionError("exact profile is not ready; prediction reads cannot prepare")
            return {key.batch_size: deepcopy(self._profiles[key]) for key in identities}

    def _values(self, job, batch_sizes, field):
        return {batch: float(profile[field]) for batch, profile in self.predict_profile_options(job, batch_sizes).items()}

    def predict_avg_vram_options(self, job, batch_sizes):
        return self._values(job, batch_sizes, "avg_vram_mib")

    def predict_avg_vram_mb(self, job, batch_size):
        return self.predict_avg_vram_options(job, [batch_size])[batch_size]

    def predict_busy_indicator_options(self, job, batch_sizes):
        return self._values(job, batch_sizes, "busy_indicator")

    def predict_busy_percent_options(self, job, batch_sizes):
        return self.predict_busy_indicator_options(job, batch_sizes)

    def predict_epoch_cost_options(self, job, batch_sizes):
        return self._values(job, batch_sizes, "epoch_cost")

    def predict_seconds_per_epoch_options(self, job, batch_sizes):
        # Nominal predictor work is never relabeled as measured wall seconds.
        raise JobPredictionError("prepared profiles require measured runtime")

    def stop(self, timeout=.2):
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("stop timeout must be finite and nonnegative")
        with self._lock:
            self._stop.set()
            for key, state in self._states.items():
                if state == "queued":
                    self._states[key] = "cancelled"
                    self._event("cancel", key)
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.worker_alive


class PreparationServiceMixin:
    """Keep normal service ticks responsive while CPU profiles are built."""
    predictor_preparation = None

    def _runnable_jobs(self):
        jobs = super()._runnable_jobs()
        preparation = self.predictor_preparation
        return jobs if preparation is None else [job for job in jobs if preparation.ready(job)]

    def _dispatch_plan(self, plan):
        preparation = self.predictor_preparation
        if preparation is not None:
            for job_id in plan.job_ids:
                job = self.store.get_job(job_id)
                batch = plan.batch_overrides.get(job_id) if plan.batch_overrides else None
                if job is None or not preparation.ready(job, batch):
                    return False
            packed = (plan.trial_metadata.get("requires_live_trial") or len(plan.job_ids) > 1
                      or bool(set(self._supervisor_active_job_ids()) - set(plan.job_ids)))
            if packed:
                try:
                    outstanding, _, _ = forecast_cohort(self)
                    missing = [job.job_id for job in outstanding if not preparation.ready(job)]
                except (ValueError, KeyError, TypeError):
                    return False
                if missing:
                    return False
        return super()._dispatch_plan(plan)

    def stop(self):
        preparation = self.predictor_preparation
        if preparation is not None:
            preparation.stop(timeout=0)
        try:
            return super().stop()
        finally:
            if preparation is not None:
                preparation.stop(timeout=.2)


class AsyncV4Service(PreparationServiceMixin, SchedulerService):
    """Original v4 admission with research preparation isolation."""


class AsyncRevisedService(PreparationServiceMixin, RevisedPilotService):
    """Revised admission with the same research preparation isolation."""
