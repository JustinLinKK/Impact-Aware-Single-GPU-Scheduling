"""Bounded bracket preparation with process-isolated prediction engines.

Workers load their own frozen CPU predictor lazily after an accepted request.
No neural capture/model object is shared between worker threads or processes.
Only completed immutable profiles cross back to the cache-only scheduler path.
"""
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from copy import deepcopy
import multiprocessing
import math,os
from . import owned_prediction_processes as owned
from queue import Empty
from threading import Lock, Thread
import time

from ..benchmarks.a10_feedback_latency.async_preparation import PreparedPredictor
from ..domain import TrainingJob

_WORKER_ENGINE = None
_WORKER_SPEC = None


def _worker_predict(spec, payload, batches):
    global _WORKER_ENGINE, _WORKER_SPEC
    if _WORKER_ENGINE is None:
        kind, settings_dict, calibration, hardware, gpu_uuid = spec
        if kind == 'dedicated_solo_template_proxy':
            from ..benchmarks.a10_resource_mechanisms.profiles import MeasuredTemplatePredictor
            _WORKER_ENGINE = MeasuredTemplatePredictor(calibration, hardware, gpu_uuid=gpu_uuid)
        elif kind == 'ml_predictor':
            from ..config import SchedulerSettings
            from ..prediction import MLVramPredictor
            from ..benchmarks.a10_feedback_latency.calibration import CalibratedPredictor
            settings = SchedulerSettings.from_dict(settings_dict)
            predictor = MLVramPredictor(settings.prediction, hardware)
            _WORKER_ENGINE = CalibratedPredictor(predictor, calibration, hardware.hardware_key)
        else:
            raise ValueError('unsupported isolated preparation source')
        _WORKER_SPEC = spec
    if _WORKER_SPEC != spec:
        raise ValueError('worker cannot switch a frozen predictor identity')
    job = TrainingJob.from_dict(payload)
    outputs = _WORKER_ENGINE.predict_profile_options(job, batches)
    return outputs, _WORKER_ENGINE.last_sources.get(job.job_id), _WORKER_ENGINE.last_errors.get(job.job_id)


def _pool_initialize():
    os.environ['CUDA_VISIBLE_DEVICES']=''


class IsolatedPredictionBackend:
    uses_predicted_profiles = True
    requires_measured_runtime = True
    requires_branch_memory = False

    def __init__(self, prototype, *, kind, settings_dict, calibration, hardware,
                 gpu_uuid, workers=2, model_init_allowance_seconds=60., timeout_seconds=None, _executor=None, _predict=None):
        if type(workers) is not int or not 1 <= workers <= 2:
            raise ValueError('one or two isolated preparation processes supported')
        from ..config import SchedulerSettings
        conversion=SchedulerSettings.from_dict(settings_dict).prediction.conversion_timeout_seconds
        if (not math.isfinite(model_init_allowance_seconds)or not 0<=model_init_allowance_seconds<=120
                or not math.isfinite(conversion)or not 0<conversion<=300):
            raise ValueError('frozen finite capture/model-init bounds required')
        derived=conversion+model_init_allowance_seconds
        if timeout_seconds is not None and timeout_seconds!=derived:
            raise ValueError('future bound must equal frozen capture timeout plus model-init allowance')
        self.artifact_hash = prototype.artifact_hash
        self.available = prototype.available
        self.unavailable_reason = getattr(prototype, 'unavailable_reason', None)
        self.model_id = getattr(prototype, 'model_id', None)
        self.hardware, self.profile_source = hardware, kind
        self.last_sources, self.last_errors = {}, {}
        self._spec = (kind, deepcopy(settings_dict), deepcopy(calibration), hardware, gpu_uuid)
        self._lock = Lock()
        self._executor = _executor or ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=_pool_initialize)
        self._predict = _predict or _worker_predict
        self.conversion_timeout_seconds=conversion
        self.model_init_allowance_seconds=model_init_allowance_seconds
        self.timeout_seconds=derived
        self.timeout_contract={'conversion_timeout_seconds':conversion,
            'model_init_allowance_seconds':model_init_allowance_seconds,
            'future_seconds_per_single_batch':derived,'capture_timeout_unchanged':True}
        self._known_owned={}
        self._stop_lock=Lock()
        self.cleanup_report=None
        self._closed = False

    def predict_profile_options(self, job, batch_sizes):
        batches = list(batch_sizes)
        with self._lock:
            if self._closed:
                raise RuntimeError('isolated preparation backend stopped')
            future = self._executor.submit(self._predict, self._spec, job.to_dict(), batches)
            owned.remember(self._executor,self._known_owned)
        try:
            deadline=time.monotonic()+self.conversion_timeout_seconds*max(1,len(batches))+self.model_init_allowance_seconds
            while True:
                with self._lock:owned.remember(self._executor,self._known_owned)
                left=deadline-time.monotonic()
                if left<=0:raise FutureTimeout('frozen capture plus explicit model-init allowance exceeded')
                try:
                    outputs,source,error=future.result(timeout=min(.05,left));break
                except FutureTimeout:
                    if future.done():raise

            if any(value.get('artifact_sha256') != self.artifact_hash for value in outputs.values()):
                raise ValueError('isolated predictor artifact changed')
            with self._lock:
                self.last_sources[job.job_id] = source
                if error: self.last_errors[job.job_id] = error
                else: self.last_errors.pop(job.job_id, None)
            return outputs
        except Exception as exc:
            future.cancel()
            if isinstance(exc,FutureTimeout):self.stop()
            with self._lock:
                self.last_sources[job.job_id] = 'failed_preparation'
                self.last_errors[job.job_id] = str(exc)
            raise

    def stop(self, timeout=2.):
        if not math.isfinite(timeout)or timeout<0:raise ValueError('finite nonnegative cleanup bound required')
        until=time.monotonic()+timeout
        if not self._stop_lock.acquire(timeout=timeout):return False
        try:
            if not self._lock.acquire(timeout=max(0.,until-time.monotonic())):return False
            try:
                self._closed=True
                if self.cleanup_report is None:
                    self.cleanup_report=owned.cleanup(self._executor,self._known_owned,timeout=max(0.,until-time.monotonic()))
                elif self.cleanup_report['remaining_live_pids']:
                    # A zero-time stop may sample just after SIGKILL, before the
                    # owned workers become terminal. Refresh that observation;
                    # an old incomplete snapshot is not a permanent live leak.
                    observation=owned.reobserve(self._known_owned,timeout=max(0.,until-time.monotonic()))
                    self.cleanup_report={**self.cleanup_report,
                        'remaining_live_pids':observation['remaining_live_pids'],
                        'reobservations':[*self.cleanup_report.get('reobservations',[]),observation]}
                return not self.cleanup_report['remaining_live_pids']
            finally:self._lock.release()
        finally:self._stop_lock.release()


class ParallelPreparedPredictor(PreparedPredictor):
    """Two preparation consumers; the backing predictor MUST be isolated."""
    def __init__(self, predictor, *, workers=2, **kwargs):
        if type(workers) is not int or not 1 <= workers <= 2:
            raise ValueError('bounded worker count required')
        backend = getattr(predictor, 'predictor', predictor)
        if not isinstance(backend, IsolatedPredictionBackend):
            raise ValueError('parallel capture requires process-isolated prediction backend')
        if kwargs.pop('asynchronous', True) is not True or kwargs.pop('prepare_on_read', False):
            raise ValueError('parallel preparation must stay asynchronous')
        self._workers = []
        self._backend = backend
        super().__init__(predictor, asynchronous=False, prepare_on_read=False, **kwargs)
        self.asynchronous = True
        for i in range(workers):
            thread = Thread(target=self._run, name=f'v5-preparation-{i}', daemon=True)
            self._workers.append(thread)
            thread.start()

    @property
    def worker_alive(self):
        return any(thread.is_alive() for thread in self._workers)

    def stop(self, timeout=3.):
        if not math.isfinite(timeout)or timeout<0:raise ValueError('finite nonnegative shutdown required')
        deadline=time.monotonic()+timeout
        super().stop(timeout=0.)
        # Kill/reap the pool first: cancellation alone cannot unblock an in-flight
        # future. Consumers then receive BrokenProcessPool and can actually join.
        cleaned=self._backend.stop(timeout=max(0.,timeout*.8))
        for thread in self._workers:thread.join(max(0.,deadline-time.monotonic()))
        return bool(cleaned and not self.worker_alive)
